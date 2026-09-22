"""Mixture-of-Experts through every stage: fake, real, kernel, pack, load, chat.

These build a small MixtralForCausalLM rather than mocking one. The weights are random, so
nothing here says anything about accuracy -- but every line of transformers' routing, expert
loop and forward is the real one, and that is what the plumbing has to survive. Mixtral-8x7B
itself does not fit this GPU at any width: 21.7 GB at int4 against 16 GB, and fake quant
would need it in bf16 at 87 GB.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from llmquant.core.config import QuantConfig
from llmquant.core.oneshot import oneshot

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _config(**kw):
    from transformers import MixtralConfig

    base = dict(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                num_attention_heads=8, num_key_value_heads=4, vocab_size=1000,
                num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=512)
    base.update(kw)
    return MixtralConfig(**base)


def _model(**kw):
    from transformers import MixtralForCausalLM

    torch.manual_seed(0)
    return MixtralForCausalLM(_config(**kw)).to(torch.bfloat16).cuda().eval()


def _recipe(mode, mlp="int8"):
    return QuantConfig(attn_weight="int8", mlp_weight=mlp,
                       head_weight="int8").to_modifier(mode=mode)


@pytest.mark.parametrize("mode,expected", [
    ("fake", "MixtralExperts"),          # quantized in place, block untouched
    ("real", "RealQuantExperts"),
    ("kernel", "KernelQuantExperts"),
])
def test_each_mode_puts_the_right_module_in_the_moe_block(mode, expected):
    model = _model()
    oneshot(model, _recipe(mode))
    assert type(model.model.layers[0].mlp.experts).__name__ == expected


def test_fake_quant_changes_the_expert_weights_in_place():
    model = _model()
    before = model.model.layers[0].mlp.experts.gate_up_proj.data.clone()
    oneshot(model, _recipe("fake"))
    after = model.model.layers[0].mlp.experts.gate_up_proj.data
    assert not torch.equal(before, after)
    assert after.shape == before.shape


def test_the_router_is_never_quantized():
    """Its output is argmaxed into a discrete choice of expert, so an error there does not
    perturb a value -- it sends the token somewhere else entirely."""
    for mode in ("fake", "real", "kernel"):
        model = _model()
        before = model.model.layers[0].mlp.gate.weight.data.clone()
        oneshot(model, _recipe(mode))
        gate = model.model.layers[0].mlp.gate
        assert torch.equal(before, gate.weight.data), mode


def test_the_expert_path_matches_the_reference_exactly_at_decode_shape():
    """Quantizing only the experts, the two paths agree bit for bit at decode shape.

    Isolated to the experts on purpose. Quantizing lm_head as well makes the comparison
    differ by about 4e-09, and that is the GEMV kernel's documented reordering -- it scales
    each lane's partial sum and reduces once at the end rather than reducing per group,
    which is the same sum in exact arithmetic and up to an ulp apart in floating point. It
    is shape-dependent, so whole-model equality is not something to assert.
    """
    ids = torch.randint(0, 1000, (1, 1), device="cuda")

    outputs = {}
    for mode in ("real", "kernel"):
        model = _model()
        oneshot(model, QuantConfig(mlp_weight="int8").to_modifier(mode=mode))
        with torch.no_grad():
            outputs[mode] = model(ids).logits.float()
        del model
        torch.cuda.empty_cache()
    assert torch.equal(outputs["real"], outputs["kernel"])


def test_attention_is_exact_too_and_the_head_is_the_one_that_reorders():
    """Which piece is exact and which is an ulp apart, so a future change that breaks the
    exact ones is not written off as the known reordering."""
    ids = torch.randint(0, 1000, (1, 1), device="cuda")

    def compare(**kw):
        outs = {}
        for mode in ("real", "kernel"):
            model = _model()
            oneshot(model, QuantConfig(**kw).to_modifier(mode=mode))
            with torch.no_grad():
                outs[mode] = model(ids).logits.float()
            del model
            torch.cuda.empty_cache()
        return (outs["real"] - outs["kernel"]).abs().max().item()

    assert compare(attn_weight="int8") == 0.0
    assert compare(mlp_weight="int8") == 0.0
    assert compare(head_weight="int8") < 1e-6   # reordered, not wrong


def test_experts_are_quantized_independently():
    """Grouping on the reduction axis of the stack gives every expert its own scales: one
    expert's range must not move another's quantization grid."""
    from llmquant.quantizers import quantize_expert_stack

    args = QuantConfig(mlp_weight="int8").scheme("mlp_weight").weights
    stack = torch.randn(4, 256, 256, device="cuda")
    q, _ = quantize_expert_stack(stack, args)

    loud = stack.clone()
    loud[0] *= 1000
    q_loud, _ = quantize_expert_stack(loud, args)
    assert torch.equal(q[1:], q_loud[1:])


@pytest.fixture
def packed_moe():
    """A packed MoE file, plus the model id it records."""
    from transformers import AutoTokenizer

    from llmquant.packing import save_packed_model

    workdir = Path(tempfile.mkdtemp(prefix="moe-test-"))
    try:
        model = _model(vocab_size=128256)
        model_id = workdir / "tiny-mixtral"
        model.save_pretrained(model_id)
        AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").save_pretrained(model_id)
        model = model.cuda().eval()
        oneshot(model, _recipe("kernel", mlp="int4"))

        ids = torch.randint(0, 1000, (1, 8), device="cuda")
        with torch.no_grad():
            logits = model(ids).logits.float()
        path = save_packed_model(model, workdir / "moe.bin", str(model_id))
        del model
        torch.cuda.empty_cache()
        yield path, ids, logits
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_a_packed_moe_reloads_bit_identically(packed_moe):
    from llmquant.packing import load_packed_model
    from llmquant.quantizers import KernelQuantExperts

    path, ids, before = packed_moe
    model = load_packed_model(path, device="cuda")
    assert isinstance(model.model.layers[0].mlp.experts, KernelQuantExperts)
    with torch.no_grad():
        after = model(ids).logits.float()
    assert torch.equal(before, after)


def test_a_packed_moe_is_smaller_than_the_bf16_model(packed_moe):
    from llmquant.packing import describe_packed

    path, _, _ = packed_moe
    summary = describe_packed(path)
    bf16_bytes = sum(p.numel() * 2 for p in _model(vocab_size=128256).parameters())
    assert summary["bytes"] < bf16_bytes
    # Not a dramatic ratio here, and the reason is the test model rather than the packing:
    # hidden is 256 against a 128k vocab, so the unquantized input embedding alone is half
    # the file. On a real MoE the experts are ~97% of the weights and dominate instead.
    assert summary["bytes"] < bf16_bytes * 0.8


def test_a_packed_moe_holds_a_multi_turn_conversation(packed_moe):
    from llmquant.runtime import ChatSession, load_for_chat

    path, _, _ = packed_moe
    model, tokenizer, cache_factory = load_for_chat(packed_path=path)
    chat = ChatSession(model=model, tokenizer=tokenizer, max_new_tokens=8,
                       cache_factory=cache_factory, max_context_tokens=400)
    first = chat.ask("Hello there")
    chat.ask("And again?")
    assert isinstance(first, str)
    assert len(chat.history) == 4      # two turns, both sides recorded


def test_parameter_split_counts_the_experts():
    """Walking Linear modules alone called OLMoE-1B-7B 57% attention, when attention is
    3.9% of it and the experts are 93%. The note it feeds says where quantizing pays."""
    from llmquant.eval.inspect import parameter_split

    split = parameter_split(_model())
    assert "experts" in split
    assert split["experts"] > split["attn"]


def test_an_expert_block_is_found_by_shape_not_by_class_name():
    """OlmoeExperts is MixtralExperts under another name, and keying off the class name
    left it unswapped: fake quant worked while kernel mode quietly kept 93% of OLMoE in
    bf16 and packed a 1.03x file."""
    import torch.nn as nn

    from llmquant.core.modifier import stacked_expert_blocks

    class NotCalledMixtral(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(4, 512, 256))
            self.down_proj = nn.Parameter(torch.zeros(4, 256, 256))
            self.act_fn = torch.nn.functional.silu

    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].mlp = nn.Module()
    model.layers[0].mlp.experts = NotCalledMixtral()

    blocks = stacked_expert_blocks(model)
    assert [name for name, _ in blocks] == ["layers.0.mlp.experts"]


def test_a_block_holding_expert_weights_but_shaped_differently_fails_loudly():
    import torch.nn as nn

    from llmquant.core.modifier import stacked_expert_blocks

    class Strange(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(4, 512, 256))
            # no down_proj, no act_fn

    # nested, because the pattern anchors on ".experts." and real models always nest
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].mlp = nn.Module()
    model.layers[0].mlp.experts = Strange()
    with pytest.raises(NotImplementedError, match="missing"):
        stacked_expert_blocks(model)


# ---------------------------------------------------------------- batched expert kernel

def _expert_block(experts=8, hidden=256, inter=128, top_k=4, bits="int8"):
    from transformers import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralExperts

    from llmquant.quantizers import KernelQuantExperts

    cfg = MixtralConfig(hidden_size=hidden, intermediate_size=inter,
                        num_local_experts=experts, num_experts_per_tok=top_k,
                        num_hidden_layers=1, num_attention_heads=8,
                        num_key_value_heads=8, vocab_size=1000)
    torch.manual_seed(0)
    ref = MixtralExperts(cfg).cuda().to(torch.bfloat16).eval()
    with torch.no_grad():
        ref.gate_up_proj.normal_(0, 0.02)
        ref.down_proj.normal_(0, 0.02)
    quant = KernelQuantExperts.from_experts(
        ref, QuantConfig(mlp_weight=bits).scheme("mlp_weight")
    ).cuda().eval()
    return ref, quant, cfg


def _routing(tokens, experts, top_k):
    index = torch.stack([torch.randperm(experts, device="cuda")[:top_k] for _ in range(tokens)])
    weights = torch.rand(tokens, top_k, device="cuda", dtype=torch.bfloat16)
    return index, weights


def test_the_batched_kernel_is_bit_exact_per_row_against_the_loop():
    """Row for row the two paths agree exactly. They can still differ once the rows are
    added back together, because index_add_ accumulates duplicate indices with atomics in
    no fixed order -- which the bf16 implementation does too."""
    ref, quant, cfg = _expert_block()
    tokens = 4
    hidden = torch.randn(tokens, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    index, _ = _routing(tokens, cfg.num_local_experts, cfg.num_experts_per_tok)

    with torch.no_grad():
        order, plan = quant._plan(index)
        rows = hidden[order // cfg.num_experts_per_tok]
        batched = quant._batched(rows, quant.qgate_up, quant.sgate_up, plan,
                                 quant.hidden_dim, quant._gate_up_padded)

        flat = index.reshape(-1)
        looped = torch.empty_like(batched)
        for expert in flat.unique().tolist():
            sel = (flat[order] == expert).nonzero().flatten()
            looped[sel] = quant._matmul(
                hidden[(order // cfg.num_experts_per_tok)[sel]],
                quant.qgate_up[expert], quant.sgate_up[expert],
                quant.hidden_dim, quant._gate_up_padded,
            )
    assert torch.equal(batched, looped)


def test_a_block_never_straddles_two_experts():
    """A block keeps its slice of the weight in registers across its rows; reloading
    halfway through would give up the reuse the batching is for."""
    _, quant, cfg = _expert_block()
    index, _ = _routing(6, cfg.num_local_experts, cfg.num_experts_per_tok)
    order, (block_expert, block_row0, block_rows) = quant._plan(index)

    experts_of_row = index.reshape(-1)[order]
    for expert, row0, count in zip(block_expert.tolist(), block_row0.tolist(),
                                   block_rows.tolist()):
        assert count <= quant._batched_max_rows
        if count:   # the plan pads every expert to the same number of slots
            assert (experts_of_row[row0:row0 + count] == expert).all()


def test_every_row_is_covered_exactly_once():
    _, quant, cfg = _expert_block()
    index, _ = _routing(6, cfg.num_local_experts, cfg.num_experts_per_tok)
    _, (_, block_row0, block_rows) = quant._plan(index)

    covered = []
    for row0, count in zip(block_row0.tolist(), block_rows.tolist()):
        covered.extend(range(row0, row0 + count))   # empty slots contribute nothing
    assert sorted(covered) == list(range(index.numel()))


def test_wide_batches_fall_back_to_the_per_expert_path():
    """Above the row limit each block would reread the weight per row, so the batched
    kernel stops paying and cuBLAS runs instead."""
    ref, quant, cfg = _expert_block(experts=2, top_k=2)
    tokens = 64   # 64 tokens x top-2 over 2 experts is far past the limit
    hidden = torch.randn(tokens, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    index, weights = _routing(tokens, cfg.num_local_experts, cfg.num_experts_per_tok)

    # the fallback is decided from the shape: an expert cannot get more rows than there
    # are tokens, so more tokens than the row limit is what sends it to the looped path
    assert tokens > quant._batched_max_rows
    with torch.no_grad():
        out = quant(hidden, index, weights)
    assert out.shape == hidden.shape and torch.isfinite(out).all()


def test_the_batched_result_tracks_bf16():
    ref, quant, cfg = _expert_block()
    tokens = 2
    hidden = torch.randn(tokens, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    index, weights = _routing(tokens, cfg.num_local_experts, cfg.num_experts_per_tok)
    with torch.no_grad():
        got = quant(hidden, index, weights).float()
        want = ref(hidden, index, weights).float()
    assert (got - want).abs().max() < want.abs().max() * 0.1


def test_the_expert_block_can_be_captured_in_a_cuda_graph():
    """torch.bincount reads its maximum back to the host to size the output, even with
    minlength, and that one round trip was the only thing in the block a graph could not
    capture. transformers' own MoE block still cannot be captured -- it loops in Python
    over a .nonzero() -- so this is a property of the batched path, not of MoE."""
    from transformers.models.mixtral.modeling_mixtral import MixtralExperts

    ref, quant, cfg = _expert_block()
    assert isinstance(ref, MixtralExperts)
    hidden = torch.randn(1, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    index, weights = _routing(1, cfg.num_local_experts, cfg.num_experts_per_tok)

    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                quant(hidden, index, weights)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = quant(hidden, index, weights)
        graph.replay()
        torch.cuda.synchronize()
    assert torch.isfinite(captured).all()


def test_the_plan_needs_no_host_round_trip():
    """Every tensor it produces stays on the device; reading one back is what broke
    capture and what cost 0.432 ms of an 0.820 ms block before that."""
    _, quant, cfg = _expert_block()
    index, _ = _routing(4, cfg.num_local_experts, cfg.num_experts_per_tok)
    order, plan = quant._plan(index)
    assert order.is_cuda
    assert all(part.is_cuda for part in plan)
