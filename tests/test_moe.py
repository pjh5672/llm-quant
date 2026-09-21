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
    from llmquant.s4_kernel import quantize_expert_stack

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

    from llmquant.s3_pack import save_packed_model

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
    from llmquant.s3_pack import load_packed_model
    from llmquant.s4_kernel import KernelQuantExperts

    path, ids, before = packed_moe
    model = load_packed_model(path, device="cuda")
    assert isinstance(model.model.layers[0].mlp.experts, KernelQuantExperts)
    with torch.no_grad():
        after = model(ids).logits.float()
    assert torch.equal(before, after)


def test_a_packed_moe_is_smaller_than_the_bf16_model(packed_moe):
    from llmquant.s3_pack import describe_packed

    path, _, _ = packed_moe
    summary = describe_packed(path)
    bf16_bytes = sum(p.numel() * 2 for p in _model(vocab_size=128256).parameters())
    assert summary["bytes"] < bf16_bytes
    # Not a dramatic ratio here, and the reason is the test model rather than the packing:
    # hidden is 256 against a 128k vocab, so the unquantized input embedding alone is half
    # the file. On a real MoE the experts are ~97% of the weights and dominate instead.
    assert summary["bytes"] < bf16_bytes * 0.8


def test_a_packed_moe_holds_a_multi_turn_conversation(packed_moe):
    from llmquant.s5_chat import ChatSession, load_for_chat

    path, _, _ = packed_moe
    model, tokenizer, cache_factory = load_for_chat(packed_path=path)
    chat = ChatSession(model=model, tokenizer=tokenizer, max_new_tokens=8,
                       cache_factory=cache_factory, max_context_tokens=400)
    first = chat.ask("Hello there")
    chat.ask("And again?")
    assert isinstance(first, str)
    assert len(chat.history) == 4      # two turns, both sides recorded
