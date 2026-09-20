"""Phase 4: replaying the decode step instead of relaunching it.

The parts that do not need a GPU are the guards -- when graphing is refused, and that the
verify stage turns it on for the baseline too. The rest needs the real stack, because what
can go wrong here is a captured graph silently answering for the state it was captured at,
and only real logits show that.
"""

import pytest
import torch

from llmquant.core.config import QuantConfig
from llmquant.core.parser import RunConfig
from llmquant.eval.verify import verification_plan
from llmquant.s4_kernel import can_graph


class CpuModel:
    def parameters(self):
        yield torch.zeros(1)


def test_a_quantized_kv_cache_is_refused():
    """A quantized cache is not a StaticCache, so it cannot be captured."""
    assert can_graph(CpuModel(), cache_factory=object) is False


def test_a_cpu_model_is_refused():
    assert can_graph(CpuModel()) is False


def test_a_model_with_no_parameters_is_refused():
    class Empty:
        def parameters(self):
            return iter(())

    assert can_graph(Empty()) is False


def _winner(**kw):
    base = {"attn_weight": "int8", "mlp_weight": "int8", "head_weight": "int8",
            "activation": "bf16", "kv_cache": "bf16"}
    return {**base, **kw}


def test_verify_graphs_the_baseline_too():
    """Graphing only the quantized side would credit quantization with the graph's win."""
    config = RunConfig(quant=QuantConfig(), tasks=("arc_easy",), ppl=False)
    plan = verification_plan(config, _winner())
    assert plan, "expected at least a baseline and the selected row"
    for label, run in plan:
        assert run.cuda_graph is True, label
    assert any(not run.quantize for _, run in plan), "the baseline must be in the plan"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_graphed_generation_matches_eager_token_for_token():
    from llmquant.core.config import ModelArgs
    from llmquant.core.model import load_pretrained
    from llmquant.core.oneshot import oneshot
    from llmquant.s4_kernel import graphed_generate

    torch.manual_seed(0)
    model, tok = load_pretrained(ModelArgs())
    oneshot(model, QuantConfig(attn_weight="int8", mlp_weight="int8",
                               head_weight="int8").to_modifier(mode="kernel"))
    model.eval()

    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    n = 16
    with torch.no_grad():
        eager = model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                               pad_token_id=tok.eos_token_id)
    graphed = graphed_generate(model, ids, n)
    assert graphed is not None
    assert torch.equal(eager, graphed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_the_cache_is_sized_for_the_calls_not_just_the_tokens():
    """A StaticLayer writes at its own cumulative_length and advances once per call, so the
    warmup and capture forwards consume slots. Sizing for tokens alone overran the cache."""
    from llmquant.core.config import ModelArgs
    from llmquant.core.model import load_pretrained
    from llmquant.s4_kernel import WARMUP_CALLS, GraphedDecoder

    model, _ = load_pretrained(ModelArgs())
    d = GraphedDecoder(model, prompt_tokens=8, new_tokens=4)
    assert d.max_len >= 8 + 4 + WARMUP_CALLS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_replayed_logits_are_bit_identical_to_eager():
    """Matching tokens only proves the argmax survived; a replay could be drifting well
    inside it. This compares the logits themselves, every step, as both sides advance --
    the state that a capture could have baked in is the position, so it has to move."""
    from transformers import StaticCache

    from llmquant.core.config import ModelArgs
    from llmquant.core.model import load_pretrained
    from llmquant.core.oneshot import oneshot
    from llmquant.s4_kernel import GraphedDecoder

    torch.manual_seed(0)
    model, tok = load_pretrained(ModelArgs())
    oneshot(model, QuantConfig(attn_weight="int8", mlp_weight="int8",
                               head_weight="int8").to_modifier(mode="kernel"))
    model.eval()

    ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
    prompt, steps = ids.shape[1], 8

    decoder = GraphedDecoder(model, prompt, steps)
    nxt = decoder.prefill(ids)
    decoder.token.copy_(nxt.reshape(1, 1))
    decoder.capture(prompt)

    # an independent cache driven eagerly, sized and prefilled the same way
    eager_cache = StaticCache(config=model.config, max_batch_size=1,
                              max_cache_len=decoder.max_len, device="cuda",
                              dtype=torch.bfloat16)
    with torch.no_grad():
        model(ids, past_key_values=eager_cache, use_cache=True,
              cache_position=torch.arange(prompt, device="cuda"))

    token = nxt
    for i in range(steps):
        decoder.step(token, prompt + i)
        replayed = decoder.logits.clone()
        with torch.no_grad():
            want = model(token.reshape(1, 1), past_key_values=eager_cache, use_cache=True,
                         cache_position=torch.tensor([prompt + i], device="cuda")).logits
        assert torch.equal(replayed, want), (
            f"step {i}: max|d| = {(replayed.float() - want.float()).abs().max().item():.3e}"
        )
        token = want[:, -1:].argmax(-1)
