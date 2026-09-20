"""The study pipeline's new stages: inspect, verify, reconcile, markdown.

All four are pure given their inputs, so they are tested against hand-built rows rather
than a GPU. The cases that matter are the two that cost real time to discover: a tied
lm_head, and a projection that does not survive measurement.
"""

import pytest
import torch.nn as nn

from llmquant.core.config import QuantConfig
from llmquant.core.selection import SelectionConfig
from llmquant.eval.analysis import summarize
from llmquant.eval.inspect import fact_warnings, model_facts, parameter_split
from llmquant.eval.markdown import format_markdown
from llmquant.eval.verify import (
    OUT_OF_KERNEL_AXES,
    reconcile,
    verdict,
    verification_plan,
)


class Config:
    def __init__(self, **kw):
        self.hidden_size = 2048
        self.num_hidden_layers = 2
        self.num_attention_heads = 32
        self.num_key_value_heads = 8
        self.vocab_size = 128256
        self.tie_word_embeddings = False
        self._name_or_path = "test/model"
        self.__dict__.update(kw)


class TinyModel(nn.Module):
    """Enough structure for the inspector: one attention block, one MLP, one head."""

    def __init__(self, **kw):
        super().__init__()
        self.config = Config(**kw)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(2048, 2048, bias=False)
        layer.self_attn.o_proj = nn.Linear(2048, 2048, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.gate_proj = nn.Linear(2048, 8192, bias=False)
        self.model.embed_tokens = nn.Embedding(128256, 2048)
        self.lm_head = nn.Linear(2048, 128256, bias=False)


# ---------------------------------------------------------------- inspect


def test_parameter_split_finds_where_the_weights_live():
    split = parameter_split(TinyModel())
    assert set(split) == {"attn", "mlp", "lm_head", "embedding"}
    assert split["mlp"] == 2048 * 8192


def test_a_tied_head_is_reported_as_growing_the_file():
    """The trap: quantizing a tied lm_head unties it and ADDS a tensor, so disk goes up
    while decode traffic goes down, and the two criteria look like they conflict."""
    facts = model_facts(TinyModel(tie_word_embeddings=True))
    assert facts["tie_word_embeddings"] is True
    notes = " ".join(fact_warnings(facts))
    assert "TIED" in notes and "BIGGER" in notes
    assert "decode" in notes


def test_an_untied_head_gets_no_such_warning():
    notes = " ".join(fact_warnings(model_facts(TinyModel(tie_word_embeddings=False))))
    assert "TIED" not in notes


def test_a_narrow_head_is_reported_as_padding_waste():
    """head_dim 64 into a group of 128 stores two slots per weight, which is what makes
    int8 attention weigh what bf16 weighs."""
    model = TinyModel(num_attention_heads=32, hidden_size=2048)
    model.config.head_dim = 64
    recipe = QuantConfig(attn_weight="int8", mlp_weight="int8").to_modifier()
    facts = model_facts(model, recipe)
    assert facts["head_dim"] == 64
    notes = " ".join(fact_warnings(facts))
    assert "narrower than group_size" in notes
    assert facts["padding"]["attn"]["overhead"] > 1.0


# ---------------------------------------------------------------- verify plan


def _config(**kw):
    from llmquant.core.parser import RunConfig

    return RunConfig(quant=QuantConfig(**kw), tasks=("arc_easy",), ppl=False)


def _winner(**kw):
    base = {"attn_weight": "int8", "mlp_weight": "int8", "head_weight": "bf16",
            "activation": "bf16", "kv_cache": "bf16"}
    return {**base, **kw}


def test_the_plan_is_baseline_plus_winner_when_nothing_runs_outside_the_kernel():
    plan = verification_plan(_config(), _winner())
    assert [label for label, _ in plan] == ["bf16 baseline", "selected"]
    assert plan[0][1].quantize is False
    assert plan[1][1].mode == "kernel"


@pytest.mark.parametrize("axis", OUT_OF_KERNEL_AXES)
def test_each_out_of_kernel_axis_gets_an_ablation(axis):
    plan = verification_plan(_config(), _winner(**{axis: "int8"}))
    assert f"selected, {axis}=bf16" in [label for label, _ in plan]
    ablated = next(c for label, c in plan if label.endswith(f"{axis}=bf16"))
    assert getattr(ablated.quant, axis) == "bf16"


def test_the_plan_carries_the_winners_dtypes_not_the_configs():
    plan = verification_plan(_config(attn_weight="bf16"), _winner(attn_weight="int4"))
    assert next(c for label, c in plan if label == "selected").quant.attn_weight == "int4"


# ---------------------------------------------------------------- reconcile


def _vrow(label, tps, **kw):
    return {"name": label, "verify_label": label, "decode_tps": tps,
            "attn_weight": "int8", "mlp_weight": "int8", "head_weight": "bf16",
            "activation": "bf16", "kv_cache": "bf16", **kw}


def test_a_projection_that_does_not_survive_is_called_out():
    """The real case: an int8 KV cache projected 1.47x and measured 0.65x."""
    rows = [
        _vrow("bf16 baseline", 77.3),
        _vrow("selected", 50.3, kv_cache="int8", decode_speedup_projected=1.47),
        _vrow("selected, kv_cache=bf16", 74.3, kv_cache="int8"),
    ]
    result = reconcile(rows)
    assert result["held_up"] is False
    assert result["measured_decode_speedup"] == pytest.approx(50.3 / 77.3)

    kv = next(a for a in result["attribution"] if a["axis"] == "kv_cache")
    assert kv["costs_decode_pct"] == pytest.approx((1 - 50.3 / 74.3) * 100)

    lines = " ".join(verdict(result))
    assert "does NOT hold up" in lines
    assert "kv_cache=int8 costs 32% of decode" in lines
    assert "outside the weight kernel" in lines


def test_a_projection_that_holds_is_not_called_out():
    rows = [
        _vrow("bf16 baseline", 50.0),
        _vrow("selected", 72.0, decode_speedup_projected=1.45),
    ]
    result = reconcile(rows)
    assert result["held_up"] is True
    assert "does NOT hold up" not in " ".join(verdict(result))


def test_decoding_slower_than_bf16_is_stated_plainly():
    rows = [_vrow("bf16 baseline", 77.3), _vrow("selected", 74.3)]
    assert "decodes SLOWER than bf16" in " ".join(verdict(reconcile(rows)))


def test_reconcile_needs_both_rows():
    assert reconcile([_vrow("selected", 50.0)]) == {}
    assert reconcile([_vrow("bf16 baseline", 50.0, decode_tps=None)]) == {}


# ---------------------------------------------------------------- markdown


@pytest.fixture
def summary():
    def row(attn, acc, quantize=True):
        return {"name": f"{attn}", "quantize": quantize, "attn_weight": attn,
                "mlp_weight": attn, "head_weight": "bf16", "activation": "bf16",
                "kv_cache": "bf16", "group_size": 128, "mean_task_acc": acc,
                "bits_per_element": 8.25, "decode_gb_per_token": 1.0, "size_gb": 1.0}
    rows = [row("bf16", 0.60, quantize=False), row("int8", 0.58), row("int4", 0.50)]
    return summarize(rows, SelectionConfig())


def test_markdown_has_the_sections_and_is_valid_pipe_tables(summary):
    text = format_markdown(summary, model_facts(TinyModel(tie_word_embeddings=True)))
    for heading in ("# Quantization study", "## Model", "## Selected", "## Every combination"):
        assert heading in text
    assert "TIED" in text  # the structural warning survives into the document
    for line in text.splitlines():
        if line.startswith("|") and "---" not in line:
            assert line.endswith("|")


def test_markdown_renders_the_verification_verdict(summary):
    verification = {
        "rows": [_vrow("bf16 baseline", 77.3, latency_mode="kernel"),
                 _vrow("selected", 50.3, latency_mode="kernel",
                       kv_cache="int8", decode_speedup_projected=1.47),
                 _vrow("selected, kv_cache=bf16", 74.3, latency_mode="kernel")],
    }
    verification["reconciliation"] = reconcile(verification["rows"])
    text = format_markdown(summary, None, verification)
    assert "## Verification" in text and "### Verdict" in text
    assert "does NOT hold up" in text


def test_markdown_without_verification_omits_the_section(summary):
    assert "## Verification" not in format_markdown(summary)
