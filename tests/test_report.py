"""The report has to stay readable when a metric is missing, which is the normal case for
rows recorded before that metric existed."""

import pytest

from llmquant.core.selection import SelectionConfig
from llmquant.eval.analysis import summarize
from llmquant.eval.report import format_report


def row(attn, mlp, acc, **extra):
    base = {
        "name": f"{attn}-{mlp}",
        "quantize": attn != "bf16",
        "attn_weight": attn,
        "mlp_weight": mlp,
        "head_weight": "bf16",
        "activation": "bf16",
        "group_size": 128,
        "size_gb": 1.0,
        "bits_per_element": 8.25,
        "decode_gb_per_token": 1.0,
    }
    if acc is not None:
        base["mean_task_acc"] = acc
    return {**base, **extra}


@pytest.fixture
def summary():
    rows = [
        row("bf16", "bf16", 0.60, bits_per_element=16.0, decode_gb_per_token=2.0,
            ttft_ms=100.0, decode_tps=30.0, ppl=10.0,
            task_acc={"arc_easy": 0.70, "openbookqa": 0.50}),
        row("int8", "int8", 0.58, ttft_ms=95.0, decode_tps=31.0, ppl=10.1,
            task_acc={"arc_easy": 0.68, "openbookqa": 0.48}),
    ]
    return summarize(rows, SelectionConfig())


def test_headline_table_carries_every_decision_column(summary):
    text = format_report(summary)
    header = text.splitlines()[0]
    for column in ("mean acc", "dacc%", "BPV", "dec GB/t", "proj", "TTFT ms", "TPS", "score"):
        assert column in header, column


def test_latency_rows_carry_the_fake_mode_warning(summary):
    assert "fake-quant path" in format_report(summary)


def test_per_task_table_and_the_ppl_disagreement_table_are_shown(summary):
    text = format_report(summary)
    assert "accuracy by task" in text and "arc_easy" in text and "openbookqa" in text
    # PPL sits in the headline but its gap against the tasks is called out separately
    assert "how far PPL disagrees" in text
    assert "PPL" in text.splitlines()[0] and "dPPL%" in text.splitlines()[0]


def test_latency_columns_are_absent_when_not_measured(summary):
    for r in summary["results"]:
        r.pop("ttft_ms", None)
        r.pop("decode_tps", None)
    text = format_report(summary)
    header = text.splitlines()[0]
    assert "TTFT" not in header and "TPS" not in header
    assert "fake-quant path" not in text  # nothing measured, so nothing to warn about
    assert "BPV" in header and "dec GB/t" in header


def test_missing_metrics_render_as_blanks_not_crashes():
    rows = [
        {"name": "bf16", "quantize": False, "attn_weight": "bf16", "mlp_weight": "bf16",
         "head_weight": "bf16", "activation": "bf16", "group_size": 128, "ppl": 10.0,
         "size_gb": 2.0},
        {"name": "int8-int8", "quantize": True, "attn_weight": "int8", "mlp_weight": "int8",
         "head_weight": "bf16", "activation": "bf16", "group_size": 128, "ppl": 10.1,
         "size_gb": 1.0},
    ]
    text = format_report(summarize(rows, SelectionConfig()))
    assert "PPL" in text.splitlines()[0]  # falls back to PPL as the accuracy column
    assert "fake-quant path" not in text  # no latency measured, so no warning
    assert "-" in text


def test_weights_are_printed_so_a_ranking_can_be_reproduced(summary):
    text = format_report(summary)
    assert "bpv=2.0" in text and "decode_speed=2.0" in text
