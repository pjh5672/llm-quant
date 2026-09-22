"""The analysis is pure, so it is checked against hand-computed numbers, not a real sweep."""

import pytest

from llmquant.core.selection import SelectionConfig
from llmquant.eval.analysis import (
    accuracy_cost_pct,
    annotate,
    axis_effects,
    baseline_row,
    interactions,
    pareto_front,
    reference_row,
    score_rows,
    select,
    shortlist,
    summarize,
)


def row(attn, mlp, acc, ppl, bpv, decode, head="bf16", act="bf16", quantize=True, **extra):
    return {
        "name": f"{attn}-{mlp}-{head}-{act}",
        "quantize": quantize,
        "attn_weight": attn,
        "mlp_weight": mlp,
        "head_weight": head,
        "activation": act,
        "group_size": 128,
        "mean_task_acc": acc,
        "ppl": ppl,
        "bits_per_element": bpv,
        "decode_gb_per_token": decode,
        "size_gb": decode,
        **extra,
    }


@pytest.fixture
def grid():
    # attn int8->int4 costs 10 points of relative accuracy, mlp int8->int4 costs 13.33,
    # and together they cost 26.67 -- more than the sum, so they interact
    return [
        row("bf16", "bf16", 0.60, 10.0, 16.00, 2.0, quantize=False, ttft_ms=100.0, decode_tps=30.0),
        row("int8", "int8", 0.58, 10.1, 8.25, 1.0, ttft_ms=95.0, decode_tps=31.0),
        row("int4", "int8", 0.52, 11.0, 6.25, 0.9),
        row("int8", "int4", 0.50, 12.0, 6.25, 0.6),
        row("int4", "int4", 0.42, 13.5, 4.25, 0.5),
    ]


def test_accuracy_cost_prefers_the_task_suite_over_perplexity(grid):
    base = baseline_row(grid)
    assert base["quantize"] is False
    # 0.58 vs 0.60 is a 3.33% relative drop; the PPL increase would say 1.0%
    assert accuracy_cost_pct(grid[1], base) == pytest.approx(100 / 30)


def test_accuracy_cost_falls_back_to_perplexity_for_older_rows(grid):
    stripped = [{k: v for k, v in r.items() if k != "mean_task_acc"} for r in grid]
    assert accuracy_cost_pct(stripped[1], baseline_row(stripped)) == pytest.approx(1.0)


def test_annotate_applies_the_accuracy_limit_not_a_ppl_limit(grid):
    annotate(grid, SelectionConfig(acc_drop_limit_pct=5.0))
    assert [r["within_limit"] for r in grid] == [True, True, False, False, False]
    annotate(grid, SelectionConfig(acc_drop_limit_pct=None))
    assert all(r["within_limit"] for r in grid)


def test_annotate_projects_decode_speed_from_traffic(grid):
    annotate(grid, SelectionConfig())
    assert grid[1]["decode_speedup_projected"] == pytest.approx(2.0)  # 2.0 -> 1.0 GB/token
    # the measured number is kept separate: on the fake path it barely moves
    assert grid[1]["decode_speedup_measured"] == pytest.approx(31.0 / 30.0)


def test_score_trades_accuracy_against_compression_and_speed(grid):
    def best(weights):
        annotate(grid, weights)
        score_rows(grid, weights)
        # the bf16 baseline scores a flat 0 and is never a candidate; select() drops it too
        return max((r for r in grid if r["quantize"]), key=lambda r: r["score"])["name"]

    assert best(
        SelectionConfig(accuracy_weight=10.0, bpv_weight=0.1, decode_speed_weight=0.1,
                        prefill_speed_weight=0.0, acc_drop_limit_pct=None)
    ) == "int8-int8-bf16-bf16"

    assert best(
        SelectionConfig(accuracy_weight=0.1, bpv_weight=2.0, decode_speed_weight=2.0,
                        prefill_speed_weight=0.0, acc_drop_limit_pct=None)
    ) == "int4-int4-bf16-bf16"


def test_missing_metrics_are_reported_not_silently_zeroed(grid):
    annotate(grid, SelectionConfig())
    bare = [{k: v for k, v in r.items() if k != "bits_per_element"} for r in grid]
    score_rows(bare, SelectionConfig())
    assert all("bits_per_element" in r["score_missing"] for r in bare)


def test_pareto_front_is_cost_against_accuracy_loss(grid):
    dominated = row("int4", "int8", 0.515, 11.1, 6.5, 1.2, head="int8")  # worse on both
    rows = [*grid, dominated]
    annotate(rows, SelectionConfig())
    front = pareto_front(rows)
    assert dominated not in front
    assert [r["decode_gb_per_token"] for r in front] == sorted(
        r["decode_gb_per_token"] for r in front
    )


def test_reference_is_the_most_precise_run_inside_the_grid(grid):
    annotate(grid, SelectionConfig())
    ref = reference_row(grid)
    # not the bf16 baseline: it sits outside the grid, so nothing differs from it in
    # exactly one axis and the decomposition would never start
    assert ref["name"] == "int8-int8-bf16-bf16" and ref["quantize"]


def test_axis_effects_average_over_contexts(grid):
    annotate(grid, SelectionConfig())
    effects = axis_effects(grid)
    attn = effects["attn_weight: int8 -> int4"]
    assert attn["n"] == 2
    assert attn["acc_cost_min"] == pytest.approx(10.0)  # with mlp int8
    assert attn["acc_cost_max"] == pytest.approx(100 / 3 * 0.4)  # with mlp int4, 13.33
    assert attn["bpv_delta_mean"] == pytest.approx(-2.0)


def test_interactions_expose_the_super_additive_residual(grid):
    annotate(grid, SelectionConfig())
    (both,) = interactions(grid)
    assert both["changed"] == ["attn_weight", "mlp_weight"]
    assert both["predicted"] == pytest.approx(10.0 + 100 / 3 * 0.4)
    assert both["actual"] == pytest.approx(30.0 - 100 / 30)
    assert both["residual"] > 0


def test_interactions_are_skipped_when_group_size_varies(grid):
    annotate(grid, SelectionConfig())
    mixed = [*grid, {**row("int4", "int4", 0.45, 13.0, 4.25, 0.5), "group_size": 64}]
    annotate(mixed, SelectionConfig())
    assert interactions(mixed) == []


def test_select_picks_the_best_score_inside_the_limit(grid):
    summary = summarize(grid, SelectionConfig(acc_drop_limit_pct=5.0))
    sel = summary["selection"]
    assert sel["best"]["name"] == "int8-int8-bf16-bf16"
    assert sel["n_within_limit"] == 1
    assert next(m["name"] for m in sel["near_misses"]) == "int4-int8-bf16-bf16"


def test_select_reports_nothing_when_the_limit_is_unreachable(grid):
    strict = SelectionConfig(acc_drop_limit_pct=0.0)
    annotate(grid, strict)
    score_rows(grid, strict)
    assert select(grid, strict)["best"] is None


def test_shortlist_excludes_the_baseline(grid):
    summary = summarize(grid, SelectionConfig())
    picked = shortlist(summary)
    assert picked and all(r.get("quantize", True) for r in picked)


def test_summarize_reports_the_weights_it_used(grid):
    summary = summarize(grid, SelectionConfig(bpv_weight=7.0))
    assert summary["weights"]["bpv"] == 7.0
    assert summary["n_runs"] == 5


def test_fake_path_ttft_is_not_scored():
    """Phase 1 measured TTFT on the fake path, where it tracks how much quant-dequant the
    simulation does rather than how fast the model would serve. Scoring it at weight 2.0
    ranked a combination that was worse on BOTH accuracy and decode traffic first."""
    better = row("int8", "int8", 0.58, 10.1, 8.25, 1.00, act="bf16",
                 ttft_ms=58.9, latency_mode="fake", kv_cache="int8")
    worse = row("int8", "int8", 0.57, 10.1, 8.25, 1.02, act="bf16",
                ttft_ms=48.6, latency_mode="fake", kv_cache="bf16")
    rows = [
        row("bf16", "bf16", 0.60, 10.0, 16.00, 2.0, quantize=False,
            ttft_ms=48.6, latency_mode="fake"),
        better,
        worse,
    ]
    weights = SelectionConfig(acc_drop_limit_pct=None)
    annotate(rows, weights)
    score_rows(rows, weights)

    assert better["score"] > worse["score"]
    assert "ttft_ms (fake path)" in better["score_missing"]


def test_a_real_run_still_scores_its_ttft():
    rows = [
        row("bf16", "bf16", 0.60, 10.0, 16.00, 2.0, quantize=False,
            ttft_ms=100.0, latency_mode="kernel"),
        row("int8", "int8", 0.58, 10.1, 8.25, 1.0, ttft_ms=50.0, latency_mode="kernel"),
    ]
    weights = SelectionConfig(acc_drop_limit_pct=None)
    annotate(rows, weights)
    score_rows(rows, weights)
    assert rows[1]["score_missing"] == []
    # halving TTFT at weight 2.0 is worth 100 points on its own
    assert rows[1]["score"] > 100
