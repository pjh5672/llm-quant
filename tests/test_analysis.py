"""The analysis is pure, so it is checked against hand-computed numbers, not a real sweep."""

import pytest

from llmquant.analysis import (
    axis_effects,
    baseline_ppl,
    format_report,
    interactions,
    pareto_front,
    reference_row,
    select,
    summarize,
)


def row(attn, mlp, ppl, size, head="bf16", act="bf16", quantize=True):
    return {
        "name": f"{attn}-{mlp}-{head}-{act}",
        "quantize": quantize,
        "attn_weight": attn,
        "mlp_weight": mlp,
        "head_weight": head,
        "activation": act,
        "group_size": 128,
        "ppl": ppl,
        "size_gb": size,
    }


@pytest.fixture
def grid():
    # attn int8->int4 costs 0.9 with mlp int8, 1.5 with mlp int4 (they interact)
    # mlp  int8->int4 costs 1.9 with attn int8, 2.5 with attn int4
    return [
        row("bf16", "bf16", 10.0, 2.0, quantize=False),
        row("int8", "int8", 10.1, 1.0),
        row("int4", "int8", 11.0, 0.9),
        row("int8", "int4", 12.0, 0.6),
        row("int4", "int4", 13.5, 0.5),
    ]


def test_baseline_is_the_unquantized_run(grid):
    assert baseline_ppl(grid) == 10.0


def test_reference_is_the_most_precise_quantized_run_not_the_baseline(grid):
    # the bf16 baseline sits outside the grid, so anchoring there would leave no row
    # differing in exactly one axis and the whole decomposition would be empty
    ref = reference_row(grid)
    assert ref["attn_weight"] == "int8" and ref["mlp_weight"] == "int8"
    assert ref["quantize"]


def test_axis_effects_average_over_contexts(grid):
    effects = axis_effects(grid)
    attn = effects["attn_weight: int8 -> int4"]
    assert attn["n"] == 2
    assert attn["ppl_delta_pct_mean"] == pytest.approx((0.9 + 1.5) / 2 / 10.0 * 100)
    assert attn["ppl_delta_pct_min"] == pytest.approx(9.0)
    assert attn["ppl_delta_pct_max"] == pytest.approx(15.0)
    assert attn["size_delta_gb_mean"] == pytest.approx((0.9 - 1.0 + 0.5 - 0.6) / 2)

    mlp = effects["mlp_weight: int8 -> int4"]
    assert mlp["ppl_delta_pct_mean"] == pytest.approx((1.9 + 2.5) / 2 / 10.0 * 100)


def test_interactions_report_the_super_additive_residual(grid):
    (both,) = interactions(grid)
    assert both["changed"] == ["attn_weight", "mlp_weight"]
    assert both["predicted_ppl_delta"] == pytest.approx(0.9 + 1.9)
    assert both["actual_ppl_delta"] == pytest.approx(3.4)
    assert both["residual"] == pytest.approx(0.6)


def test_interactions_are_skipped_when_group_size_varies(grid):
    varied = [*grid, {**row("int4", "int4", 13.0, 0.5), "group_size": 64}]
    assert interactions(varied) == []


def test_pareto_front_drops_dominated_rows(grid):
    dominated = row("int4", "int8", 11.1, 1.2, head="int8")  # bigger and worse than int8/int8
    front = pareto_front([*grid, dominated])
    assert dominated not in front
    assert [r["size_gb"] for r in front] == sorted(r["size_gb"] for r in front)


def test_select_picks_smallest_within_the_limit_and_lists_near_misses(grid):
    sel = select(grid, limit_ratio=1.05)  # limit = 10.5
    assert sel["best"]["name"] == "int8-int8-bf16-bf16"
    assert sel["n_within_limit"] == 1
    assert [m["ppl"] for m in sel["near_misses"]] == [11.0, 12.0, 13.5]


def test_select_reports_nothing_when_the_limit_is_unreachable(grid):
    assert select(grid, limit_ratio=1.0)["best"] is None


def test_summarize_fills_derived_fields_and_formats(grid):
    s = summarize(grid)
    assert s["n_runs"] == 5
    assert all("ppl_increase_pct" in r and "within_limit" in r for r in s["results"])
    text = format_report(s)
    assert "selection" in text and "where the loss comes from" in text
    assert "do the losses add up?" in text


from llmquant.analysis import score_rows, shortlist
from llmquant.args.selection import SelectionConfig


def costed(r, bpv, decode):
    return {**r, "bits_per_element": bpv, "decode_gb_per_token": decode}


@pytest.fixture
def costed_grid(grid):
    costs = {  # (bpv, decode GB/token)
        "bf16-bf16-bf16-bf16": (16.0, 2.0),
        "int8-int8-bf16-bf16": (8.25, 1.0),
        "int4-int8-bf16-bf16": (6.25, 0.9),
        "int8-int4-bf16-bf16": (6.25, 0.6),
        "int4-int4-bf16-bf16": (4.25, 0.5),
    }
    return [costed(r, *costs[r["name"]]) for r in grid]


def test_score_rewards_bpv_and_speed_and_penalizes_accuracy(costed_grid):
    only_accuracy = SelectionConfig(accuracy_weight=1.0, bpv_weight=0.0, decode_speed_weight=0.0)
    score_rows(costed_grid, only_accuracy)
    # with no reward for compression, the least aggressive run wins
    assert max(costed_grid, key=lambda r: r["score"])["name"] == "bf16-bf16-bf16-bf16"

    compression_heavy = SelectionConfig(accuracy_weight=0.1, bpv_weight=2.0, decode_speed_weight=2.0)
    score_rows(costed_grid, compression_heavy)
    assert max(costed_grid, key=lambda r: r["score"])["name"] == "int4-int4-bf16-bf16"


def test_weights_can_flip_the_pick_inside_the_limit(costed_grid):
    # relax the limit so more than one row qualifies
    lenient = 1.4
    by_size = select(costed_grid, lenient)
    assert by_size["best"]["name"] == "int4-int4-bf16-bf16"  # old rule: smallest

    accuracy_first = SelectionConfig(accuracy_weight=10.0, bpv_weight=0.1, decode_speed_weight=0.1)
    weighted = select(costed_grid, lenient, selection=accuracy_first)
    assert weighted["weighted"] and weighted["best"]["name"] == "int8-int8-bf16-bf16"


def test_rows_without_the_new_metrics_still_score(grid):
    score_rows(grid, SelectionConfig())
    assert all("score" in r for r in grid)
    assert all(r["score_missing"] for r in grid)  # reported, not silently zeroed


def test_shortlist_is_pareto_plus_within_limit_minus_baseline(costed_grid):
    s = summarize(costed_grid, 1.05, SelectionConfig())
    picked = shortlist(s)
    assert all(r.get("quantize", True) for r in picked)  # the baseline is the reference, not a candidate
    assert any(r["name"] == "int8-int8-bf16-bf16" for r in picked)  # within limit
    assert len(picked) < len(costed_grid)
