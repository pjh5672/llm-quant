"""Turn measured runs into a decision.

The accuracy number is the generation task suite, not perplexity. PPL is teacher-forced and
understates the damage -- a combination reading +0.11% PPL still changed 12.5% of the
generations -- so it is reported but is not what the criterion optimizes. When a set of rows
has no task scores (a sweep recorded before they existed), the PPL increase is used instead
so old results still analyse.

Everything here is pure: it takes the rows a sweep produced (or a saved sweep.json) and
returns plain dicts, so re-analysing needs no GPU and no model.
"""

import itertools
from collections import defaultdict

DTYPE_AXES = ("attn_weight", "mlp_weight", "head_weight", "activation", "kv_cache")
NUMERIC_AXES = ("group_size",)
AXES = DTYPE_AXES + NUMERIC_AXES

# higher = more precise; used to orient every comparison as "precision lost"
PRECISION = {"bf16": 16, "int8": 8, "int4": 4}


def _quantized(rows):
    return [r for r in rows if r.get("quantize", True)]


def _context(row, exclude):
    return tuple(row.get(a, "bf16") for a in AXES if a != exclude)


def _ordered_values(axis, values):
    if axis in DTYPE_AXES:
        return sorted(values, key=lambda v: -PRECISION[v])  # most precise first
    return sorted(values)  # group_size: finer first


def baseline_row(rows):
    """The unquantized run; every cost is measured against it."""
    unquantized = [r for r in rows if not r.get("quantize", True)]
    if unquantized:
        return unquantized[0]
    all_bf16 = [r for r in rows if all(r.get(a, "bf16") == "bf16" for a in DTYPE_AXES)]
    return (all_bf16 or rows or [None])[0]


def accuracy_cost_pct(row, base):
    """How much accuracy this run gave up, in percent of the baseline. Higher is worse.

    Prefers the generation task suite, which scores tokens the model actually produced.
    Falls back to the PPL increase so sweeps recorded before the tasks existed still rank.
    """
    if base is None:
        return 0.0
    if "mean_task_acc" in row and base.get("mean_task_acc"):
        return (base["mean_task_acc"] - row["mean_task_acc"]) / base["mean_task_acc"] * 100
    if "ppl" in row and base.get("ppl"):
        return (row["ppl"] / base["ppl"] - 1) * 100
    return 0.0


def annotate(rows, selection=None):
    """Attach the derived per-row fields the rest of this module reads."""
    base = baseline_row(rows)
    limit = getattr(selection, "acc_drop_limit_pct", 5.0) if selection else 5.0
    for r in rows:
        r["acc_cost_pct"] = accuracy_cost_pct(r, base)
        r["within_limit"] = limit is None or r["acc_cost_pct"] <= limit
        if base and "ppl" in r and base.get("ppl"):
            r["ppl_increase_pct"] = (r["ppl"] / base["ppl"] - 1) * 100
        if r.get("decode_gb_per_token") is not None:
            # decode re-reads the whole KV cache every step, so its share grows with the
            # context; the weights alone would make a quantized cache look free
            context = getattr(selection, "context_tokens", 2048) if selection else 2048
            r["context_tokens"] = context
            r["decode_gb_at_context"] = r["decode_gb_per_token"] + (
                context / 1024 * r.get("kv_gb_per_1k_context", 0.0)
            )
        if base and base.get("decode_gb_at_context") and r.get("decode_gb_at_context"):
            # decode is memory bound, so throughput tracks the bytes read per token
            r["decode_speedup_projected"] = (
                base["decode_gb_at_context"] / r["decode_gb_at_context"]
            )
        if base and base.get("decode_tps") and r.get("decode_tps"):
            r["decode_speedup_measured"] = r["decode_tps"] / base["decode_tps"]
    return rows


def prefill_is_measurable(row) -> bool:
    """Whether this row's TTFT says anything about a deployment's prefill.

    Under mode="fake" it does not. Fake quant stores dequantized bf16 weights and runs the
    same bf16 GEMM whatever the weight dtype, so prefill compute is identical across the
    grid -- but quantizing the activations and the KV cache still costs real wall time in
    the simulation. The measured TTFT is therefore the *simulation's* overhead, ranking the
    combinations by how much fake work they do rather than by how fast they would serve.

    The phase 1 sweep showed this plainly: TTFT was flat at ~48ms across every weight dtype
    and moved only with activation (~48 -> ~85ms) and KV cache (~48 -> ~59ms) quantization,
    and at weight 2.0 it was enough to rank a strictly worse combination first.
    """
    return row.get("latency_mode") not in ("fake",)


def score_rows(rows, selection):
    """Weighted score in percent-better-than-bf16 units, so the weights are comparable."""
    base = baseline_row(rows)
    weights = {
        "bits_per_element": selection.bpv_weight,
        "decode_gb_at_context": selection.decode_speed_weight,
        "ttft_ms": selection.prefill_speed_weight,
    }
    for r in rows:
        gains, missing = 0.0, []
        for metric, weight in weights.items():
            if not weight:
                continue
            if metric == "ttft_ms" and not prefill_is_measurable(r):
                missing.append("ttft_ms (fake path)")
                continue
            if base is None or not r.get(metric) or not base.get(metric):
                missing.append(metric)
                continue
            gains += (1 - r[metric] / base[metric]) * 100 * weight
        r["score"] = gains - r.get("acc_cost_pct", 0.0) * selection.accuracy_weight
        r["score_missing"] = missing
    return rows


def pareto_front(rows, cost_key="decode_gb_at_context"):
    """Rows nothing else dominates on (cost, accuracy loss)."""
    usable = [r for r in rows if r.get(cost_key) is not None]
    front = []
    for r in usable:
        dominated = any(
            o is not r
            and o[cost_key] <= r[cost_key]
            and o["acc_cost_pct"] <= r["acc_cost_pct"]
            and (o[cost_key] < r[cost_key] or o["acc_cost_pct"] < r["acc_cost_pct"])
            for o in usable
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r[cost_key])


def reference_row(rows):
    """Most precise combination the grid actually contains.

    Not the unquantized baseline: that sits outside the grid, so if every run quantizes both
    attn and mlp, no row differs from it in exactly one axis and the one-axis-at-a-time
    decomposition never starts.
    """
    grid = _quantized(rows)
    if not grid:
        return None
    best = {a: _ordered_values(a, {r.get(a, "bf16") for r in grid})[0] for a in DTYPE_AXES}
    matches = [r for r in grid if all(r.get(a, "bf16") == best[a] for a in DTYPE_AXES)]
    return min(matches, key=lambda r: r["acc_cost_pct"]) if matches else None


def axis_effects(rows):
    """Accuracy cost of changing one axis with every other axis held fixed.

    Averaged over the contexts the sweep happens to contain, so a wide min..max spread is
    itself the signal that the axis interacts with another one.
    """
    effects = {}
    for axis in AXES:
        by_context = defaultdict(dict)
        for r in rows:
            by_context[_context(r, axis)][r.get(axis, "bf16")] = r
        pairs = defaultdict(list)
        for context in by_context.values():
            values = _ordered_values(axis, context)
            for hi, lo in itertools.combinations(values, 2):  # precise -> less precise
                a, b = context[hi], context[lo]
                pairs[(hi, lo)].append(
                    {
                        "acc": b["acc_cost_pct"] - a["acc_cost_pct"],
                        "bpv": (b.get("bits_per_element") or 0) - (a.get("bits_per_element") or 0),
                        "decode": (b.get("decode_gb_at_context") or 0)
                        - (a.get("decode_gb_at_context") or 0),
                    }
                )
        for (hi, lo), samples in sorted(pairs.items()):
            costs = [s["acc"] for s in samples]
            effects[f"{axis}: {hi} -> {lo}"] = {
                "axis": axis,
                "from": hi,
                "to": lo,
                "n": len(samples),
                "acc_cost_mean": sum(costs) / len(costs),
                "acc_cost_min": min(costs),
                "acc_cost_max": max(costs),
                "bpv_delta_mean": sum(s["bpv"] for s in samples) / len(samples),
                "decode_delta_mean": sum(s["decode"] for s in samples) / len(samples),
            }
    return effects


def interactions(rows):
    """Measured loss vs the sum of the single-axis losses, for multi-axis changes.

    A positive residual means the changes hurt more together than apart, so the combination
    cannot be reasoned about one axis at a time.
    """
    ref = reference_row(rows)
    if ref is None or len({r["group_size"] for r in _quantized(rows)}) > 1:
        return []

    singles = {}
    for r in _quantized(rows):
        changed = [a for a in DTYPE_AXES if r.get(a, "bf16") != ref.get(a, "bf16")]
        if len(changed) == 1:
            singles[(changed[0], r.get(changed[0], "bf16"))] = (
                r["acc_cost_pct"] - ref["acc_cost_pct"]
            )

    out = []
    for r in _quantized(rows):
        changed = [a for a in DTYPE_AXES if r.get(a, "bf16") != ref.get(a, "bf16")]
        if len(changed) < 2:
            continue
        parts = [singles.get((a, r.get(a, "bf16"))) for a in changed]
        if any(p is None for p in parts):
            continue
        predicted = sum(parts)
        actual = r["acc_cost_pct"] - ref["acc_cost_pct"]
        out.append(
            {
                "name": r["name"],
                "changed": changed,
                "predicted": predicted,
                "actual": actual,
                "residual": actual - predicted,
            }
        )
    return sorted(out, key=lambda d: -abs(d["residual"]))


def select(rows, selection, near_miss=3):
    """Highest score among the runs inside the accuracy limit, plus the nearest misses."""
    quantized = _quantized(rows)
    within = [r for r in quantized if r["within_limit"]]
    missed = sorted(
        (r for r in quantized if not r["within_limit"]), key=lambda r: r["acc_cost_pct"]
    )
    return {
        "acc_drop_limit_pct": getattr(selection, "acc_drop_limit_pct", 5.0),
        "best": max(within, key=lambda r: r["score"]) if within else None,
        "n_within_limit": len(within),
        "near_misses": missed[:near_miss],
        "ranked": sorted(quantized, key=lambda r: -r["score"])[:8],
    }


def shortlist(summary):
    """Runs worth paying an expensive stage-2 evaluation for: the pareto front plus anything
    inside the limit. The bf16 baseline is the reference, not a candidate."""
    keep = {id(r) for r in summary["pareto_front"]}
    keep |= {id(r) for r in summary["results"] if r.get("within_limit")}
    return [r for r in summary["results"] if id(r) in keep and r.get("quantize", True)]


def summarize(rows, selection):
    annotate(rows, selection)
    score_rows(rows, selection)
    return {
        "n_runs": len(rows),
        "baseline": baseline_row(rows),
        "reference": reference_row(rows),
        "selection": select(rows, selection),
        "pareto_front": pareto_front(rows),
        "axis_effects": axis_effects(rows),
        "interactions": interactions(rows),
        "context_tokens": getattr(selection, "context_tokens", 2048),
        "weights": {
            "accuracy": selection.accuracy_weight,
            "bpv": selection.bpv_weight,
            "decode_speed": selection.decode_speed_weight,
            "prefill_speed": selection.prefill_speed_weight,
        },
        "results": rows,
    }
