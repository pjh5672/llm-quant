"""Turn sweep rows into the comparisons that actually drive the bit-combination decision.

Everything here is pure: it takes the rows a sweep produced (or a saved sweep.json) and
returns plain dicts, so re-analysing an old run needs no GPU and no model.

What it answers, in the order the questions come up:
  1. which combinations are even on the table          -> pareto_front
  2. where does the loss come from                     -> axis_effects
  3. do the losses just add up, or do they interact    -> interactions
  4. what does the selection criterion pick            -> select
"""

import itertools
from collections import defaultdict

DTYPE_AXES = ("attn_weight", "mlp_weight", "head_weight", "activation")
NUMERIC_AXES = ("group_size",)
AXES = DTYPE_AXES + NUMERIC_AXES

# higher = more precise; used to orient every comparison as "precision lost"
PRECISION = {"bf16": 16, "int8": 8, "int4": 4}


def _context(row, exclude):
    return tuple(row[a] for a in AXES if a != exclude)


def _ordered_values(axis, values):
    if axis in DTYPE_AXES:
        return sorted(values, key=lambda v: -PRECISION[v])  # most precise first
    return sorted(values)  # group_size: finer first


def _quantized(rows):
    return [r for r in rows if r.get("quantize", True)]


def baseline_row(rows):
    """The unquantized run; every PPL increase is measured against it."""
    unquantized = [r for r in rows if not r.get("quantize", True)]
    if unquantized:
        return min(unquantized, key=lambda r: r["ppl"])
    all_bf16 = [r for r in rows if all(r[a] == "bf16" for a in DTYPE_AXES)]
    return min(all_bf16 or rows, key=lambda r: r["ppl"]) if rows else None


def baseline_ppl(rows):
    row = baseline_row(rows)
    return row["ppl"] if row else min(r["ppl"] for r in rows)


def reference_row(rows):
    """Most precise combination the grid actually contains.

    Not the unquantized baseline: that one sits outside the grid, so if every run quantizes
    (say) both attn and mlp, no row differs from it in exactly one axis and the
    one-axis-at-a-time chain never starts. Anchoring inside the grid keeps it well defined.
    """
    grid = _quantized(rows)
    if not grid:
        return None
    best = {a: _ordered_values(a, {r[a] for r in grid})[0] for a in DTYPE_AXES}
    matches = [r for r in grid if all(r[a] == best[a] for a in DTYPE_AXES)]
    return min(matches, key=lambda r: r["ppl"]) if matches else None


def pareto_front(rows):
    """Rows nothing else dominates: no other run is both no larger and no worse in PPL."""
    front = []
    for r in rows:
        dominated = any(
            o is not r
            and o["size_gb"] <= r["size_gb"]
            and o["ppl"] <= r["ppl"]
            and (o["size_gb"] < r["size_gb"] or o["ppl"] < r["ppl"])
            for o in rows
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r["size_gb"])


def axis_effects(rows, base=None):
    """Cost of changing one axis with every other axis held fixed.

    Averaged over all the contexts the sweep happens to contain, so a wide spread between
    min and max is itself the signal that the axis interacts with something else.
    """
    base = base or baseline_ppl(rows)
    effects = {}
    for axis in AXES:
        by_context = defaultdict(dict)
        for r in rows:
            by_context[_context(r, axis)][r[axis]] = r
        pairs = defaultdict(list)
        for context in by_context.values():
            values = _ordered_values(axis, context)
            for hi, lo in itertools.combinations(values, 2):  # precise -> less precise
                a, b = context[hi], context[lo]
                pairs[(hi, lo)].append(
                    {
                        "ppl_delta": b["ppl"] - a["ppl"],
                        "ppl_delta_pct": (b["ppl"] - a["ppl"]) / base * 100,
                        "size_delta_gb": b["size_gb"] - a["size_gb"],
                    }
                )
        for (hi, lo), samples in sorted(pairs.items()):
            pcts = [s["ppl_delta_pct"] for s in samples]
            effects[f"{axis}: {hi} -> {lo}"] = {
                "axis": axis,
                "from": hi,
                "to": lo,
                "n": len(samples),
                "ppl_delta_pct_mean": sum(pcts) / len(pcts),
                "ppl_delta_pct_min": min(pcts),
                "ppl_delta_pct_max": max(pcts),
                "size_delta_gb_mean": sum(s["size_delta_gb"] for s in samples) / len(samples),
            }
    return effects


def interactions(rows):
    """Measured loss vs the sum of the single-axis losses, for multi-axis changes.

    A positive residual means the changes hurt more together than apart, so the combination
    cannot be reasoned about one axis at a time.
    """
    ref = reference_row(rows)
    if ref is None or len({r["group_size"] for r in _quantized(rows)}) > 1:
        return []  # a varying group_size makes "one axis at a time" ill-defined

    singles = {}
    for r in _quantized(rows):
        changed = [a for a in DTYPE_AXES if r[a] != ref[a]]
        if len(changed) == 1:
            singles[(changed[0], r[changed[0]])] = r["ppl"] - ref["ppl"]

    out = []
    for r in _quantized(rows):
        changed = [a for a in DTYPE_AXES if r[a] != ref[a]]
        if len(changed) < 2:
            continue
        parts = [singles.get((a, r[a])) for a in changed]
        if any(p is None for p in parts):
            continue
        predicted = sum(parts)
        actual = r["ppl"] - ref["ppl"]
        out.append(
            {
                "name": r["name"],
                "changed": changed,
                "predicted_ppl_delta": predicted,
                "actual_ppl_delta": actual,
                "residual": actual - predicted,
                "residual_pct_of_actual": (actual - predicted) / actual * 100 if actual else 0.0,
            }
        )
    return sorted(out, key=lambda d: -abs(d["residual"]))


def summarize(rows, limit_ratio=1.05, selection=None):
    base = baseline_ppl(rows)
    for r in rows:
        r.setdefault("ppl_increase_pct", (r["ppl"] / base - 1) * 100)
        r.setdefault("within_limit", r["ppl"] <= base * limit_ratio)
    return {
        "n_runs": len(rows),
        "reference": reference_row(rows),
        "selection": select(rows, limit_ratio, selection=selection),
        "pareto_front": pareto_front(rows),
        "axis_effects": axis_effects(rows, base),
        "interactions": interactions(rows),
        "results": rows,
    }


def _row_label(row):
    return " ".join(f"{row[a]:<5}" for a in DTYPE_AXES)


def _fmt(value, spec, blank="-"):
    return blank.rjust(len(f"{0:{spec}}")) if value is None else f"{value:{spec}}"


def format_report(summary) -> str:
    """Human-readable version of summarize(), in the order the questions come up."""
    rows = summary["results"]
    sel = summary["selection"]
    header = " ".join(
        f"{a.replace('_weight', '').replace('activation', 'act'):<5}" for a in DTYPE_AXES
    )
    out = []

    out.append(
        f"{'':<5}{header}  {'PPL':>9} {'dPPL%':>8} {'BPV':>6} {'dec GB/t':>9} "
        f"{'disk GB':>8}  lim"
    )
    out.append("-" * 80)
    for i, r in enumerate(sorted(rows, key=lambda r: r["ppl"]), 1):
        mark = "*" if r is sel["best"] else " "
        out.append(
            f"{mark}{i:>3} {_row_label(r)}  {r['ppl']:>9.4f} {r['ppl_increase_pct']:>+8.2f} "
            f"{_fmt(r.get('bits_per_element'), '6.2f')} "
            f"{_fmt(r.get('decode_gb_per_token'), '9.4f')} "
            f"{r['size_gb']:>8.3f}  {'ok' if r['within_limit'] else 'X'}"
        )

    generation = [r for r in rows if "lambada_acc" in r]
    if generation:
        task = next((r["task"] for r in generation if r.get("task")), "task")
        out.append("")
        out.append("== generation (stage 2, shortlist only) ==")
        out.append(
            f"{'':<5}{header}  {task:>9} {'LAMBADA':>8} {'agree':>7} {'exact':>7} {'1st div':>8}"
        )
        for r in sorted(generation, key=lambda r: r["ppl"]):
            out.append(
                f"{'':<5}{_row_label(r)}  {_fmt(r.get('task_acc'), '9.4f')} "
                f"{r['lambada_acc']:>8.4f} "
                f"{_fmt(r.get('generation_agreement'), '7.4f')} "
                f"{_fmt(r.get('generation_exact_match'), '7.4f')} "
                f"{_fmt(r.get('generation_first_divergence'), '8.4f')}"
            )
        out.append(f"  {task} = generated answers matched against the known ones (absolute)")
        out.append("  agree = token agreement with the bf16 model's greedy output (relative);")
        out.append("  1st div = mean position of the first divergence, as a fraction of the output")

    out.append("")
    limit = f"PPL <= {sel['ppl_limit']:.4f}" if sel["limit_ratio"] else "no PPL limit"
    out.append(f"== selection ({limit}, baseline {sel['baseline_ppl']:.4f}) ==")
    if sel["best"]:
        b = sel["best"]
        out.append(
            f"  best: {b['name']}  PPL {b['ppl']:.4f} ({b['ppl_increase_pct']:+.2f}%)  "
            f"{b['size_gb']:.3f} GB"
        )
        out.append(f"  {sel['n_within_limit']} of {summary['n_runs']} runs are within the limit")
    else:
        out.append("  nothing is within the limit")
    for m in sel["near_misses"]:
        out.append(f"  near miss: {m['name']}  {m['ppl_increase_pct']:+.2f}%  {m['size_gb']:.3f} GB")

    if sel.get("weighted") and sel.get("ranked"):
        out.append("")
        out.append("== weighted ranking (higher is better) ==")
        out.append(f"  {'score':>8}  {'PPL%':>7} {'BPV':>6} {'dec GB/t':>9}  name")
        for r in sel["ranked"]:
            out.append(
                f"  {r['score']:>8.2f}  {r['ppl_increase_pct']:>+7.2f} "
                f"{_fmt(r.get('bits_per_element'), '6.2f')} "
                f"{_fmt(r.get('decode_gb_per_token'), '9.4f')}  {r['name']}"
            )
        if any(r.get("score_missing") for r in sel["ranked"]):
            out.append("  (rows missing a metric scored on what they had)")

    out.append("")
    out.append("== pareto front (nothing is both smaller and more accurate) ==")
    for r in summary["pareto_front"]:
        out.append(
            f"  {r['size_gb']:>7.3f} GB  {r['ppl']:>9.4f}  {r['ppl_increase_pct']:>+7.2f}%  {r['name']}"
        )

    out.append("")
    out.append("== where the loss comes from (one axis changed, everything else fixed) ==")
    out.append(f"  {'change':<32} {'n':>3} {'dPPL% mean':>11} {'[min..max]':>18} {'dsize GB':>9}")
    for label, e in sorted(
        summary["axis_effects"].items(), key=lambda kv: -abs(kv[1]["ppl_delta_pct_mean"])
    ):
        span = f"[{e['ppl_delta_pct_min']:+.2f}..{e['ppl_delta_pct_max']:+.2f}]"
        out.append(
            f"  {label:<32} {e['n']:>3} {e['ppl_delta_pct_mean']:>+11.2f} {span:>18} "
            f"{e['size_delta_gb_mean']:>+9.3f}"
        )

    if summary["interactions"]:
        ref = summary["reference"]
        out.append("")
        out.append(f"== do the losses add up? (vs {ref['name']}) ==")
        out.append("  residual > 0 means the changes hurt more together than apart")
        out.append(
            f"  {'combination':<22} {'predicted':>10} {'actual':>10} {'residual':>10} {'of actual':>10}"
        )
        for d in summary["interactions"][:6]:
            out.append(
                f"  {'+'.join(a.replace('_weight', '') for a in d['changed']):<22} "
                f"{d['predicted_ppl_delta']:>10.4f} {d['actual_ppl_delta']:>10.4f} "
                f"{d['residual']:>+10.4f} {d['residual_pct_of_actual']:>+9.1f}%"
            )
    return "\n".join(out)


def shortlist(summary):
    """Runs worth paying the stage-2 generation evaluation for: pareto front + anything
    inside the accuracy limit. The bf16 baseline is handled separately as the reference."""
    keep = {id(r) for r in summary["pareto_front"]}
    keep |= {id(r) for r in summary["results"] if r.get("within_limit")}
    return [r for r in summary["results"] if id(r) in keep and r.get("quantize", True)]


SCORE_METRICS = ("bits_per_element", "decode_gb_per_token")


def score_rows(rows, selection):
    """Weighted score in 'percent better than bf16' units, so the weights are comparable.

    accuracy is a cost (PPL increase), bpv and decode traffic are gains (reduction vs bf16).
    Rows measured before these metrics existed simply score on what they have.
    """
    base = baseline_row(rows)
    base_ppl = base["ppl"] if base else None
    for r in rows:
        gains, missing = {}, []
        for metric, weight in (
            ("bits_per_element", selection.bpv_weight),
            ("decode_gb_per_token", selection.decode_speed_weight),
        ):
            if base is None or metric not in r or metric not in base or not base[metric]:
                missing.append(metric)
                continue
            gains[metric] = (1 - r[metric] / base[metric]) * 100 * weight
        # never let a missing accuracy number score as "free": that would hand the win to
        # the most aggressive combination every time
        if "ppl_increase_pct" in r:
            accuracy_cost = r["ppl_increase_pct"]
        elif base_ppl:
            accuracy_cost = (r["ppl"] / base_ppl - 1) * 100
        else:
            accuracy_cost = 0.0
            missing.append("ppl_increase_pct")
        cost = accuracy_cost * selection.accuracy_weight
        # an absolute task score beats a divergence rate: agreement says the output
        # changed, the task says whether it got worse
        if base is not None and "task_acc" in r and "task_acc" in base:
            cost += (base["task_acc"] - r["task_acc"]) * 100 * selection.generation_weight
        elif "generation_agreement" in r:
            cost += (1 - r["generation_agreement"]) * 100 * selection.generation_weight
        r["score"] = sum(gains.values()) - cost
        r["score_missing"] = missing
    return rows


def select(rows, limit_ratio=1.05, near_miss=3, selection=None):
    """Highest score among the runs that satisfy the accuracy limit, plus the nearest misses.

    With no SelectionConfig this falls back to the original rule -- smallest model inside the
    limit -- so old sweeps keep reproducing the same pick.
    """
    base = baseline_ppl(rows)
    limit = base * limit_ratio if limit_ratio else float("inf")
    quantized = _quantized(rows)
    within = [r for r in quantized if r["ppl"] <= limit]
    missed = sorted((r for r in quantized if r["ppl"] > limit), key=lambda r: r["ppl"])

    if selection is not None:
        score_rows(rows, selection)
        best = max(within, key=lambda r: r["score"]) if within else None
        ranked = sorted(quantized, key=lambda r: -r["score"])
    else:
        best = min(within, key=lambda r: (r["size_gb"], r["ppl"])) if within else None
        ranked = []
    return {
        "baseline_ppl": base,
        "limit_ratio": limit_ratio,
        "ppl_limit": limit,
        "best": best,
        "n_within_limit": len(within),
        "near_misses": missed[:near_miss],
        "ranked": ranked[:8],
        "weighted": selection is not None,
    }
