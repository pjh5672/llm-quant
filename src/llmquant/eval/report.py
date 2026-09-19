"""Render a summarized sweep as tables.

Kept apart from llmquant.eval.analysis so the computation stays testable without going
through string formatting, and so the layout can change without touching the numbers.

The headline table puts accuracy, compression and both speed regimes side by side, because
the decision needs all of them at once: a combination can be cheap in bits, neutral in
decode traffic and still lose accuracy, and each of those lives in a different column.
"""

DTYPE_AXES = ("attn_weight", "mlp_weight", "head_weight", "activation")

FAKE_MODE_WARNING = (
    "  note: TTFT/TPS come from the fake-quant path, where weights are stored dequantized in\n"
    "  bf16 -- every combination moves the same bytes, so measured decode speed barely moves\n"
    "  and A8 looks slower for its extra activation quant. `proj` is the analytic estimate\n"
    "  from decode traffic and is the number to trust until a real stage exists."
)


def _fmt(value, spec, blank="-"):
    return blank.rjust(len(f"{0:{spec}}")) if value is None else f"{value:{spec}}"


def _label(row):
    return " ".join(f"{row[a]:<5}" for a in DTYPE_AXES)


def _header():
    return " ".join(
        f"{a.replace('_weight', '').replace('activation', 'act'):<5}" for a in DTYPE_AXES
    )


def _headline(summary, out):
    rows = summary["results"]
    sel = summary["selection"]
    has_task = any("mean_task_acc" in r for r in rows)
    has_ppl = any("ppl" in r for r in rows)
    # only shown when actually measured: under mode="fake" they cannot separate the
    # combinations at all, so printing near-identical columns would only mislead
    has_latency = any(r.get("ttft_ms") for r in rows)

    header = f"{'':<5}{_header()}  {'mean acc' if has_task else 'accuracy':>9} {'dacc%':>7}"
    if has_ppl:
        header += f" {'PPL':>9} {'dPPL%':>7}"
    header += f" {'BPV':>6} {'dec GB/t':>9} {'proj':>6}"
    if has_latency:
        header += f" {'TTFT ms':>8} {'TPS':>7}"
    header += f" {'score':>8}  lim"
    out.append(header)
    out.append("-" * len(header))

    for i, r in enumerate(sorted(rows, key=lambda r: r["acc_cost_pct"]), 1):
        mark = "*" if r is sel["best"] else " "
        projected = r.get("decode_speedup_projected")
        line = (
            f"{mark}{i:>3} {_label(r)}  {_fmt(r.get('mean_task_acc'), '9.4f')} "
            f"{r['acc_cost_pct']:>+7.2f}"
        )
        if has_ppl:
            line += f" {_fmt(r.get('ppl'), '9.4f')} {_fmt(r.get('ppl_increase_pct'), '+7.2f')}"
        line += (
            f" {_fmt(r.get('bits_per_element'), '6.2f')} "
            f"{_fmt(r.get('decode_gb_per_token'), '9.4f')} "
            f"{(_fmt(projected, '5.2f') + 'x') if projected else '     -'}"
        )
        if has_latency:
            line += f" {_fmt(r.get('ttft_ms'), '8.1f')} {_fmt(r.get('decode_tps'), '7.1f')}"
        line += f" {_fmt(r.get('score'), '8.2f')}  {'ok' if r['within_limit'] else 'X'}"
        out.append(line)

    out.append("  dacc% = accuracy lost vs bf16 on the generation tasks; PPL is secondary")
    out.append("  proj  = decode speedup implied by the traffic, which is what decode is bound by")
    if has_latency:
        out.append(FAKE_MODE_WARNING)


def _per_task(summary, out):
    rows = [r for r in summary["results"] if r.get("task_acc")]
    if not rows:
        return
    names = sorted({name for r in rows for name in r["task_acc"]})
    out.append("")
    out.append("== accuracy by task ==")
    out.append(f"{'':<5}{_header()}  " + " ".join(f"{n:>13}" for n in names) + f" {'mean':>8}")
    for r in sorted(rows, key=lambda r: r["acc_cost_pct"]):
        cells = " ".join(f"{_fmt(r['task_acc'].get(n), '13.4f')}" for n in names)
        out.append(f"{'':<5}{_label(r)}  {cells} {r['mean_task_acc']:>8.4f}")


def _secondary(summary, out):
    """How far perplexity disagrees with the generation tasks, for the rows that have both.

    Worth calling out rather than leaving in adjacent columns: PPL is teacher-forced, so a
    combination can read near-lossless there while its generations move a lot.
    """
    rows = [
        r
        for r in summary["results"]
        if "ppl" in r and "mean_task_acc" in r and r.get("ppl_increase_pct") is not None
    ]
    if len(rows) < 2:
        return
    out.append("")
    out.append("== how far PPL disagrees with the tasks ==")
    out.append(f"{'':<5}{_header()}  {'dPPL%':>8} {'dacc%':>8} {'understated by':>15}")
    for r in sorted(rows, key=lambda r: r["acc_cost_pct"]):
        gap = r["acc_cost_pct"] - r["ppl_increase_pct"]
        out.append(
            f"{'':<5}{_label(r)}  {r['ppl_increase_pct']:>+8.2f} {r['acc_cost_pct']:>+8.2f} "
            f"{gap:>+15.2f}"
        )
    out.append("  positive = the tasks lost more than perplexity suggested")


def _generation(summary, out):
    rows = [r for r in summary["results"] if "lambada_acc" in r]
    if not rows:
        return
    out.append("")
    out.append("== stage 2: relative generation check (shortlist only) ==")
    out.append(f"{'':<5}{_header()}  {'LAMBADA':>8} {'agree':>7} {'exact':>7} {'1st div':>8}")
    for r in sorted(rows, key=lambda r: r["acc_cost_pct"]):
        out.append(
            f"{'':<5}{_label(r)}  {r['lambada_acc']:>8.4f} "
            f"{_fmt(r.get('generation_agreement'), '7.4f')} "
            f"{_fmt(r.get('generation_exact_match'), '7.4f')} "
            f"{_fmt(r.get('generation_first_divergence'), '8.4f')}"
        )
    out.append("  agree measures divergence from the bf16 output, not whether it got worse")


def _selection(summary, out):
    sel = summary["selection"]
    weights = summary["weights"]
    limit = sel["acc_drop_limit_pct"]
    out.append("")
    out.append(
        f"== selection (accuracy drop <= {limit}% ; weights "
        + ", ".join(f"{k}={v}" for k, v in weights.items())
        + ") =="
    )
    if sel["best"]:
        b = sel["best"]
        out.append(
            f"  best: {b['name']}  acc {b['acc_cost_pct']:+.2f}%  "
            f"BPV {_fmt(b.get('bits_per_element'), '.2f')}  "
            f"decode {_fmt(b.get('decode_gb_per_token'), '.4f')} GB/token  "
            f"score {_fmt(b.get('score'), '.2f')}"
        )
        out.append(f"  {sel['n_within_limit']} of {summary['n_runs']} runs are within the limit")
    else:
        out.append("  nothing is within the limit")
    for m in sel["near_misses"]:
        out.append(f"  near miss: {m['name']}  {m['acc_cost_pct']:+.2f}%")

    if sel["ranked"]:
        out.append("")
        out.append("== weighted ranking (higher is better) ==")
        out.append(f"  {'score':>8}  {'dacc%':>7} {'BPV':>6} {'dec GB/t':>9}  name")
        for r in sel["ranked"]:
            out.append(
                f"  {_fmt(r.get('score'), '8.2f')}  {r['acc_cost_pct']:>+7.2f} "
                f"{_fmt(r.get('bits_per_element'), '6.2f')} "
                f"{_fmt(r.get('decode_gb_per_token'), '9.4f')}  {r['name']}"
            )
        if any(r.get("score_missing") for r in sel["ranked"]):
            out.append("  (rows missing a metric scored on what they had)")


def _structure(summary, out):
    out.append("")
    out.append("== pareto front (nothing is both cheaper to decode and more accurate) ==")
    for r in summary["pareto_front"]:
        out.append(
            f"  {r['decode_gb_per_token']:>7.4f} GB/token  {r['acc_cost_pct']:>+7.2f}%  {r['name']}"
        )

    out.append("")
    out.append("== where the accuracy loss comes from (one axis changed, the rest fixed) ==")
    out.append(f"  {'change':<32} {'n':>3} {'dacc% mean':>11} {'[min..max]':>18} {'dBPV':>7}")
    for label, e in sorted(
        summary["axis_effects"].items(), key=lambda kv: -abs(kv[1]["acc_cost_mean"])
    ):
        span = f"[{e['acc_cost_min']:+.2f}..{e['acc_cost_max']:+.2f}]"
        out.append(
            f"  {label:<32} {e['n']:>3} {e['acc_cost_mean']:>+11.2f} {span:>18} "
            f"{e['bpv_delta_mean']:>+7.2f}"
        )

    if summary["interactions"]:
        out.append("")
        out.append(f"== do the losses add up? (vs {summary['reference']['name']}) ==")
        out.append("  residual > 0 means the changes hurt more together than apart")
        out.append(f"  {'combination':<26} {'predicted':>10} {'actual':>10} {'residual':>10}")
        for d in summary["interactions"][:6]:
            changed = "+".join(a.replace("_weight", "") for a in d["changed"])
            out.append(
                f"  {changed:<26} {d['predicted']:>10.2f} {d['actual']:>10.2f} "
                f"{d['residual']:>+10.2f}"
            )


def format_report(summary) -> str:
    """The comprehensive table, plus the breakdowns behind it."""
    out = []
    _headline(summary, out)
    _per_task(summary, out)
    _secondary(summary, out)
    _generation(summary, out)
    _selection(summary, out)
    _structure(summary, out)
    return "\n".join(out)
