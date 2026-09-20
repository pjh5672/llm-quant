"""Stage 3: measure what the sweep only projected, and say so when they disagree.

A fake-quant sweep cannot measure speed. It stores dequantized bf16 weights, so every
combination moves the same bytes, and the ranking has to fall back on an analytic estimate
of decode traffic. That estimate is a ratio of bytes and it is right about bytes -- but it
knows nothing about work done outside the kernel, and one axis does a great deal of it.

That is not hypothetical. On Llama-3.2-1B the sweep projected an int8 KV cache at 1.47x
decode and it measured 0.65x: the cache saves 1.8% of decode traffic (the weights are
1.58 GB per token against a 0.034 GB cache) and costs 32% of decode time, because it is
quantized in PyTorch on every step while the weights are read by a kernel.

So the winner gets re-measured on the real path, and every axis that is handled outside the
kernel gets an ablation run -- the same combination with that axis back at bf16. The gap
between the two is that axis's true cost, which is the number the projection cannot see.
"""

from dataclasses import replace

from llmquant.core.config import BF16, normalize_dtype

# Axes whose work does NOT go through the weight kernel, so the analytic decode traffic
# cannot price them. Each one gets an ablation when it is not already bf16.
OUT_OF_KERNEL_AXES = ("kv_cache", "activation")

# Below this fraction of the projection, the projection is reported as not holding up.
PROJECTION_TOLERANCE = 0.85


def verification_plan(config, winner_row, mode="kernel"):
    """(label, config) for every run stage 3 needs.

    The baseline is re-measured rather than taken from the sweep so that the comparison is
    free of drift between processes -- clock, memory state and driver all move between runs.
    """
    quant = replace(
        config.quant,
        attn_weight=winner_row["attn_weight"],
        mlp_weight=winner_row["mlp_weight"],
        head_weight=winner_row["head_weight"],
        activation=winner_row["activation"],
        kv_cache=winner_row["kv_cache"],
    )
    base = replace(config, quant=quant, mode=mode)

    plan = [("bf16 baseline", replace(base, quantize=False))]
    plan.append(("selected", base))
    for axis in OUT_OF_KERNEL_AXES:
        if normalize_dtype(getattr(quant, axis)) == BF16:
            continue
        plan.append(
            (f"selected, {axis}=bf16", replace(base, quant=replace(quant, **{axis: BF16})))
        )
    return plan


def reconcile(rows) -> dict:
    """Projection against measurement, and what the difference is attributable to.

    `rows` are the verification rows, labelled, with the baseline first.
    """
    by_label = {r["verify_label"]: r for r in rows}
    baseline = by_label.get("bf16 baseline")
    selected = by_label.get("selected")
    if baseline is None or selected is None:
        return {}
    if not baseline.get("decode_tps") or not selected.get("decode_tps"):
        return {}

    measured = selected["decode_tps"] / baseline["decode_tps"]
    projected = selected.get("decode_speedup_projected")
    result = {
        "measured_decode_speedup": measured,
        "projected_decode_speedup": projected,
        "baseline_tps": baseline["decode_tps"],
        "selected_tps": selected["decode_tps"],
        "held_up": None if not projected else measured >= projected * PROJECTION_TOLERANCE,
        "attribution": [],
    }
    if baseline.get("ttft_ms") and selected.get("ttft_ms"):
        result["ttft_ratio"] = selected["ttft_ms"] / baseline["ttft_ms"]
    if baseline.get("peak_vram_gb") and selected.get("peak_vram_gb"):
        result["vram_ratio"] = selected["peak_vram_gb"] / baseline["peak_vram_gb"]

    # each ablation says what putting one axis back at bf16 recovers
    for label, row in by_label.items():
        if not label.startswith("selected, ") or not row.get("decode_tps"):
            continue
        axis = label.split(", ", 1)[1].split("=")[0]
        recovered = row["decode_tps"] / selected["decode_tps"]
        result["attribution"].append(
            {
                "axis": axis,
                "dtype": normalize_dtype(selected.get(axis)),
                "tps_without_it": row["decode_tps"],
                "recovered": recovered,
                "costs_decode_pct": (1 - selected["decode_tps"] / row["decode_tps"]) * 100,
            }
        )
    result["attribution"].sort(key=lambda a: -a["recovered"])
    return result


def verdict(reconciliation: dict) -> list[str]:
    """Plain statements a reader can act on, including the ones we got wrong by hand."""
    if not reconciliation:
        return []
    lines = []
    measured = reconciliation["measured_decode_speedup"]
    projected = reconciliation.get("projected_decode_speedup")
    if projected:
        verb = "holds up" if reconciliation["held_up"] else "does NOT hold up"
        lines.append(
            f"decode: projected {projected:.2f}x, measured {measured:.2f}x -- the projection "
            f"{verb} ({reconciliation['selected_tps']:.1f} vs "
            f"{reconciliation['baseline_tps']:.1f} tok/s)."
        )
    if measured < 1.0:
        lines.append(
            "the selected combination decodes SLOWER than bf16. Quantized weights only pay "
            "when weight bandwidth dominates the step; on a small model it may not."
        )
    for item in reconciliation["attribution"]:
        if item["costs_decode_pct"] > 5:
            lines.append(
                f"{item['axis']}={item['dtype']} costs {item['costs_decode_pct']:.0f}% of "
                f"decode throughput ({item['tps_without_it']:.1f} tok/s without it). It runs "
                "outside the weight kernel, so the analytic traffic estimate cannot see it."
            )
        elif item["costs_decode_pct"] < -5:
            lines.append(
                f"{item['axis']}={item['dtype']} is worth keeping: removing it would lose "
                f"{-item['costs_decode_pct']:.0f}% of decode throughput."
            )
    if reconciliation.get("ttft_ratio", 0) > 1.2:
        lines.append(
            f"TTFT is {reconciliation['ttft_ratio']:.2f}x the baseline: prefill dequantizes "
            "weights every pass. Fusing dequant into the GEMM mainloop is what removes it."
        )
    if reconciliation.get("vram_ratio"):
        lines.append(f"peak VRAM is {reconciliation['vram_ratio']:.2f}x the baseline.")
    return lines
