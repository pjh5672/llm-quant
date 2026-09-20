"""The study written to a file, so results outlive the terminal.

Separate from llmquant.eval.report because the two have different jobs. The terminal report
is fixed-width and meant to be skimmed while a sweep runs; this is a document meant to be
committed, diffed against the previous model, and read by someone who was not watching.
So it carries the things the terminal tables leave implicit: what model this was, what the
structural traps are, and whether the projection survived measurement.
"""

from llmquant.eval.inspect import fact_warnings
from llmquant.eval.verify import verdict

DTYPE_AXES = ("attn_weight", "mlp_weight", "head_weight", "activation", "kv_cache")
GB = 1024**3


def _cell(value, spec="", blank="-"):
    if value is None:
        return blank
    return f"{value:{spec}}" if spec else str(value)


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def _facts(facts, out):
    if not facts:
        return
    out.append("## Model")
    out.append("")
    shape = [
        ("model", facts.get("model_id")),
        ("layers", facts.get("num_layers")),
        ("hidden", facts.get("hidden_size")),
        ("heads", facts.get("num_attention_heads")),
        ("kv heads", facts.get("num_key_value_heads")),
        ("head_dim", facts.get("head_dim")),
        ("vocab", facts.get("vocab_size")),
        ("tied embeddings", facts.get("tie_word_embeddings")),
    ]
    out += _table(["field", "value"], [[k, _cell(v)] for k, v in shape if v is not None])
    out.append("")

    notes = fact_warnings(facts)
    if notes:
        out.append("### What this means for the numbers below")
        out.append("")
        for note in notes:
            out.append(f"- {note}")
        out.append("")


def _results(summary, out):
    rows = summary["results"]
    out.append("## Every combination")
    out.append("")
    headers = ["attn", "mlp", "head", "act", "kv", "acc", "dacc%", "PPL", "BPV",
               "kv KB/t", "dec GB/t", "TTFT ms", "TPS"]
    body = []
    for r in sorted(rows, key=lambda r: r["acc_cost_pct"]):
        body.append([
            *(str(r.get(a, "bf16")) for a in DTYPE_AXES),
            _cell(r.get("mean_task_acc"), ".4f"),
            f"{r['acc_cost_pct']:+.2f}",
            _cell(r.get("ppl"), ".3f"),
            _cell(r.get("bits_per_element"), ".2f"),
            _cell(
                r["kv_gb_per_1k_context"] * GB / 1024 / 1024
                if r.get("kv_gb_per_1k_context") else None, ".1f"),
            _cell(r.get("decode_gb_at_context"), ".4f"),
            _cell(r.get("ttft_ms"), ".1f"),
            _cell(r.get("decode_tps"), ".1f"),
        ])
    out += _table(headers, body)
    out.append("")
    out.append("`dacc%` is accuracy lost against bf16 on the generation tasks. TTFT and TPS "
               "come from the sweep path; if that path is fake quant they describe the "
               "simulation, not a deployment -- see Verification.")
    out.append("")


def _axes(summary, out):
    effects = summary.get("axis_effects") or {}
    if effects:
        out.append("## Where the accuracy goes")
        out.append("")
        out.append("One axis changed, the rest held fixed.")
        out.append("")
        out += _table(
            ["change", "n", "mean dacc%", "min", "max", "dBPV"],
            [[f"`{change}`", str(e["n"]), f"{e['acc_cost_mean']:+.2f}",
              f"{e['acc_cost_min']:+.2f}", f"{e['acc_cost_max']:+.2f}",
              _cell(e.get("bpv_delta_mean"), "+.2f")]
             for change, e in sorted(effects.items(), key=lambda kv: -kv[1]["acc_cost_mean"])],
        )
        out.append("")
    combos = summary.get("interactions") or []
    if combos:
        out.append("### Do the losses add up?")
        out.append("")
        out.append("A positive residual means the changes hurt more together than apart, "
                   "which is the case worth knowing about: it means the axes cannot be "
                   "chosen independently.")
        out.append("")
        out += _table(
            ["combination", "predicted", "actual", "residual"],
            [["+".join(c["changed"]), f"{c['predicted']:.2f}", f"{c['actual']:.2f}",
              f"{c['residual']:+.2f}"] for c in combos],
        )
        out.append("")


def _choice(summary, out):
    sel = summary.get("selection") or {}
    out.append("## Selected")
    out.append("")
    best = sel.get("best")
    if not best:
        out.append("Nothing met the accuracy limit.")
        out.append("")
        return
    out.append(f"**`{best['name']}`** -- gives up {best['acc_cost_pct']:.2f}% of the bf16 "
               f"accuracy for BPV {_cell(best.get('bits_per_element'), '.2f')} and "
               f"{_cell(best.get('decode_gb_at_context'), '.4f')} GB/token of decode traffic.")
    out.append("")
    out.append(f"{sel.get('n_within_limit', 0)} of {summary['n_runs']} runs met the limit.")
    out.append("")
    ranked = sel.get("ranked") or []
    if ranked:
        out.append("### Weighted ranking")
        out.append("")
        out += _table(
            ["score", "dacc%", "BPV", "dec GB/t", "name"],
            [[_cell(r.get("score"), ".2f"), f"{r['acc_cost_pct']:+.2f}",
              _cell(r.get("bits_per_element"), ".2f"),
              _cell(r.get("decode_gb_at_context"), ".4f"), f"`{r['name']}`"] for r in ranked],
        )
        out.append("")


def _verification(verification, out):
    if not verification:
        return
    out.append("## Verification (measured on the real path)")
    out.append("")
    rows = verification.get("rows") or []
    if rows:
        out += _table(
            ["run", "mode", "TPS", "TTFT ms", "peak VRAM", "acc"],
            [[r.get("verify_label", r["name"]), _cell(r.get("latency_mode")),
              _cell(r.get("decode_tps"), ".1f"), _cell(r.get("ttft_ms"), ".1f"),
              _cell(r.get("peak_vram_gb"), ".2f"), _cell(r.get("mean_task_acc"), ".4f")]
             for r in rows],
        )
        out.append("")
    lines = verdict(verification.get("reconciliation") or {})
    if lines:
        out.append("### Verdict")
        out.append("")
        for line in lines:
            out.append(f"- {line}")
        out.append("")


def format_markdown(summary, facts=None, verification=None, title=None) -> str:
    """The whole study as one committable document."""
    out = [f"# {title or 'Quantization study'}", ""]
    if summary.get("context_tokens"):
        out.append(f"Decode traffic is quoted at {summary['context_tokens']} tokens of "
                   "context: weights plus the whole KV cache, re-read every step.")
        out.append("")
    _facts(facts, out)
    _choice(summary, out)
    _verification(verification, out)
    _results(summary, out)
    _axes(summary, out)
    return "\n".join(out) + "\n"
