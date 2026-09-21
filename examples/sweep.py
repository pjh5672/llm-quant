"""One command: point it at a model, get a finished quantization study.

    python examples/sweep.py --cfg configs/sweep.yaml
    python examples/sweep.py --cfg configs/sweep.yaml --model Qwen/Qwen2.5-1.5B-Instruct
    python examples/sweep.py --cfg configs/sweep.yaml --skip-verify   # sweep only

Six stages, in the order the conclusions actually depend on each other:

  1 inspect    the structural facts that change how the results read -- tied embeddings,
               head_dim against group_size, where the parameters live. Printed first
               because they decide what the later tables mean.
  2 sweep      every combination in the grid, on the fake path. Accuracy is real here;
               speed is not, because fake quant stores dequantized bf16 weights.
  3 select     the accuracy limit, the weighted ranking, axis effects and interactions.
  4 verify     the selected combination re-measured on the REAL path, plus an ablation
               for every axis handled outside the weight kernel.
  5 reconcile  projection against measurement, and which axis owns the difference.
  6 report     to the terminal, and to experiments/<project>/report.md.

Stage 4 is the one a sweep alone cannot replace. The analytic decode estimate is a ratio of
bytes and cannot price work done outside the kernel: an int8 KV cache projected at 1.47x on
Llama-3.2-1B and measured 0.65x, because it saves 1.8% of decode traffic and costs 32% of
decode time. Anything that runs per step in PyTorch has the same shape of error, so the
winner is measured rather than trusted. Use --skip-verify if there is no GPU for it.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

from llmquant.core.config import ModelArgs, QuantConfig
from llmquant.core.model import load_pretrained
from llmquant.core.parser import build_config, expand_sweep
from llmquant.eval.analysis import shortlist, summarize
from llmquant.eval.inspect import fact_warnings, model_facts
from llmquant.eval.markdown import format_markdown
from llmquant.eval.report import format_report
from llmquant.eval.run import run_generation, run_one
from llmquant.eval.verify import reconcile, verdict, verification_plan

ROOT = Path(__file__).resolve().parents[1]


def planned_runs(config):
    """bf16 baseline first: it defines the limit and the generation reference."""
    yield replace(config, quantize=False)
    targets = ("attn_weight", "mlp_weight", "head_weight")
    for run in expand_sweep(config):
        if all(run.quant.scheme(t) is None for t in targets):
            continue  # identical to the baseline
        yield run


def stage_inspect(config):
    """Load the model once, just to describe it. Freed before the sweep allocates.

    The padding geometry is probed with every target quantized, whatever the config says.
    Padding depends on group_size and head_dim, not on the bit width, and the base config
    of a sweep usually leaves the dtypes at bf16 for the grid to fill in -- so asking it
    directly would report no padding at all for the very runs that pay for it.
    """
    model, _ = load_pretrained(ModelArgs(model_id=config.model, device=config.device))
    probe = QuantConfig(
        attn_weight="int8",
        mlp_weight="int8",
        head_weight="int8",
        group_size=config.quant.group_size,
        qkv_out_scale_per_head=config.quant.qkv_out_scale_per_head,
    ).to_modifier()
    facts = model_facts(model, probe, group_size=config.quant.group_size)
    del model
    torch.cuda.empty_cache()
    return facts


def stage_sweep(config):
    pairs = []
    for run in planned_runs(config):
        row = run_one(run)
        pairs.append((run, row))
        print(json.dumps(row), flush=True)
    return pairs


def stage_generation(pairs, summary):
    """The relative generation check, against the bf16 run's own output."""
    picked = shortlist(summary)
    if not picked:
        return
    baseline_run, baseline_row = next((r, row) for r, row in pairs if not row["quantize"])
    print(f"stage 2: generation for {len(picked)} of {len(pairs) - 1} runs", flush=True)
    metrics, reference = run_generation(baseline_run, reference=None)
    baseline_row.update(metrics)
    baseline_row.update(
        generation_agreement=1.0, generation_exact_match=1.0, generation_first_divergence=1.0
    )
    for run, row in pairs:
        if not any(row is p for p in picked):
            continue
        row_metrics, _ = run_generation(run, reference=reference)
        row.update(row_metrics)


def stage_verify(config, summary, mode, task_limit):
    """Re-measure the winner for real, and ablate every out-of-kernel axis."""
    winner = (summary.get("selection") or {}).get("best")
    if winner is None:
        print("verify: nothing met the accuracy limit, skipping", flush=True)
        return None

    rows = []
    for label, run in verification_plan(config, winner, mode=mode):
        # accuracy is already known from the sweep; this stage is about speed, so the task
        # suite runs short and perplexity and the generation check are dropped entirely
        run = replace(run, ppl=False, generation=False, latency=True, task_limit=task_limit)
        print(f"verify: {label} (mode={run.mode})", flush=True)
        row = run_one(run)
        row["verify_label"] = label
        rows.append(row)
        print(json.dumps(row), flush=True)

    # the projection lives on the sweep row, and is what this stage is checking
    for row in rows:
        if row["verify_label"] == "selected":
            row["decode_speedup_projected"] = winner.get("decode_speedup_projected")
    return {"rows": rows, "reconciliation": reconcile(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-verify", action="store_true",
                        help="stop after the sweep; no real-path measurement")
    parser.add_argument("--verify-mode", default="kernel", choices=("real", "kernel"),
                        help="path to verify on (default: kernel)")
    parser.add_argument("--verify-task-limit", type=int, default=100,
                        help="examples per task in the verify stage (default: 100)")
    known, rest = parser.parse_known_args()

    config = build_config(rest, root_dir=ROOT)
    torch.manual_seed(config.seed)

    print("== 1. model ==", flush=True)
    facts = stage_inspect(config)
    print(json.dumps({k: v for k, v in facts.items() if k != "padding"}, indent=2), flush=True)
    for note in fact_warnings(facts):
        print(f"  ! {note}", flush=True)
    print(flush=True)

    print("== 2. sweep ==", flush=True)
    pairs = stage_sweep(config)
    results = [row for _, row in pairs]

    summary = summarize(results, config.selection)
    if config.generation:
        stage_generation(pairs, summary)
        summary = summarize(results, config.selection)  # rescore with the stage-2 metrics

    verification = None
    if not known.skip_verify:
        print()
        print("== 4. verify on the real path ==", flush=True)
        verification = stage_verify(
            config, summary, known.verify_mode, known.verify_task_limit
        )

    print()
    print(format_report(summary))
    if verification:
        print()
        print("== projection vs measurement ==")
        for line in verdict(verification["reconciliation"]):
            print(f"  {line}")

    payload = {**summary, "model_facts": facts, "verification": verification}
    sweep_path = config.project_dir / "sweep.json"
    sweep_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # reports/ rather than experiments/, which is gitignored along with the model files in
    # it. The document is the part worth committing: it is how one model's study gets
    # compared against the next one's.
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"{config.project}.md"
    report_path.write_text(
        format_markdown(summary, facts, verification, title=f"Quantization study: {config.model}"),
        encoding="utf-8",
    )
    print()
    print(f"wrote {sweep_path}")
    print(f"wrote {report_path}")
    print(f"re-analyse without a GPU: python examples/analyze.py {sweep_path}")


if __name__ == "__main__":
    sys.exit(main())
