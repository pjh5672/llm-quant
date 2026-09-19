"""Phase 1: expand the `sweep:` grid of a config and measure every combination.

    python examples/phase1_sweep.py --cfg configs/phase1/sweep.yaml

Two stages, because the evaluations cost very different amounts:

  stage 1  generation task accuracy (the number the criterion ranks on), plus perplexity,
           latency and the cost metrics -- for every combination in the grid.
  stage 2  LAMBADA and greedy agreement with the bf16 model, on a shortlist only
           (pareto front plus anything inside the accuracy limit).

Every run goes through llmquant.eval.run, the same code path examples/auto_llm.py uses, so
a sweep row and a standalone run of that config cannot drift apart. Results land in
experiments/<project>/.
"""

import json
from dataclasses import replace
from pathlib import Path

import torch

from llmquant.core.parser import build_config, expand_sweep
from llmquant.eval.analysis import shortlist, summarize
from llmquant.eval.report import format_report
from llmquant.eval.run import run_generation, run_one

ROOT = Path(__file__).resolve().parents[1]


def planned_runs(config):
    """bf16 baseline first: it defines the limit and the generation reference."""
    yield replace(config, quantize=False)
    targets = ("attn_weight", "mlp_weight", "head_weight")
    for run in expand_sweep(config):
        if all(run.quant.scheme(t) is None for t in targets):
            continue  # identical to the baseline
        yield run


def run_stage_two(pairs, summary):
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
    print(json.dumps({"name": baseline_row["name"], **metrics}), flush=True)

    for run, row in pairs:
        if not any(row is p for p in picked):
            continue
        row_metrics, _ = run_generation(run, reference=reference)
        row.update(row_metrics)
        print(json.dumps({"name": row["name"], **row_metrics}), flush=True)


def main():
    config = build_config(root_dir=ROOT)
    torch.manual_seed(config.seed)

    pairs = []
    for run in planned_runs(config):
        row = run_one(run)
        pairs.append((run, row))
        print(json.dumps(row), flush=True)
    results = [row for _, row in pairs]

    summary = summarize(results, config.selection)
    if config.generation:
        run_stage_two(pairs, summary)
        summary = summarize(results, config.selection)  # rescore with the stage-2 metrics

    out = config.project_dir / "sweep.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(format_report(summary))
    print()
    print(f"wrote {out}")
    print(f"re-analyse later with: python examples/phase1_analyze.py {out}")


if __name__ == "__main__":
    main()
