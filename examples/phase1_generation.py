"""Top up a finished sweep with stage-2 generation metrics, without redoing stage 1.

    python examples/phase1_generation.py experiments/phase1-sweep/sweep.json
    python examples/phase1_generation.py experiments/phase1-sweep/sweep.json --generation-task gsm8k

Stage 1 (PPL over the whole grid) is the expensive part and does not change when a new
generation metric is added, so this rebuilds the shortlist from the saved rows and runs
only the generation evaluations against them.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from llmquant.core.parser import build_config
from llmquant.core.selection import SelectionConfig
from llmquant.eval.analysis import format_report, shortlist, summarize
from llmquant.eval.run import run_generation

ROOT = Path(__file__).resolve().parents[1]


def config_for(base, row):
    """Rebuild the RunConfig a saved row came from."""
    quant = replace(
        base.quant,
        attn_weight=row["attn_weight"],
        mlp_weight=row["mlp_weight"],
        head_weight=row["head_weight"],
        activation=row["activation"],
        group_size=row["group_size"],
    )
    return replace(base, quant=quant, quantize=row.get("quantize", True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="sweep.json from examples/phase1_sweep.py")
    p.add_argument("--cfg", type=str, default="configs/phase1/sweep.yaml")
    p.add_argument("--generation-task", type=str, default=None, help="arc_easy | gsm8k")
    p.add_argument("--generation-task-limit", type=int, default=None)
    args = p.parse_args()

    argv = ["--cfg", args.cfg]
    if args.generation_task:
        argv += ["--generation-task", args.generation_task]
    if args.generation_task_limit:
        argv += ["--generation-task-limit", str(args.generation_task_limit)]
    base = build_config(argv, root_dir=ROOT)
    torch.manual_seed(base.seed)

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    rows = payload["results"] if isinstance(payload, dict) else payload
    selection = base.selection
    summary = summarize(rows, selection.ppl_limit_ratio, selection)

    baseline = next(r for r in rows if not r.get("quantize", True))
    print("stage 2 reference: bf16", flush=True)
    metrics, reference = run_generation(config_for(base, baseline), reference=None)
    baseline.update(metrics)
    baseline.update(
        generation_agreement=1.0, generation_exact_match=1.0, generation_first_divergence=1.0
    )
    print(json.dumps({"name": baseline["name"], **metrics}), flush=True)

    picked = shortlist(summary)
    print(f"stage 2: {len(picked)} shortlisted runs", flush=True)
    for row in picked:
        row_metrics, _ = run_generation(config_for(base, row), reference=reference)
        row.update(row_metrics)
        print(json.dumps({"name": row["name"], **row_metrics}), flush=True)

    summary = summarize(rows, selection.ppl_limit_ratio, selection)
    args.path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(format_report(summary))
    print()
    print(f"updated {args.path}")


if __name__ == "__main__":
    main()
