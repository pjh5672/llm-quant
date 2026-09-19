"""Re-analyse a finished sweep without re-running it (no GPU, no model).

    python examples/phase1_analyze.py experiments/phase1-sweep/sweep.json
    python examples/phase1_analyze.py experiments/phase1-sweep/sweep.json --bpv-weight 5

The weights are what turn the same measurements into different answers, so they are flags
rather than something baked into the saved file.
"""

import argparse
import json
from pathlib import Path

from llmquant.core.selection import SelectionConfig
from llmquant.eval.analysis import summarize
from llmquant.eval.report import format_report

DERIVED = (
    "acc_cost_pct",
    "within_limit",
    "ppl_increase_pct",
    "score",
    "score_missing",
    "decode_speedup_projected",
    "decode_speedup_measured",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="sweep.json produced by examples/phase1_sweep.py")
    p.add_argument("--acc-limit", type=float, default=5.0, help="max accuracy drop, percent")
    p.add_argument("--no-limit", action="store_true", help="rank everything, ignore the limit")
    p.add_argument("--accuracy-weight", type=float, default=1.0)
    p.add_argument("--bpv-weight", type=float, default=2.0)
    p.add_argument("--speed-weight", type=float, default=2.0, help="decode traffic")
    p.add_argument("--prefill-weight", type=float, default=1.0, help="TTFT")
    args = p.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    rows = payload["results"] if isinstance(payload, dict) else payload
    for r in rows:  # recompute everything derived against the requested weights
        for key in DERIVED:
            r.pop(key, None)

    selection = SelectionConfig(
        acc_drop_limit_pct=None if args.no_limit else args.acc_limit,
        accuracy_weight=args.accuracy_weight,
        bpv_weight=args.bpv_weight,
        decode_speed_weight=args.speed_weight,
        prefill_speed_weight=args.prefill_weight,
    )
    print(format_report(summarize(rows, selection)))


if __name__ == "__main__":
    main()
