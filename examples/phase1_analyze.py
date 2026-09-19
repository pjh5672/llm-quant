"""Re-analyse a finished sweep without re-running it (no GPU, no model).

    python examples/phase1_analyze.py experiments/phase1-sweep/sweep.json
"""

import argparse
import json
from pathlib import Path

from llmquant.eval.analysis import format_report, summarize
from llmquant.core.selection import SelectionConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="sweep.json produced by examples/phase1_sweep.py")
    p.add_argument("--limit-ratio", type=float, default=1.05, help="PPL limit vs the bf16 baseline")
    p.add_argument("--bpv-weight", type=float, default=2.0)
    p.add_argument("--speed-weight", type=float, default=2.0)
    p.add_argument("--accuracy-weight", type=float, default=1.0)
    args = p.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    rows = payload["results"] if isinstance(payload, dict) else payload
    for r in rows:  # recompute against the requested limit
        r.pop("ppl_increase_pct", None)
        r.pop("within_limit", None)
    selection = SelectionConfig(
        ppl_limit_ratio=args.limit_ratio,
        accuracy_weight=args.accuracy_weight,
        bpv_weight=args.bpv_weight,
        decode_speed_weight=args.speed_weight,
    )
    print(format_report(summarize(rows, args.limit_ratio, selection)))


if __name__ == "__main__":
    main()
