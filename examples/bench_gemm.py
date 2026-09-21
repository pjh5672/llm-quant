"""How much can an int8 activation actually buy, at this model's GEMM shapes?

    python examples/bench_gemm.py
    python examples/bench_gemm.py --m 1 64 2048

Times the GEMM primitives directly, so the numbers are free of fake-quant overhead.
See llmquant/benchmark.py for how to read the three columns.
"""

import argparse
import json
from pathlib import Path

import torch

from llmquant.eval.benchmark import benchmark_gemms, format_gemm_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, nargs="+", default=[1, 64, 512, 2048], help="token counts")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", type=Path, default=None, help="optional json output")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    rows = benchmark_gemms(tuple(args.m), group_size=args.group_size, iters=args.iters)
    print(format_gemm_report(rows))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
