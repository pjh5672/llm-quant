"""Config-driven single run: load -> quantize -> evaluate -> report.

    python examples/auto_llm.py --cfg configs/phase1/w4a8.yaml
    python examples/auto_llm.py --cfg configs/phase1/w4a8.yaml --mlp-weight int8

Saving, packing and loading a packed model are Phase 3/4; those config slots exist and
raise a clear NotImplementedError rather than silently doing nothing.
"""

import json
from pathlib import Path

import torch

from llmquant.core.parser import build_config
from llmquant.eval.run import run_one

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = build_config(root_dir=ROOT)
    torch.manual_seed(config.seed)

    row = run_one(config)
    out = config.project_dir / "result.json"
    out.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
