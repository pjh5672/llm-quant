"""YAML config + CLI overrides, mapped onto a single RunConfig.

Unlike gaia-compressor, an explicitly passed CLI flag wins over the YAML file; the YAML
only supplies defaults. Anything neither sets falls back to the dataclass default.
"""

import argparse
import itertools
import shutil
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import yaml

from llmquant.core.config import BF16, QuantConfig
from llmquant.core.datasets.tasks import DEFAULT_TASKS, TASKS
from llmquant.core.selection import SelectionConfig

SECTIONS = ("defaults", "quantization", "evaluation")
SWEEP_SECTION = "sweep"
SELECTION_SECTION = "selection"


@dataclass
class RunConfig:
    project: str = "test"
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda"
    seq_len: int = 2048
    seed: int = 0
    save_path: str | None = None
    pack: bool = False
    load_packed: str | None = None

    quantize: bool = True
    quant_method: str = "rtn"
    mode: str = "fake"
    quant: QuantConfig = field(default_factory=QuantConfig)

    metric: str | None = "wikitext2"
    generation: bool = True  # stage 2 of a sweep: LAMBADA + greedy agreement vs bf16
    lambada_limit: int = 500
    max_new_tokens: int = 64
    tasks: tuple[str, ...] = DEFAULT_TASKS    # generation tasks -> the accuracy number
    task_limit: int | None = None             # None uses each task's default n
    ppl: bool = True                          # secondary signal, cheap
    latency: bool = False                     # TTFT + decode TPS; see the note below
    latency_prompt_tokens: int = 512
    latency_new_tokens: int = 64
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    project_dir: Path = Path(".")
    sweep: dict = field(default_factory=dict)  # QuantConfig field -> list of values

    def __post_init__(self):
        if self.quant_method != "rtn":
            raise NotImplementedError(
                f"quant_method={self.quant_method!r}: only calibration-free 'rtn' exists. "
                "The slot is kept so the schema is stable."
            )
        if self.mode != "fake":
            raise NotImplementedError(
                f"mode={self.mode!r}: only 'fake' exists so far "
                "('real' is Phase 2, 'kernel' is Phase 4; see docs/w4a8_rtn_notes.md)."
            )
        for flag, phase in (("pack", "Phase 3"), ("load_packed", "Phase 4")):
            if getattr(self, flag):
                raise NotImplementedError(f"{flag} is not implemented yet ({phase}).")
        object.__setattr__(self, "tasks", tuple(self.tasks or ()))
        unknown = [t for t in self.tasks if t not in TASKS]
        if unknown:
            raise ValueError(f"unknown task(s) {unknown}, expected from {sorted(TASKS)}")
        if not self.tasks and not self.ppl:
            raise ValueError("nothing to measure: enable at least one task, or ppl")

    def to_modifier(self):
        return self.quant.to_modifier(mode=self.mode) if self.quantize else None


_QUANT_KEYS = tuple(f.name for f in fields(QuantConfig))
_SELECTION_KEYS = tuple(f.name for f in fields(SelectionConfig))
_RUN_KEYS = tuple(
    f.name
    for f in fields(RunConfig)
    if f.name not in ("quant", "project_dir", "sweep", "selection")
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="llm-quant: RTN quantization run")
    p.add_argument("--cfg", type=str, default=None, help="Path to a YAML config")
    p.add_argument("--project", type=str, help="Experiment name")
    p.add_argument("--model", type=str, help="HF model id or path")
    p.add_argument("--device", type=str, help="cuda | cpu | cuda:N")
    p.add_argument("--seq-len", type=int, dest="seq_len")
    p.add_argument("--seed", type=int)
    p.add_argument("--save-path", type=str, dest="save_path")
    p.add_argument("--pack", action="store_true", default=None, help="Pack weights to .bin")
    p.add_argument("--load-packed", type=str, dest="load_packed", help="Load a packed model and skip quantization")
    p.add_argument("--no-quantize", action="store_false", dest="quantize", default=None)
    p.add_argument("--quant-method", type=str, dest="quant_method")
    p.add_argument("--mode", type=str, choices=("fake", "real", "kernel"))
    p.add_argument("--metric", type=str)
    p.add_argument("--no-generation", action="store_false", dest="generation", default=None)
    p.add_argument("--lambada-limit", type=int, dest="lambada_limit")
    p.add_argument("--max-new-tokens", type=int, dest="max_new_tokens")
    p.add_argument("--tasks", type=str, nargs="+",
                   help="arc_easy arc_challenge openbookqa gsm8k")
    p.add_argument("--task-limit", type=int, dest="task_limit")
    p.add_argument("--no-ppl", action="store_false", dest="ppl", default=None)
    p.add_argument("--no-latency", action="store_false", dest="latency", default=None)
    for key in _QUANT_KEYS:
        if key == "group_size":
            p.add_argument("--group-size", type=int, dest="group_size")
        else:
            p.add_argument(f"--{key.replace('_', '-')}", type=str, dest=key,
                           help="int4 | int8 | bf16")
    return p


def _yaml_flat(path: Path) -> tuple[dict, dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - set(SECTIONS) - {SWEEP_SECTION, SELECTION_SECTION}
    if unknown:
        raise ValueError(
            f"unknown config section(s) {sorted(unknown)}, "
            f"expected {[*SECTIONS, SWEEP_SECTION, SELECTION_SECTION]}"
        )
    flat = {}
    for section in SECTIONS:
        flat.update(raw.get(section) or {})
    known = set(_RUN_KEYS) | set(_QUANT_KEYS)
    unknown_keys = set(flat) - known
    if unknown_keys:
        raise ValueError(f"unknown config key(s) {sorted(unknown_keys)}, expected {sorted(known)}")

    sweep = raw.get(SWEEP_SECTION) or {}
    bad_keys = set(sweep) - set(_QUANT_KEYS)
    if bad_keys:
        raise ValueError(
            f"unknown sweep key(s) {sorted(bad_keys)}, expected {sorted(_QUANT_KEYS)}"
        )
    for key, values in sweep.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"sweep.{key} must be a non-empty list, got {values!r}")

    selection = raw.get(SELECTION_SECTION) or {}
    bad = set(selection) - set(_SELECTION_KEYS)
    if bad:
        raise ValueError(
            f"unknown selection key(s) {sorted(bad)}, expected {sorted(_SELECTION_KEYS)}"
        )
    return flat, sweep, selection


def build_config(argv=None, root_dir: Path | None = None) -> RunConfig:
    cli = vars(_build_argparser().parse_args(argv))
    cfg_path = cli.pop("cfg", None)
    from_yaml, sweep, selection = _yaml_flat(Path(cfg_path)) if cfg_path else ({}, {}, {})

    def pick(key, default):
        if cli.get(key) is not None:  # an explicit CLI flag always wins
            return cli[key]
        return from_yaml.get(key, default)

    quant_defaults = QuantConfig()
    quant = QuantConfig(**{k: pick(k, getattr(quant_defaults, k)) for k in _QUANT_KEYS})

    run_defaults = RunConfig()
    values = {k: pick(k, getattr(run_defaults, k)) for k in _RUN_KEYS}

    root = Path(root_dir) if root_dir else Path.cwd()
    project_dir = root / "experiments" / values["project"]
    project_dir.mkdir(parents=True, exist_ok=True)
    if cfg_path:
        shutil.copyfile(cfg_path, project_dir / Path(cfg_path).name)

    return RunConfig(
        **values,
        quant=quant,
        selection=SelectionConfig(**selection),
        project_dir=project_dir,
        sweep=sweep,
    )


def expand_sweep(config: RunConfig) -> list[RunConfig]:
    """Cross product of the `sweep:` grid, as full RunConfigs sharing everything else.

    Combinations that resolve to the same schemes *and* the same KV cache are collapsed:
    once every weight target is bf16 the activation dtype has nothing to act on, so those
    runs would be identical.
    """
    if not config.sweep:
        return [config]

    keys = [k for k in _QUANT_KEYS if k in config.sweep]
    runs, seen = [], set()
    for combo in itertools.product(*(config.sweep[k] for k in keys)):
        quant = replace(config.quant, **dict(zip(keys, combo)))
        # the KV cache is not part of any weight scheme, so it has to join the key
        # explicitly or every cache dtype would collapse into one run
        key = (
            *(quant.scheme(t) for t in ("attn_weight", "mlp_weight", "head_weight")),
            quant.kv_cache_bits,
        )
        if key in seen:
            continue
        seen.add(key)
        runs.append(replace(config, quant=quant))
    return runs


__all__ = ["BF16", "QuantConfig", "RunConfig", "SelectionConfig", "build_config", "expand_sweep"]
