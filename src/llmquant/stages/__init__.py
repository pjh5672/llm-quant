"""One folder per phase of docs/w4a8_rtn_notes.md.

The folders line up with the `mode` config field, so "which code runs for mode=real" is
answered by looking at s2_real rather than by grepping:

  s1_fake    mode="fake"    Phase 1  bf16 dequant, for measuring the accuracy cost
  s2_real    mode="real"    Phase 2  int weights + verified PyTorch ops, the reference
  s3_pack                   Phase 3  packing, the .bin format, the loader
  s4_kernel  mode="kernel"  Phase 4  custom CUDA kernels reading packed data
  s5_chat                   Phase 5  chat

Only the stages that exist yet carry code; the rest state what will live there.
"""


def quant_linear_for(mode: str):
    """The nn.Module a given mode swaps each Linear for.

    Imported lazily so that `core` never depends on a stage at module import time: the
    dependency runs core -> stages, and only when a modifier is actually applied.
    """
    if mode == "fake":
        from llmquant.stages.s1_fake import FakeQuantLinear

        return FakeQuantLinear
    raise ValueError(f"unsupported mode {mode!r}, expected one of {sorted(AVAILABLE_MODES)}")


AVAILABLE_MODES = frozenset({"fake"})
PLANNED_MODES = {"real": "Phase 2 (s2_real)", "kernel": "Phase 4 (s4_kernel)"}
