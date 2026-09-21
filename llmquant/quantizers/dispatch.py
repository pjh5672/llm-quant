"""Which implementation a `mode` selects.

  fake    mode="fake"    dequantize and run bf16 -- what a dtype costs in accuracy
  real    mode="real"    keep the integers, multiply in fp32 -- the reference
  kernel  mode="kernel"  hand them to CUDA -- the path that gets deployed

One lookup per kind of layer, because a Mixture-of-Experts block is not an nn.Linear and
cannot be swapped for one.
"""


def quant_linear_for(mode: str):
    """The nn.Module a given mode swaps each Linear for.

    Imported lazily so that `core` never depends on a stage at module import time: the
    dependency runs core -> stages, and only when a modifier is actually applied.
    """
    if mode == "fake":
        from llmquant.quantizers import FakeQuantLinear

        return FakeQuantLinear
    if mode == "real":
        from llmquant.quantizers import RealQuantLinear

        return RealQuantLinear
    if mode == "kernel":
        from llmquant.quantizers import KernelQuantLinear

        return KernelQuantLinear
    raise ValueError(f"unsupported mode {mode!r}, expected one of {sorted(AVAILABLE_MODES)}")


AVAILABLE_MODES = frozenset({"fake", "real", "kernel"})
PLANNED_MODES: dict[str, str] = {}


def quant_experts_for(mode: str):
    """The nn.Module a given mode swaps a stacked expert block for, or None for fake.

    Fake quant has no module: it writes dequantized weights back into the parameters and
    leaves the block alone, which is why it is the one mode a MoE has always worked on.
    """
    if mode == "fake":
        return None
    if mode == "real":
        from llmquant.quantizers import RealQuantExperts

        return RealQuantExperts
    if mode == "kernel":
        from llmquant.quantizers import KernelQuantExperts

        return KernelQuantExperts
    raise ValueError(f"unsupported mode {mode!r}, expected one of {sorted(AVAILABLE_MODES)}")
