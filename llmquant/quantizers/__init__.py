"""The three implementations of a quantized layer, and the lookup that picks one.

They differ in what they do with the integers, not in what the integers are:

  fake    dequantize them and run bf16, to measure what the dtype costs in accuracy
  real    keep them and multiply in fp32, to be the thing the kernel is checked against
  kernel  hand them to CUDA, which is the path that gets deployed

`mode` in a config selects between them, through dispatch.quant_linear_for.
"""

from llmquant.quantizers.dispatch import (
    AVAILABLE_MODES,
    quant_experts_for,
    quant_linear_for,
)
from llmquant.quantizers.fake import FakeQuantCache, FakeQuantLinear, make_cache_factory
from llmquant.quantizers.kernel import (
    KERNEL_MAX_ROWS,
    KernelQuantExperts,
    KernelQuantLinear,
    pad_activation,
    quantize_expert_stack,
)
from llmquant.quantizers.real import (
    RealQuantExperts,
    RealQuantLinear,
    exact_fp32_matmul,
    quantize_weight,
)

__all__ = [
    "AVAILABLE_MODES",
    "KERNEL_MAX_ROWS",
    "FakeQuantCache",
    "FakeQuantLinear",
    "KernelQuantExperts",
    "KernelQuantLinear",
    "RealQuantExperts",
    "RealQuantLinear",
    "exact_fp32_matmul",
    "make_cache_factory",
    "pad_activation",
    "quant_experts_for",
    "quant_linear_for",
    "quantize_expert_stack",
    "quantize_weight",
]
