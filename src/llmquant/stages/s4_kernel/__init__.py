"""Phase 4 -- custom CUDA kernels.

The fused quant-dequant kernel, bit-exact with the PyTorch reference, and a weight-only
matmul that reads int8 weights straight from memory -- about 2.4x over fake quant at the
decode shape, which is the regime the whole project is aimed at.
"""

from llmquant.stages.s4_kernel.build import load_extension
from llmquant.stages.s4_kernel.ops import fake_quantize_cuda
from llmquant.stages.s4_kernel.quant_linear import KERNEL_MAX_ROWS, KernelQuantLinear

__all__ = [
    "KERNEL_MAX_ROWS",
    "KernelQuantLinear",
    "fake_quantize_cuda",
    "load_extension",
]
