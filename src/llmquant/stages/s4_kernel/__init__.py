"""Phase 4 -- custom CUDA kernels.

Today: the fused quant-dequant kernel, bit-exact with the PyTorch reference.
Next: the GEMM that reads packed int4/int8 weights directly.
"""

from llmquant.stages.s4_kernel.build import load_extension
from llmquant.stages.s4_kernel.ops import fake_quantize_cuda

__all__ = ["fake_quantize_cuda", "load_extension"]
