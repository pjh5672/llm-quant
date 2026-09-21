"""Compiling the CUDA kernels and calling them.

Two kernels live in csrc/: a fused quant-dequant that is bit-exact with the PyTorch
reference, and a weight-only matmul that reads int8 straight from memory. They are built on
first use by torch.utils.cpp_extension and cached after that; build.py deals with the
Windows toolchain, including the one where nvcc dies silently because %TMP% has a space.
"""

from llmquant.cuda.build import load_extension
from llmquant.cuda.ops import fake_quantize_cuda

__all__ = ["fake_quantize_cuda", "load_extension"]
