import torch

from llmquant.core.scheme import QuantizationArgs
from llmquant.core.observers import compute_scale, group_view, ungroup


def quantize(x: torch.Tensor, scale: torch.Tensor, args: QuantizationArgs) -> torch.Tensor:
    """Integer-valued fp32 tensor. torch.round is round-half-to-even; kernels must match.

    Elementwise: for "group", pass x already as group_view(x, group_size).
    """
    qmax = 2 ** (args.num_bits - 1) - 1
    return torch.clamp(torch.round(x.float() / scale), -qmax - 1, qmax)


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q * scale


def fake_quantize(x: torch.Tensor, args: QuantizationArgs) -> torch.Tensor:
    """Reference quant-dequant. The CUDA kernel in llmquant.stages.s4_kernel must match bit-exactly."""
    scale = compute_scale(x, args)
    if args.strategy != "group":
        return dequantize(quantize(x, scale, args), scale).to(x.dtype)
    view = group_view(x, args.group_size, args.head_dim)
    out = dequantize(quantize(view, scale, args), scale)
    return ungroup(out, x.shape, args.group_size, args.head_dim).to(x.dtype)
