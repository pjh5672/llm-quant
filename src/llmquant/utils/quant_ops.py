import torch

from llmquant.modifiers.quantization.scheme import QuantizationArgs
from llmquant.observers.minmax import compute_scale, group_view


def quantize(x: torch.Tensor, scale: torch.Tensor, args: QuantizationArgs) -> torch.Tensor:
    """Integer-valued fp32 tensor. torch.round is round-half-to-even; kernels must match.

    Elementwise: for "group", pass x already as group_view(x, group_size).
    """
    qmax = 2 ** (args.num_bits - 1) - 1
    return torch.clamp(torch.round(x.float() / scale), -qmax - 1, qmax)


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q * scale


def fake_quantize(x: torch.Tensor, args: QuantizationArgs) -> torch.Tensor:
    """Reference quant-dequant. The CUDA kernel in llmquant.kernels must match bit-exactly."""
    scale = compute_scale(x, args)
    xv = group_view(x, args.group_size) if args.strategy == "group" else x
    return dequantize(quantize(xv, scale, args), scale).reshape(x.shape).to(x.dtype)
