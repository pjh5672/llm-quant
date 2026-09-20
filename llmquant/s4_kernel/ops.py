import torch

from llmquant.s4_kernel.build import load_extension
from llmquant.core.observers import pad_to_group
from llmquant.core.scheme import QuantizationArgs


def fake_quantize_cuda(x: torch.Tensor, args: QuantizationArgs, return_scale: bool = False):
    """CUDA quant-dequant, bit-exact with llmquant.core.quant_ops.fake_quantize.

    Grouping is along the last (reduction) axis for both weights and activations;
    "channel"/"token" are the single-group case.
    """
    if args.strategy == "group":
        group_size = args.group_size
    elif args.strategy in ("channel", "token"):
        group_size = x.shape[-1]
    else:
        raise ValueError(f"unsupported strategy {args.strategy!r}")

    # the kernel requires a whole number of groups, so the same zero padding the PyTorch
    # reference applies happens here too -- otherwise the two would stop agreeing on any
    # tensor whose last axis is not a multiple of the group size
    original = x.shape[-1]
    padded = pad_to_group(x, group_size)
    out, scale = load_extension().fake_quant_group(padded, args.num_bits, group_size, return_scale)
    if padded.shape[-1] != original:
        out = out[..., :original]
    return (out, scale) if return_scale else out
