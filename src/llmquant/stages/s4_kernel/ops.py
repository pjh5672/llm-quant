import torch

from llmquant.stages.s4_kernel.build import load_extension
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

    out, scale = load_extension().fake_quant_group(x, args.num_bits, group_size, return_scale)
    return (out, scale) if return_scale else out
