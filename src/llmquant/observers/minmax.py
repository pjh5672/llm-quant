import torch

from llmquant.modifiers.quantization.scheme import QuantizationArgs

SCALE_EPS = 1e-8


def group_view(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Split the last (reduction) axis into groups: [..., K] -> [..., K // group_size, group_size]."""
    k = x.shape[-1]
    if k % group_size:
        raise ValueError(f"last dim {k} is not divisible by group_size {group_size}")
    return x.reshape(*x.shape[:-1], k // group_size, group_size)


def compute_scale(x: torch.Tensor, args: QuantizationArgs) -> torch.Tensor:
    """Symmetric abs-max scale in fp32, keepdim so it broadcasts.

    Weights ([N, K]) and activations ([..., K]) both reduce over the last axis, so
    "channel" and "token" share a branch. "group" reduces over the grouped view and
    returns [..., K // group_size, 1], which broadcasts against group_view(x).
    """
    qmax = 2 ** (args.num_bits - 1) - 1
    xf = x.float()
    if args.strategy == "group":
        amax = group_view(xf, args.group_size).abs().amax(dim=-1, keepdim=True)
    elif args.strategy in ("channel", "token"):
        amax = xf.abs().amax(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unsupported strategy {args.strategy!r}")
    # Divide by a TENSOR, not a Python scalar: `tensor / 7` lowers to a reciprocal
    # multiply in PyTorch and lands 1 ulp off true IEEE division, which a CUDA kernel
    # cannot reproduce. True division is the bit-exact rule. See docs/w4a8_rtn_notes.md.
    divisor = torch.tensor(float(qmax), dtype=torch.float32, device=amax.device)
    return (amax / divisor).clamp(min=SCALE_EPS)
