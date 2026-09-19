import torch

from llmquant.core.scheme import QuantizationArgs

SCALE_EPS = 1e-8


def pad_to_group(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Zero-pad the last axis up to a multiple of group_size.

    Zeros are safe here and nowhere near arbitrary: the scale is a symmetric abs-max, and
    max(|x|) does not see a zero. So the real elements quantize exactly as they would in a
    group of their own (shorter) length -- padding changes the bookkeeping, not the numbers.
    A short tail therefore costs accuracy only in the sense that it shares a scale with
    nothing, which is the same thing a shorter group would do.
    """
    remainder = x.shape[-1] % group_size
    if not remainder:
        return x
    return torch.nn.functional.pad(x, (0, group_size - remainder))


def group_view(x: torch.Tensor, group_size: int, head_dim: int | None = None) -> torch.Tensor:
    """Split the last (reduction) axis into groups, padding a short tail.

    [..., K] -> [..., ceil(K / group_size), group_size]

    With head_dim set, the axis is first split into heads and each head is grouped on its
    own. o_proj is the case that needs it: its input is the concatenated attention output,
    so K is heads x head_dim, and a flat group of 128 would straddle two 64-wide heads and
    force them to share a scale. Each head is padded up to a whole group instead.
    """
    if head_dim is None:
        padded = pad_to_group(x, group_size)
        return padded.reshape(*padded.shape[:-1], padded.shape[-1] // group_size, group_size)

    lead, k = x.shape[:-1], x.shape[-1]
    if k % head_dim:
        raise ValueError(f"last dim {k} is not a multiple of head_dim {head_dim}")
    per_head = pad_to_group(x.reshape(*lead, k // head_dim, head_dim), group_size)
    return per_head.reshape(*lead, -1, group_size)


def ungroup(y: torch.Tensor, shape, group_size: int, head_dim: int | None = None) -> torch.Tensor:
    """Undo group_view: flatten the groups back and drop whatever padding was added."""
    lead, k = tuple(shape[:-1]), shape[-1]
    if head_dim is None:
        return y.reshape(*lead, -1)[..., :k]
    padded_head = -(-head_dim // group_size) * group_size
    per_head = y.reshape(*lead, k // head_dim, padded_head)
    return per_head[..., :head_dim].reshape(*lead, k)


def out_block_view(x: torch.Tensor, out_group: int) -> torch.Tensor:
    """Block the *output* axis so a tile of rows can share one scale.

    [out, ...] -> [ceil(out / out_group), out_group, ...], zero-padding a short tail. The
    padding is free for the same reason it is on the reduction axis: a zero cannot move a
    symmetric abs-max.
    """
    out = x.shape[0]
    remainder = out % out_group
    if remainder:
        pad = [0, 0] * (x.dim() - 1) + [0, out_group - remainder]
        x = torch.nn.functional.pad(x, pad)
    return x.reshape(-1, out_group, *x.shape[1:])


def out_unblock(y: torch.Tensor, out: int) -> torch.Tensor:
    """Undo out_block_view and drop the padded rows."""
    return y.reshape(-1, *y.shape[2:])[:out]


def compute_scale(x: torch.Tensor, args: QuantizationArgs) -> torch.Tensor:
    """Symmetric abs-max scale in fp32, keepdim so it broadcasts.

    Weights ([N, K]) and activations ([..., K]) both reduce over the last axis, so
    "channel" and "token" share a branch. "group" reduces over the grouped view and
    returns [..., K // group_size, 1], which broadcasts against group_view(x).
    """
    qmax = 2 ** (args.num_bits - 1) - 1
    xf = x.float()
    if args.strategy == "group":
        view = group_view(xf, args.group_size, args.head_dim)
        if args.out_group:
            # reduce over the output tile as well, so its rows share one scale
            amax = out_block_view(view, args.out_group).abs().amax(dim=(1, -1), keepdim=True)
        else:
            amax = view.abs().amax(dim=-1, keepdim=True)
    elif args.strategy in ("channel", "token"):
        amax = xf.abs().amax(dim=-1, keepdim=True)
    else:
        raise ValueError(f"unsupported strategy {args.strategy!r}")
    # Divide by a TENSOR, not a Python scalar: `tensor / 7` lowers to a reciprocal
    # multiply in PyTorch and lands 1 ulp off true IEEE division, which a CUDA kernel
    # cannot reproduce. True division is the bit-exact rule. See docs/w4a8_rtn_notes.md.
    divisor = torch.tensor(float(qmax), dtype=torch.float32, device=amax.device)
    return (amax / divisor).clamp(min=SCALE_EPS)
