"""Phase 2 -- real quant: the weight stays an integer.

Not quantize-dequantize. s1_fake stores a dequantized bf16 weight and runs an ordinary bf16
matmul, which measures the accuracy cost but nothing else. Here the weight is held as
integers and the matmul consumes it that way, which is the arithmetic a kernel has to
reproduce:

    A16  dequantize each weight group and accumulate the groups in fp32
    A8   quantize the activation per token and group, accumulate int32 *inside* each group,
         then multiply by the two scales and sum the partial products

    y[m,n] = sum_g  s_x[m,g] * s_w[n,g] * sum_{k in g} a[m,k] * q[n,k]

The A8 form is not the same computation as fake quant and is not meant to be: fake quant
rounds both sides to bf16 and accumulates there, while this accumulates exact integers and
scales once per group. It should come out slightly *better*, and that gap is the thing the
Phase 4 kernel then has to match bit for bit.

What makes this exact: with group_size 128 a partial sum peaks at 128 * 127 * 127 = 2.06M,
inside fp32's exactly-representable integer range of 2^24. So an fp32 matmul over integer
values *is* integer arithmetic, and no int32 path is needed. Accumulating over a whole K of
8192 would peak near 132M and leave that range, which is one more reason the groups are cut
at 128 rather than run to the end of the axis.

TF32 turns out not to threaten that, measured rather than assumed: its 11 bits of mantissa
hold every int8 input exactly (|v| <= 127 < 2^11) and it still accumulates in fp32, so the
result is identical with the flag either way. The guard below is therefore cheap insurance,
not load-bearing -- it only starts to matter if an input ever needs more than 11 bits.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn

from llmquant.core.observers import compute_scale, group_view
from llmquant.core.quant_ops import quantize
from llmquant.core.scheme import QuantizationScheme


@contextmanager
def exact_fp32_matmul():
    """fp32 matmul with TF32 disabled, so integer products stay exact."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def quantize_weight(weight, args):
    """Integer weight in the original [out, in] shape, plus its scales.

    The integers are kept in their natural shape rather than the padded group view: padding
    is a storage question for Phase 3, and a zero contributes nothing to a partial sum.
    """
    scale = compute_scale(weight, args)
    grouped = group_view(weight, args.group_size, args.head_dim)
    if args.out_group:
        from llmquant.core.observers import out_block_view, out_unblock

        blocked = out_block_view(grouped, args.out_group)
        q = out_unblock(quantize(blocked, scale, args), weight.shape[0])
    else:
        q = quantize(grouped, scale, args)
    from llmquant.core.observers import ungroup

    q = ungroup(q, weight.shape, args.group_size, args.head_dim)
    return q.to(torch.int8), scale


class RealQuantLinear(nn.Module):
    """Linear whose weight is stored as integers and multiplied as integers."""

    def __init__(self, qweight, wscale, bias, scheme: QuantizationScheme):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("wscale", wscale)
        self.bias = bias
        self.scheme = scheme
        self.out_features, self.in_features = qweight.shape

    @classmethod
    def from_linear(cls, linear: nn.Linear, scheme: QuantizationScheme):
        if scheme.weights is None:
            raise ValueError("real quant needs a weight scheme; bf16 weights stay nn.Linear")
        qweight, wscale = quantize_weight(linear.weight.detach(), scheme.weights)
        return cls(qweight, wscale, linear.bias, scheme)

    def _group_weight_scale(self, index):
        """Weight scale for one reduction group, broadcast to [out_features]."""
        args = self.scheme.weights
        if args.out_group:
            per_head = self.wscale[:, 0, index, 0]  # [heads]
            return per_head.repeat_interleave(args.out_group)[: self.out_features]
        return self.wscale[:, index, 0]

    def forward(self, x):
        args = self.scheme.weights
        act = self.scheme.input_activations
        shape = x.shape
        flat = x.reshape(-1, shape[-1])

        weight_groups = group_view(
            self.qweight.float(), args.group_size, args.head_dim
        )  # [out, G, g]
        if act is None:
            x_groups = group_view(flat.float(), args.group_size, args.head_dim)
            act_scale = None
        else:
            act_scale = compute_scale(flat, act)  # [tokens, G, 1]
            x_groups = quantize(group_view(flat, act.group_size, act.head_dim), act_scale, act)

        out = torch.zeros(flat.shape[0], self.out_features, device=x.device, dtype=torch.float32)
        with exact_fp32_matmul():
            for index in range(weight_groups.shape[1]):
                partial = x_groups[:, index] @ weight_groups[:, index].T  # [tokens, out]
                partial = partial * self._group_weight_scale(index)
                if act_scale is not None:
                    partial = partial * act_scale[:, index]
                out += partial

        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype).reshape(*shape[:-1], self.out_features)

    def extra_repr(self):
        bits = self.scheme.weights.num_bits
        act = self.scheme.input_activations
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"w{bits}a{act.num_bits if act else 16}, group_size={self.scheme.weights.group_size}"
        )
