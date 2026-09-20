"""Phase 4 -- the int weight is consumed by a CUDA kernel.

Dispatches on how many rows are being multiplied, because the two regimes want opposite
things and no single simple kernel serves both:

    decode, M small    memory bound. Every token re-reads the whole weight, so reading it
                       as int8 instead of bf16 halves the traffic. The kernel does that and
                       measures ~2.4x over fake quant at M=1.
    prefill, M large   compute bound. cuBLAS is on tensor cores at ~48 TFLOPS and a scalar
                       kernel cannot approach that, so the weight is dequantized once and
                       handed over. The dequant is O(N*K) and amortizes over M rows.

The crossover was measured, not guessed: the kernel wins at M=1 (2.4x), breaks even near
M=2 and loses from M=4 on. Beating cuBLAS above that needs a tensor-core mainloop, which is
a different and much larger piece of work -- and worth little here, since int8 math measured
only 1.03-1.04x over bf16 on this GPU. The win is bandwidth, and bandwidth only matters
where decode lives.
"""

import torch.nn as nn

from llmquant.core.observers import group_view, pad_to_group
from llmquant.core.scheme import QuantizationScheme
from llmquant.s2_real.real_quant_linear import quantize_weight
from llmquant.s4_kernel.build import load_extension


def pad_activation(flat, args):
    """Pad the activation exactly the way the weight was, heads included."""
    if args.head_dim:
        lead, k = flat.shape[:-1], flat.shape[-1]
        per_head = pad_to_group(
            flat.reshape(*lead, k // args.head_dim, args.head_dim), args.group_size
        )
        return per_head.reshape(*lead, -1)
    return pad_to_group(flat, args.group_size)

# below this many rows the kernel wins; above it cuBLAS does
KERNEL_MAX_ROWS = 4


class KernelQuantLinear(nn.Module):
    """Linear backed by int weights in the canonical [N, groups, group_size] layout."""

    def __init__(self, qweight, wscale, bias, scheme: QuantizationScheme, in_features: int):
        super().__init__()
        self.register_buffer("qweight", qweight)  # [N, groups, group_size] int8
        self.register_buffer("wscale", wscale)  # [scale_rows, groups] fp32
        self.bias = bias
        self.scheme = scheme
        self.out_features = qweight.shape[0]
        # the real width, not the canonical one: o_proj pads 32 heads of 64 up to 32 groups
        # of 128, so the stored tensor is twice as wide as the layer actually is
        self.in_features = in_features
        self.padded_in_features = qweight.shape[1] * qweight.shape[2]
        self.out_group = scheme.weights.out_group or 0
        # resolved once: decode runs this module ~112 times per token, so a dict lookup and
        # a couple of attribute walks per call are not free at 13ms a token
        self._gemv = load_extension().wq_gemv
        self._needs_padding = self.in_features != self.padded_in_features
        self._weight_args = scheme.weights
        rows_per_scale = qweight.shape[0] // wscale.shape[0]
        self.register_buffer(
            "_broadcast_scale", wscale.repeat_interleave(rows_per_scale, dim=0).unsqueeze(-1)
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, scheme: QuantizationScheme):
        if scheme.weights is None:
            raise ValueError("the kernel path needs a weight scheme; bf16 weights stay nn.Linear")
        args = scheme.weights
        qweight, scale = quantize_weight(linear.weight.detach(), args)
        # canonical form: the padding and any head split are baked in here, once, so the
        # kernel never has to know about heads
        canonical = group_view(qweight, args.group_size, args.head_dim).contiguous()
        flat_scale = scale.reshape(-1, canonical.shape[1]).contiguous().float()
        return cls(canonical, flat_scale, linear.bias, scheme, linear.in_features)

    def _dequantized_weight(self, dtype):
        """[out, padded_in], rebuilt for the cuBLAS path.

        In the activation dtype, not fp32: a first version materialized fp32 here and cost
        3x on TTFT, because prefill then wrote and read twice the bytes for every Linear in
        the model. Left padded rather than trimmed so it lines up with the padded
        activation; the padding columns are zero and contribute nothing.

        Not cached -- holding a dequantized copy would undo the memory saving that is the
        whole point of storing the weight as int.
        """
        return (self.qweight * self._broadcast_scale).to(dtype).reshape(self.out_features, -1)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if self._needs_padding:
            # the activation has to land in the same padded groups the weight was cut into
            flat = pad_activation(flat, self._weight_args)

        if flat.shape[0] <= KERNEL_MAX_ROWS:
            out = self._gemv(flat, self.qweight, self.wscale, self.out_group)
        else:
            out = flat @ self._dequantized_weight(flat.dtype).T

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*shape[:-1], self.out_features).to(x.dtype)

    def extra_repr(self):
        act = self.scheme.input_activations
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"w{self.scheme.weights.num_bits}a{act.num_bits if act else 16}, "
            f"group_size={self.scheme.weights.group_size}"
        )
