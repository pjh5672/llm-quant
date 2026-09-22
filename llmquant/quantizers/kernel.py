"""Kernel quant: integer weights read straight from memory by CUDA.

The deployed path. Weights stay int into the kernel and are dequantized in registers, so
nothing at bf16 width is ever written or read back. Below a few rows the hand-written GEMV
runs; above it the weight is dequantized for cuBLAS, which is where prefill pays.
"""

import torch.nn as nn
from llmquant.core.observers import group_view, pad_to_group
from llmquant.core.scheme import QuantizationScheme
from llmquant.quantizers.real import quantize_weight
from llmquant.cuda.build import load_extension
import torch


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


__all__ = ["KernelQuantExperts", "quantize_expert_stack"]

# the two stacked projections transformers gives a Mixtral block, and the older split names
STACK_NAMES = ("gate_up_proj", "down_proj", "w1", "w2", "w3")


def quantize_expert_stack(stack: torch.Tensor, args):
    """[E, out, in] -> canonical int [E, out, groups, group_size] and fp32 scales.

    Each expert is quantized independently, which falls out of grouping on the last axis:
    the scale of expert 3's row 7 is computed from expert 3's row 7 and nothing else.
    """
    experts = stack.shape[0]
    qslices, sslices = [], []
    for e in range(experts):
        qweight, scale = quantize_weight(stack[e].detach(), args)
        canonical = group_view(qweight, args.group_size, args.head_dim).contiguous()
        qslices.append(canonical)
        sslices.append(scale.reshape(-1, canonical.shape[1]).contiguous().float())
    return torch.stack(qslices), torch.stack(sslices)


class KernelQuantExperts(nn.Module):
    """Stands in for MixtralExperts, holding int weights instead of bf16 ones."""

    def __init__(self, qgate_up, sgate_up, qdown, sdown, scheme, hidden_dim, intermediate_dim,
                 act_fn):
        super().__init__()
        self.register_buffer("qgate_up", qgate_up)  # [E, 2I, groups, gs] int8
        self.register_buffer("sgate_up", sgate_up)  # [E, rows, groups]   fp32
        self.register_buffer("qdown", qdown)
        self.register_buffer("sdown", sdown)
        self.scheme = scheme
        self.num_experts = qgate_up.shape[0]
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        self._weight_args = scheme.weights
        extension = load_extension()
        self._gemv = extension.wq_gemv
        self._gemv_batched = extension.wq_gemv_batched
        self._batched_max_rows = extension.wq_gemv_batched_max_rows()
        self._gate_up_padded = qgate_up.shape[2] * qgate_up.shape[3]
        self._down_padded = qdown.shape[2] * qdown.shape[3]

    @classmethod
    def from_experts(cls, experts: nn.Module, scheme: QuantizationScheme):
        if scheme.weights is None:
            raise ValueError("the kernel path needs a weight scheme")
        args = scheme.weights
        qgate_up, sgate_up = quantize_expert_stack(experts.gate_up_proj.data, args)
        qdown, sdown = quantize_expert_stack(experts.down_proj.data, args)
        return cls(
            qgate_up, sgate_up, qdown, sdown, scheme,
            hidden_dim=experts.gate_up_proj.shape[2],
            intermediate_dim=experts.down_proj.shape[2],
            act_fn=experts.act_fn,
        )

    def _matmul(self, x, qweight, wscale, real_in, padded_in):
        """One expert's projection, through the kernel when the token count is small."""
        if real_in != padded_in:
            x = pad_activation(x, self._weight_args)
        if x.shape[0] <= KERNEL_MAX_ROWS:
            return self._gemv(x, qweight, wscale, 0)
        rows_per_scale = qweight.shape[0] // wscale.shape[0]
        broadcast = wscale.repeat_interleave(rows_per_scale, dim=0).unsqueeze(-1)
        weight = (qweight * broadcast).to(x.dtype).reshape(qweight.shape[0], -1)
        return x @ weight.T

    def _plan(self, top_k_index):
        """Which rows go to which expert, and the block list the batched kernel needs.

        Rows are laid out grouped by expert so a block never straddles two of them: a block
        keeps its slice of the weight in registers across its rows, and reloading halfway
        through would give that up.

        Built entirely on the device. The obvious version walks the counts in Python, which
        needs them on the host, and those three round trips cost 0.432 ms of an 0.820 ms
        block -- more than the two matmuls they were setting up. CUDA-event profiling cannot
        see a host stall, so the block measured fine while decode stayed slower than bf16.

        """
        rows = top_k_index.numel()
        flat = top_k_index.reshape(-1)
        order = torch.argsort(flat, stable=True)
        # not torch.bincount: on CUDA it reads the maximum back to the host to size its
        # output, even with minlength, and that host round trip is the one thing in this
        # block a CUDA graph cannot capture. scatter_add_ into a fixed-size buffer is the
        # same count with no synchronisation -- and the size is known, it is num_experts.
        counts = torch.zeros(self.num_experts, dtype=torch.int64, device=flat.device)
        counts.scatter_add_(0, flat, torch.ones_like(flat))
        starts = torch.cumsum(counts, 0) - counts

        # worst case: one expert takes every row. Slots beyond an expert's share carry zero
        # rows and the kernel returns on them immediately.
        limit = self._batched_max_rows
        slots = (rows + limit - 1) // limit
        offset = torch.arange(slots, device=flat.device, dtype=torch.int64) * limit

        block_expert = torch.arange(
            self.num_experts, device=flat.device, dtype=torch.int64
        ).repeat_interleave(slots)
        taken = offset.repeat(self.num_experts)
        per_expert = counts.repeat_interleave(slots)
        block_row0 = starts.repeat_interleave(slots) + taken
        block_rows = (per_expert - taken).clamp(0, limit)

        return order, (block_expert.int(), block_row0.int(), block_rows.int())

    def _batched(self, x, qweight, wscale, plan, real_in, padded_in):
        if real_in != padded_in:
            x = pad_activation(x, self._weight_args)
        return self._gemv_batched(x, qweight, wscale, *plan, 0)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Route, then run every hit expert in one launch per projection.

        The loop this replaces was not slow because of the matmuls. On OLMoE-1B-7B it spent
        4.01 ms per block against bf16's 0.955 ms, while the matmuls inside it came to about
        0.19 ms -- top-8 of 64, two projections, sixteen layers is 256 separate launches for
        one token. Above the row limit the kernel stops paying (each block would reread the
        weight per row) and the per-expert cuBLAS path runs instead.
        """
        # A token cannot pick the same expert twice, so no expert ever gets more rows than
        # there are tokens. That bounds the batched path from the shape alone -- asking the
        # data would mean reading a count back from the device, which is the stall this
        # whole rewrite was about.
        if hidden_states.shape[0] > self._batched_max_rows:
            return self._forward_looped(hidden_states, top_k_index, top_k_weights)

        with torch.no_grad():
            order, plan = self._plan(top_k_index)

        tokens_per_row = order // top_k_index.shape[-1]
        slot_per_row = order % top_k_index.shape[-1]
        rows = hidden_states[tokens_per_row]

        fused = self._batched(rows, self.qgate_up, self.sgate_up, plan,
                              self.hidden_dim, self._gate_up_padded)
        gate, up = fused.chunk(2, dim=-1)
        out = self._batched(self.act_fn(gate) * up, self.qdown, self.sdown, plan,
                            self.intermediate_dim, self._down_padded)
        out = out * top_k_weights[tokens_per_row, slot_per_row, None]

        final = torch.zeros_like(hidden_states)
        final.index_add_(0, tokens_per_row, out.to(final.dtype))
        return final

    def _forward_looped(self, hidden_states, top_k_index, top_k_weights):
        """One expert at a time, for the shapes the batched kernel is not built for."""
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            mask = mask.permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert in hit:
            expert = expert[0]
            if expert == self.num_experts:
                continue
            slot, token_idx = torch.where(mask[expert])
            tokens = hidden_states[token_idx]

            fused = self._matmul(
                tokens, self.qgate_up[expert], self.sgate_up[expert],
                self.hidden_dim, self._gate_up_padded,
            )
            gate, up = fused.chunk(2, dim=-1)
            activated = self.act_fn(gate) * up
            out = self._matmul(
                activated, self.qdown[expert], self.sdown[expert],
                self.intermediate_dim, self._down_padded,
            )
            out = out * top_k_weights[token_idx, slot, None]
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final

    def extra_repr(self):
        return (
            f"num_experts={self.num_experts}, hidden={self.hidden_dim}, "
            f"intermediate={self.intermediate_dim}, w{self.scheme.weights.num_bits}, "
            f"group_size={self.scheme.weights.group_size}"
        )
