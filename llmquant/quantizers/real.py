"""Real quant: integer weights, arithmetic done the obviously-correct way.

The reference, not a deployment path. It keeps the integers and multiplies in fp32 with
TF32 off, which is slower than bf16 on purpose -- it exists to be the thing the CUDA
kernels are checked against, bit for bit.
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


__all__ = ["RealQuantExperts"]


class RealQuantExperts(nn.Module):
    """MixtralExperts with int weights and an fp32 matmul."""

    def __init__(self, qgate_up, sgate_up, qdown, sdown, scheme, hidden_dim, intermediate_dim,
                 act_fn):
        super().__init__()
        self.register_buffer("qgate_up", qgate_up)
        self.register_buffer("sgate_up", sgate_up)
        self.register_buffer("qdown", qdown)
        self.register_buffer("sdown", sdown)
        self.scheme = scheme
        self.num_experts = qgate_up.shape[0]
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn

    @classmethod
    def from_experts(cls, experts: nn.Module, scheme: QuantizationScheme):
        if scheme.weights is None:
            raise ValueError("the real path needs a weight scheme")
        from llmquant.quantizers import quantize_expert_stack

        args = scheme.weights
        qgate_up, sgate_up = quantize_expert_stack(experts.gate_up_proj.data, args)
        qdown, sdown = quantize_expert_stack(experts.down_proj.data, args)
        return cls(
            qgate_up, sgate_up, qdown, sdown, scheme,
            hidden_dim=experts.gate_up_proj.shape[2],
            intermediate_dim=experts.down_proj.shape[2],
            act_fn=experts.act_fn,
        )

    def _dequantized(self, qweight, wscale, real_in):
        rows_per_scale = qweight.shape[0] // wscale.shape[0]
        broadcast = wscale.repeat_interleave(rows_per_scale, dim=0).unsqueeze(-1)
        weight = (qweight.float() * broadcast).reshape(qweight.shape[0], -1)
        return weight[:, :real_in]

    def _matmul(self, x, qweight, wscale, real_in):
        weight = self._dequantized(qweight, wscale, real_in)
        with exact_fp32_matmul():
            return (x.float() @ weight.T).to(x.dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
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
                tokens, self.qgate_up[expert], self.sgate_up[expert], self.hidden_dim
            )
            gate, up = fused.chunk(2, dim=-1)
            activated = self.act_fn(gate) * up
            out = self._matmul(
                activated, self.qdown[expert], self.sdown[expert], self.intermediate_dim
            )
            out = out * top_k_weights[token_idx, slot, None]
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final
