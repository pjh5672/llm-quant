"""Quantized Mixture-of-Experts weights, read by the same kernel the dense path uses.

A MoE looked like it needed a batched GEMM with routing built in. It does not.
`MixtralExperts.forward` already loops over the experts a batch actually hit and calls
`F.linear(tokens_for_this_expert, gate_up_proj[e])` on each -- and `gate_up_proj[e]` is an
ordinary `[out, in]` weight with the reduction axis last. So the existing weight-only GEMV
runs it unchanged, once per hit expert, and no new CUDA is needed.

What is stacked is the storage, not the arithmetic. One `[E, out, in]` parameter becomes E
canonical `[out, groups, group_size]` int tensors with their own scales, which is also what
makes each expert's dynamic range its own: grouping along the reduction axis of the stack
gives per-expert scales for free.

The router is not here and must not be. It is a `[num_experts, hidden]` matrix whose output
is argmaxed into a discrete choice, so an error in it does not perturb a value -- it sends
the token to a different expert entirely.
"""

import torch
import torch.nn as nn

from llmquant.core.observers import group_view
from llmquant.s2_real.real_quant_linear import quantize_weight
from llmquant.core.scheme import QuantizationScheme
from llmquant.s4_kernel.build import load_extension
from llmquant.s4_kernel.quant_linear import KERNEL_MAX_ROWS, pad_activation

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
        self._gemv = load_extension().wq_gemv
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

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """The routing of MixtralExperts.forward, with our kernel doing the projections."""
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
