"""The exactness reference for quantized Mixture-of-Experts weights.

Same relationship to s4_kernel's KernelQuantExperts that RealQuantLinear has to
KernelQuantLinear: integers stored, arithmetic done the obviously-correct way, speed not a
goal. It exists so the kernel has something to be equal to.

The routing is deliberately identical to the kernel module's, so a comparison between them
isolates the projections. If the two disagreed about which experts were hit, a difference
in the output would say nothing about the arithmetic.
"""

import torch
import torch.nn as nn

from llmquant.core.scheme import QuantizationScheme
from llmquant.s2_real.real_quant_linear import exact_fp32_matmul

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
        from llmquant.s4_kernel.expert_linear import quantize_expert_stack

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
