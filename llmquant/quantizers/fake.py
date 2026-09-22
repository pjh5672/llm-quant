"""Fake quant: quantize, dequantize, and carry on in bf16.

The arithmetic is the quantized arithmetic while the model still runs as a normal bf16
model, which makes this the cheapest way to ask what a dtype combination costs in accuracy
-- and the only thing it is good for. Its speed numbers describe the simulation: every
combination stores the same bf16 bytes, so decode is identical across a grid and int8
activations measure slower for the quant they add.
"""

import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from llmquant.core.quant_ops import fake_quantize
from llmquant.core.scheme import GROUP_SIZE, QuantizationArgs, QuantizationScheme


class FakeQuantLinear(nn.Module):
    """Weights stored already dequantized; activations fake-quantized per forward."""

    def __init__(self, weight, bias, scheme: QuantizationScheme):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = bias
        self.scheme = scheme

    @classmethod
    def from_linear(cls, linear: nn.Linear, scheme: QuantizationScheme):
        w = linear.weight.detach()
        w = fake_quantize(w, scheme.weights) if scheme.weights else w.clone()
        return cls(w, linear.bias, scheme)

    def forward(self, x):
        if self.scheme.input_activations:
            x = fake_quantize(x, self.scheme.input_activations)
        return F.linear(x, self.weight, self.bias)


class FakeQuantCache(DynamicCache):
    """DynamicCache that fake-quantizes keys and values on write."""

    def __init__(self, num_bits: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits

    def _args(self, states):
        # the last axis is one head; a short head is padded up to a whole group
        return QuantizationArgs(
            num_bits=self.num_bits, strategy="group", group_size=GROUP_SIZE, dynamic=True
        )

    def update(self, key_states, value_states, layer_idx: int, *args, **kwargs):
        key_states = fake_quantize(key_states, self._args(key_states))
        value_states = fake_quantize(value_states, self._args(value_states))
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


def make_cache_factory(num_bits: int | None):
    """A zero-arg factory for generate(past_key_values=...), or None to leave the cache alone."""
    if num_bits is None:
        return None
    return lambda: FakeQuantCache(num_bits)
