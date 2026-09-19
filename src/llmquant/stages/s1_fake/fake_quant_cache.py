"""Fake-quantized KV cache.

Keys and values are quantized as they are written and immediately dequantized, so the rest
of attention is untouched and the accuracy cost is measurable without an int kernel -- the
same trick FakeQuantLinear uses for weights.

Granularity is per token and per head. The group size is the project-wide 128, and a head
narrower than that is zero-padded up to it -- head_dim is 64 here. That padding is free:
the scale is a symmetric abs-max and a zero cannot move it, so the result is bit-identical
to grouping at 64. What it buys is one consistent rule across weights, activations and the
cache, rather than a special case for the one axis that happens to be short.

Only the newly written tokens are quantized on each call. Everything already in the cache
was quantized when it was written, which is what a real quantized cache does too -- errors
do not accumulate by re-quantizing, but they also never get corrected.
"""

from transformers.cache_utils import DynamicCache

from llmquant.core.quant_ops import fake_quantize
from llmquant.core.scheme import GROUP_SIZE, QuantizationArgs


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
