"""The KV cache is the one axis that only shows up once the model generates.

A perplexity pass never reads the cache back, so before accuracy moved to generation tasks
this change was literally unmeasurable.
"""

import pytest
import torch

from llmquant.core.quant_ops import fake_quantize
from llmquant.core.scheme import QuantizationArgs
from llmquant.stages.s1_fake import FakeQuantCache, make_cache_factory

HEAD_DIM = 64
SHAPE = (1, 2, 5, HEAD_DIM)  # batch, kv heads, seq, head_dim


def test_bf16_asks_for_no_cache_at_all():
    assert make_cache_factory(None) is None
    assert isinstance(make_cache_factory(8)(), FakeQuantCache)


@pytest.mark.parametrize("bits", [4, 8])
def test_update_stores_quantized_keys_and_values(bits):
    torch.manual_seed(0)
    keys, values = torch.randn(*SHAPE), torch.randn(*SHAPE)
    cached_keys, cached_values = FakeQuantCache(bits).update(keys, values, 0)

    args = QuantizationArgs(num_bits=bits, strategy="group", group_size=128, dynamic=True)
    assert torch.equal(cached_keys, fake_quantize(keys, args))
    assert torch.equal(cached_values, fake_quantize(values, args))
    assert not torch.equal(cached_keys, keys)  # something actually happened


def test_the_group_is_128_with_a_short_head_padded_up_to_it():
    cache = FakeQuantCache(8)
    assert cache._args(torch.randn(*SHAPE)).group_size == 128


@pytest.mark.parametrize("bits", [4, 8])
def test_padding_a_64_wide_head_to_128_changes_nothing(bits):
    """Zero padding cannot move a symmetric abs-max, so this matches grouping at head_dim."""
    torch.manual_seed(0)
    keys = torch.randn(*SHAPE)
    padded, _ = FakeQuantCache(bits).update(keys.clone(), keys.clone(), 0)
    exact = fake_quantize(
        keys, QuantizationArgs(num_bits=bits, strategy="group", group_size=HEAD_DIM, dynamic=True)
    )
    assert torch.equal(padded, exact)


def test_fewer_bits_cost_more_error():
    torch.manual_seed(0)
    keys = torch.randn(*SHAPE)
    error = {}
    for bits in (8, 4):
        stored, _ = FakeQuantCache(bits).update(keys.clone(), keys.clone(), 0)
        error[bits] = (stored - keys).abs().mean().item()
    assert error[4] > error[8] > 0


def test_tokens_already_in_the_cache_are_not_requantized():
    """Only the new tokens are quantized on each call, as a real quantized cache does.

    Re-quantizing what is already stored would compound the error every step.
    """
    torch.manual_seed(0)
    cache = FakeQuantCache(4)
    first = torch.randn(1, 2, 3, HEAD_DIM)
    stored_first, _ = cache.update(first, first.clone(), 0)
    kept = stored_first.clone()

    second = torch.randn(1, 2, 2, HEAD_DIM)
    stored_all, _ = cache.update(second, second.clone(), 0)
    assert torch.equal(stored_all[:, :, : first.shape[2]], kept)
