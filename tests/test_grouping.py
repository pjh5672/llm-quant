"""Padding and head-aware grouping.

Two separate ideas that both changed how the last axis is cut up:

  padding     a tail shorter than the group size is zero-padded into a whole group
  head_dim    o_proj's K axis is heads x head_dim, so its groups stay inside a head
"""

import pytest
import torch
import torch.nn as nn

from llmquant.core.config import QuantConfig
from llmquant.core.metrics import model_metrics
from llmquant.core.modifier import QuantizationModifier
from llmquant.core.observers import compute_scale, group_view, pad_to_group, ungroup
from llmquant.core.quant_ops import fake_quantize
from llmquant.core.scheme import QuantizationArgs

HEADS, HEAD_DIM, GROUP = 32, 64, 128
K = HEADS * HEAD_DIM  # 2048, the o_proj input


def args(group_size=GROUP, head_dim=None, bits=4):
    return QuantizationArgs(
        num_bits=bits, strategy="group", group_size=group_size, head_dim=head_dim
    )


@pytest.mark.parametrize("k,expected", [(128, 128), (100, 128), (129, 256), (256, 256)])
def test_pad_to_group_rounds_the_last_axis_up(k, expected):
    assert pad_to_group(torch.randn(3, k), GROUP).shape[-1] == expected


def test_padding_cannot_move_a_symmetric_abs_max():
    torch.manual_seed(0)
    x = torch.randn(4, HEAD_DIM)
    # a 64-wide tail padded into a 128 group quantizes exactly as a 64 group would
    assert torch.equal(fake_quantize(x, args(GROUP)), fake_quantize(x, args(HEAD_DIM)))


def test_group_view_and_ungroup_round_trip():
    for k, head_dim in [(K, None), (K, HEAD_DIM), (100, None)]:
        x = torch.randn(2, k)
        view = group_view(x, GROUP, head_dim)
        assert view.shape[-1] == GROUP
        assert torch.equal(ungroup(view, x.shape, GROUP, head_dim), x)


def test_head_aware_grouping_keeps_heads_apart():
    x = torch.randn(4, K)
    # flat: 2 heads per 128 group. head-aware: one group per head
    assert compute_scale(x, args()).shape == (4, K // GROUP, 1)
    assert compute_scale(x, args(head_dim=HEAD_DIM)).shape == (4, HEADS, 1)


def test_head_aware_grouping_is_more_accurate_than_straddling_two_heads():
    torch.manual_seed(0)
    # heads with very different dynamic ranges: sharing a scale wastes range on the small one
    x = torch.randn(4, HEADS, HEAD_DIM)
    x[:, ::2] *= 50
    x = x.reshape(4, K)

    flat_error = (fake_quantize(x, args()) - x).abs().mean()
    head_error = (fake_quantize(x, args(head_dim=HEAD_DIM)) - x).abs().mean()
    assert head_error < flat_error


def test_head_dim_must_divide_the_axis():
    with pytest.raises(ValueError, match="not a multiple of head_dim"):
        group_view(torch.randn(2, 100), GROUP, HEAD_DIM)


class TinyConfig:
    num_hidden_layers = 1
    num_attention_heads = 2
    num_key_value_heads = 2
    hidden_size = K
    head_dim = HEAD_DIM


class TinyLlama(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TinyConfig()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(K, K, bias=False)
        layer.self_attn.o_proj = nn.Linear(K, K, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(K, K, bias=False)
        self.model.embed_tokens = nn.Embedding(GROUP, K)
        self.lm_head = nn.Linear(K, GROUP, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_only_o_proj_is_grouped_by_head():
    recipe = QuantConfig(attn_weight="int4", mlp_weight="int4", activation="int8").to_modifier()
    recipe.resolve(TinyLlama())
    assert recipe.head_dim == HEAD_DIM

    o_proj = recipe.scheme_for("model.layers.0.self_attn.o_proj")
    q_proj = recipe.scheme_for("model.layers.0.self_attn.q_proj")
    mlp = recipe.scheme_for("model.layers.0.mlp.down_proj")
    # both sides of o_proj, or the int GEMM would not decompose into matching groups
    assert o_proj.weights.head_dim == HEAD_DIM and o_proj.input_activations.head_dim == HEAD_DIM
    assert q_proj.weights.head_dim is None and mlp.weights.head_dim is None


def test_head_grouping_costs_extra_scales():
    """A 64-wide head fills a whole 128 group, so o_proj carries twice the scales.

    Checked on the accounting directly: model_metrics resolves head_dim off the model, so
    there is no way to ask it for the flat number once a model is in hand.
    """
    from llmquant.core.metrics import _num_scales

    linear = nn.Linear(K, K, bias=False)
    flat = _num_scales(linear, args())
    head_aware = _num_scales(linear, args(head_dim=HEAD_DIM))
    assert head_aware == flat * 2 == K * HEADS


def test_costing_a_model_resolves_head_dim_before_counting():
    """The extra o_proj scales are real but small; quantizing attn still shrinks the model."""
    model = TinyLlama()
    quantized = model_metrics(model, QuantConfig(attn_weight="int4").to_modifier())
    untouched = model_metrics(model, None)
    assert quantized["deployed_bytes"] < untouched["deployed_bytes"]


def test_resolve_is_a_no_op_once_head_dim_is_known():
    recipe = QuantizationModifier(attn_scheme="W4A16", head_dim=8)
    recipe.resolve(TinyLlama())
    assert recipe.head_dim == 8
