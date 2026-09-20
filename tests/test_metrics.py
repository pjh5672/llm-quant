"""Disk size, decode traffic and BPV disagree with each other; that is the point of them."""

import pytest
import torch
import torch.nn as nn

from llmquant.core.config import QuantConfig
from llmquant.core.metrics import BF16_BITS, bits_per_element, model_metrics

GROUP = 128
HEAD_DIM = 32


class TinyConfig:
    """Only the fields the KV cache sizing reads."""

    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    hidden_size = GROUP
    head_dim = 32


class TinyLlama(nn.Module):
    """Names mirror Llama so the attn / mlp patterns match; lm_head is tied to the embedding."""

    def __init__(self, tied=True):
        super().__init__()
        self.config = TinyConfig()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(GROUP, GROUP, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(GROUP, GROUP, bias=False)
        self.model.embed_tokens = nn.Embedding(GROUP, GROUP)
        self.lm_head = nn.Linear(GROUP, GROUP, bias=False)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


@pytest.mark.parametrize(
    "bits,group,expected",
    [(4, 128, 4.25), (8, 128, 8.25), (4, 32, 5.0), (None, 128, float(BF16_BITS))],
)
def test_bits_per_element_amortizes_the_scale(bits, group, expected):
    assert bits_per_element(bits, group) == expected


def test_bf16_model_reports_16_bits_per_element():
    m = model_metrics(TinyLlama(), None)
    assert m["bits_per_element"] == float(BF16_BITS)
    assert m["deployed_bytes"] == m["decode_bytes_per_token"]  # nothing quantized, no embedding


def test_quantizing_weights_lowers_all_three():
    base = model_metrics(TinyLlama(), None)
    q = model_metrics(
        TinyLlama(),
        QuantConfig(
            attn_weight="int8", mlp_weight="int8", qkv_out_scale_per_head=False
        ).to_modifier(),
    )
    assert q["bits_per_element"] < base["bits_per_element"]
    assert q["deployed_bytes"] < base["deployed_bytes"]
    assert q["decode_bytes_per_token"] < base["decode_bytes_per_token"]


def test_int8_qkv_buys_nothing_once_heads_are_padded_into_128_tiles():
    """A 64-wide head in a 128 tile stores two rows per real row, which exactly cancels the
    halving int8 was supposed to buy. int4 still wins, but 2x rather than 4x."""
    from llmquant.core.metrics import _linear_bytes

    linear = nn.Linear(GROUP, 4 * HEAD_DIM, bias=False)  # out = 4 heads x head_dim
    recipe = QuantConfig(attn_weight="int8").to_modifier()
    recipe.head_dim = HEAD_DIM
    scheme = recipe.scheme_for("model.layers.0.self_attn.q_proj")

    bf16_bytes = linear.weight.numel() * 2
    int8_bytes = _linear_bytes(linear, scheme)
    assert int8_bytes >= bf16_bytes  # the padding ate the whole saving


def test_bits_per_element_counts_the_layout_padding():
    """Otherwise BPV would read the same whether or not the padding is there, while the
    disk number moves."""
    padded = model_metrics(TinyLlama(), QuantConfig(attn_weight="int8").to_modifier())
    unpadded = model_metrics(
        TinyLlama(), QuantConfig(attn_weight="int8", qkv_out_scale_per_head=False).to_modifier()
    )
    assert padded["bits_per_element"] > unpadded["bits_per_element"]
    assert padded["deployed_bytes"] > unpadded["deployed_bytes"]


def test_quantizing_lm_head_grows_disk_but_shrinks_decode_when_tied():
    """The tie means quantizing lm_head adds a second copy on disk, yet decode reads lm_head
    in full every token while the embedding is only a row lookup."""
    head_bf16 = model_metrics(
        TinyLlama(), QuantConfig(attn_weight="int8", mlp_weight="int8").to_modifier()
    )
    head_int8 = model_metrics(
        TinyLlama(),
        QuantConfig(attn_weight="int8", mlp_weight="int8", head_weight="int8").to_modifier(),
    )
    assert head_int8["deployed_bytes"] > head_bf16["deployed_bytes"]
    assert head_int8["decode_bytes_per_token"] < head_bf16["decode_bytes_per_token"]


def test_untied_model_counts_the_embedding_once():
    untied = model_metrics(TinyLlama(tied=False), None)
    tied = model_metrics(TinyLlama(tied=True), None)
    assert untied["deployed_bytes"] > tied["deployed_bytes"]


@pytest.mark.parametrize(
    "bits,expected_ratio",
    [(None, 1.0), (8, 0.5), (4, 0.25)],
)
def test_kv_cache_bytes_scale_with_the_bit_width(bits, expected_ratio):
    from llmquant.core.metrics import kv_cache_bytes_per_token

    model = TinyLlama()
    reference = kv_cache_bytes_per_token(model, None)
    got = kv_cache_bytes_per_token(model, bits)
    # int8/int4 also carry one fp32 scale per head per token, so they land slightly above
    assert got >= reference * expected_ratio
    assert got <= reference * expected_ratio * 1.3


def test_kv_cache_costs_nothing_on_disk_but_grows_decode_with_context():
    from llmquant.core.metrics import decode_bytes_at_context

    bf16 = model_metrics(TinyLlama(), QuantConfig(attn_weight="int8", kv_cache="bf16").to_modifier())
    int8 = model_metrics(TinyLlama(), QuantConfig(attn_weight="int8", kv_cache="int8").to_modifier())
    assert bf16["deployed_bytes"] == int8["deployed_bytes"]  # the cache is not stored
    assert int8["kv_bytes_per_token"] < bf16["kv_bytes_per_token"]
    # at zero context the cache costs nothing; the gap opens as context grows
    assert decode_bytes_at_context(bf16, 0) == decode_bytes_at_context(int8, 0)
    assert decode_bytes_at_context(int8, 8192) < decode_bytes_at_context(bf16, 8192)


# ---------------------------------------------------------------- Mixture-of-Experts

class StackedExperts(nn.Module):
    """transformers 5.x stores experts as stacked Parameters, not Linear modules."""

    def __init__(self, experts, hidden, inter):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(experts, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.zeros(experts, hidden, inter))


def _moe_model(experts=8, live=2):
    model = TinyLlama()
    hidden = model.config.hidden_size
    block = nn.Module()
    block.experts = StackedExperts(experts, hidden, 4 * hidden)
    model.model.layers[0].block_sparse_moe = block
    model.config.num_local_experts = experts
    model.config.num_experts_per_tok = live
    return model


def test_stacked_experts_are_costed_as_quantized():
    """They are quantized in place rather than swapped, so the Linear walk never sees them
    -- and on a real MoE they are most of the model."""
    model = _moe_model()
    bf16 = model_metrics(model, QuantConfig().to_modifier())
    w8 = model_metrics(model, QuantConfig(attn_weight="int8", mlp_weight="int8").to_modifier())
    assert w8["deployed_bytes"] < bf16["deployed_bytes"]
    assert w8["bits_per_element"] < 16.0


def test_only_the_live_experts_count_toward_decode():
    """A router picks top_k of num_experts, so a dense read of every expert overstates
    decode traffic by num_experts / top_k."""
    model = _moe_model(experts=8, live=2)
    recipe = QuantConfig().to_modifier()
    routed = model_metrics(model, recipe)["decode_bytes_per_token"]

    dense = _moe_model(experts=8, live=8)
    dense_bytes = model_metrics(dense, QuantConfig().to_modifier())["decode_bytes_per_token"]
    assert routed < dense_bytes

    expert_bytes = sum(
        p.numel() * 2 for n, p in model.named_parameters() if "experts" in n
    )
    # the saving is exactly the three quarters of the experts that are not read
    assert dense_bytes - routed == pytest.approx(expert_bytes * 0.75, rel=0.01)


def test_the_baseline_is_routed_too():
    """Scaling top-k only on the quantized run would credit quantization with the router's
    work: the bf16 model reads the same two experts per token."""
    model = _moe_model()
    bf16 = model_metrics(model, QuantConfig().to_modifier())["decode_bytes_per_token"]
    w8 = model_metrics(
        model, QuantConfig(attn_weight="int8", mlp_weight="int8").to_modifier()
    )["decode_bytes_per_token"]
    assert 1.0 < bf16 / w8 < 2.5, f"implausible decode ratio {bf16 / w8:.1f}x"
