"""Disk size, decode traffic and BPV disagree with each other; that is the point of them."""

import pytest
import torch.nn as nn

from llmquant.args import QuantConfig
from llmquant.utils.size import BF16_BITS, bits_per_element, model_metrics

GROUP = 128


class TinyLlama(nn.Module):
    """Names mirror Llama so the attn / mlp patterns match; lm_head is tied to the embedding."""

    def __init__(self, tied=True):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(GROUP, GROUP, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(GROUP, GROUP, bias=False)
        self.model.embed_tokens = nn.Embedding(64, GROUP)
        self.lm_head = nn.Linear(GROUP, 64, bias=False)
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
    q = model_metrics(TinyLlama(), QuantConfig(attn_weight="int8", mlp_weight="int8").to_modifier())
    assert q["bits_per_element"] < base["bits_per_element"]
    assert q["deployed_bytes"] < base["deployed_bytes"]
    assert q["decode_bytes_per_token"] < base["decode_bytes_per_token"]


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
