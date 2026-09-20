import pytest
import torch.nn as nn

from llmquant.core.config import QuantConfig
from llmquant.core.config import BF16, dtype_to_args, normalize_dtype
from llmquant.core.scheme import GROUP_SIZE
from llmquant.stages.s1_fake import FakeQuantLinear


@pytest.mark.parametrize("value,expected", [(None, BF16), ("bf16", BF16), ("INT4", "int4"), (" int8 ", "int8")])
def test_normalize_dtype_accepts_null_and_case(value, expected):
    assert normalize_dtype(value) == expected


def test_normalize_dtype_rejects_unknown():
    with pytest.raises(ValueError, match="unknown dtype"):
        normalize_dtype("int3")


def test_dtype_to_args_uses_the_shared_group_size():
    args = dtype_to_args("int4", group_size=64, dynamic=True)
    assert (args.num_bits, args.strategy, args.group_size, args.dynamic) == (4, "group", 64, True)
    assert args.symmetric
    assert dtype_to_args("bf16", group_size=64, dynamic=False) is None


def test_bf16_weight_keeps_the_whole_target_bf16_including_activation():
    # an int8 activation feeding a bf16 weight has no int GEMM to run
    cfg = QuantConfig(attn_weight="int4", head_weight=BF16, activation="int8")
    assert cfg.scheme("head_weight") is None
    assert cfg.scheme("attn_weight").input_activations.num_bits == 8


def test_activation_shares_the_weight_group_size():
    cfg = QuantConfig(attn_weight="int4", activation="int8", group_size=64)
    scheme = cfg.scheme("attn_weight")
    assert scheme.weights.group_size == scheme.input_activations.group_size == 64


def test_weight_only_leaves_activation_unquantized():
    cfg = QuantConfig(mlp_weight="int4", activation=BF16)
    scheme = cfg.scheme("mlp_weight")
    assert scheme.weights.num_bits == 4 and scheme.input_activations is None


@pytest.mark.parametrize("dtype,bits", [(BF16, None), ("int8", 8)])
def test_kv_cache_resolves_to_a_bit_width(dtype, bits):
    cfg = QuantConfig(attn_weight="int4", kv_cache=dtype)
    assert cfg.kv_cache_bits == bits
    assert cfg.to_modifier().kv_cache_bits == bits


def test_kv_cache_is_not_part_of_a_weight_scheme():
    # the cache is grouped by head_dim, not by the weight group size, so it cannot ride
    # along in a QuantizationScheme
    cfg = QuantConfig(attn_weight="int4", kv_cache="int8")
    assert cfg.scheme("attn_weight").weights.group_size == cfg.group_size
    assert "kv" not in str(cfg.scheme("attn_weight"))


def test_rejects_bad_group_size():
    with pytest.raises(ValueError, match="positive int"):
        QuantConfig(attn_weight="int4", group_size=0)


def test_unknown_target_raises():
    with pytest.raises(ValueError, match="unknown target"):
        QuantConfig().scheme("embed_weight")


def test_describe_is_a_stable_label():
    cfg = QuantConfig(attn_weight="int8", mlp_weight="int4", activation="int8")
    assert cfg.describe() == f"attn-int8_mlp-int4_head-bf16_act-int8_g{GROUP_SIZE}"


class TinyLlama(nn.Module):
    """Module names mirror Llama so the attn / mlp patterns are exercised."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(GROUP_SIZE, GROUP_SIZE, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(GROUP_SIZE, GROUP_SIZE, bias=False)
        self.model.embed_tokens = nn.Linear(GROUP_SIZE, GROUP_SIZE, bias=False)
        self.lm_head = nn.Linear(GROUP_SIZE, GROUP_SIZE, bias=False)


@pytest.mark.parametrize(
    "head_weight,head_is_quantized", [(BF16, False), ("int8", True)]
)
def test_modifier_routes_each_target(head_weight, head_is_quantized):
    cfg = QuantConfig(attn_weight="int8", mlp_weight="int4", head_weight=head_weight, activation="int8")
    model = cfg.to_modifier().apply(TinyLlama())

    attn = model.model.layers[0].self_attn.q_proj
    mlp = model.model.layers[0].mlp.down_proj
    assert attn.scheme.weights.num_bits == 8
    assert mlp.scheme.weights.num_bits == 4
    assert isinstance(model.lm_head, FakeQuantLinear) is head_is_quantized
    # embed_tokens is a Linear here but matches neither attn nor mlp, so it stays bf16
    assert not isinstance(model.model.embed_tokens, FakeQuantLinear)


def test_all_bf16_config_changes_nothing():
    model = QuantConfig().to_modifier().apply(TinyLlama())
    assert not any(isinstance(m, FakeQuantLinear) for m in model.modules())


@pytest.mark.parametrize("dtype", ["int8", BF16, None])
def test_activation_accepts_int8_and_bf16(dtype):
    assert QuantConfig(attn_weight="int4", activation=dtype) is not None


def test_activation_rejects_int4():
    with pytest.raises(ValueError, match=r"activations support only"):
        QuantConfig(attn_weight="int4", activation="int4")


def test_sweep_grid_with_int4_activation_is_rejected(tmp_path):
    import yaml

    from llmquant.core.parser import build_config, expand_sweep

    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump({"defaults": {"project": "p"}, "sweep": {"activation": ["int8", "int4"]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"activations support only"):
        expand_sweep(build_config(["--cfg", str(path)], root_dir=tmp_path))


def test_an_int4_kv_cache_is_refused():
    """Measured, not assumed: int4 costs +23.23% accuracy alone and falls below random
    guessing with int4 MLP weights, while perplexity reports +0.18% because it never reads
    the cache back."""
    with pytest.raises(ValueError, match=r"kv_cache='int4'"):
        QuantConfig(kv_cache="int4")


@pytest.mark.parametrize("dtype", ["int8", "bf16", None])
def test_the_supported_kv_cache_dtypes_are_accepted(dtype):
    assert QuantConfig(kv_cache=dtype).kv_cache_bits in (8, None)


def test_an_int4_lm_head_is_refused():
    """lm_head writes the distribution the sampler reads and int4 there is unmeasured;
    attn and mlp are where int4 gets tried."""
    with pytest.raises(ValueError, match=r"head_weight='int4'"):
        QuantConfig(head_weight="int4")


@pytest.mark.parametrize("dtype", ["int8", "bf16", None])
def test_the_supported_head_dtypes_are_accepted(dtype):
    QuantConfig(head_weight=dtype)
