import pytest
import torch
import torch.nn as nn

from llmquant import QuantizationModifier
from llmquant.core import QuantizationArgs, preset_name_to_scheme
from llmquant.core.observers import compute_scale, group_view
from llmquant.core.quant_ops import fake_quantize, quantize
from llmquant.core.scheme import GROUP_SIZE
from llmquant.quantizers import FakeQuantLinear


@pytest.mark.parametrize("bits", [4, 8])
def test_weight_quant_is_per_group_symmetric_and_in_range(bits):
    torch.manual_seed(0)
    w = torch.randn(16, 2 * GROUP_SIZE)
    args = preset_name_to_scheme(f"W{bits}A16").weights
    scale = compute_scale(w, args)
    q = quantize(group_view(w, args.group_size), scale, args)
    qmax = 2 ** (bits - 1) - 1

    assert args.group_size == GROUP_SIZE
    assert scale.shape == (16, 2, 1)
    assert torch.equal(q, q.round())
    assert q.min() >= -qmax - 1 and q.max() <= qmax
    # abs-max element of each group maps exactly to +-qmax
    assert torch.equal(q.abs().amax(dim=-1), torch.full((16, 2), float(qmax)))


def test_activation_scale_is_per_token_and_per_group():
    x = torch.randn(2, 5, 4 * GROUP_SIZE)
    args = preset_name_to_scheme("W8A8").input_activations
    assert args.dynamic and args.group_size == GROUP_SIZE
    # one scale per (token, group): depends only on that token, so it stays causal
    assert compute_scale(x, args).shape == (2, 5, 4, 1)


def test_group_size_equal_to_k_reproduces_per_channel():
    torch.manual_seed(0)
    w = torch.randn(8, 256)
    channel = QuantizationArgs(num_bits=4, strategy="channel")
    grouped = QuantizationArgs(num_bits=4, strategy="group", group_size=256)
    assert torch.equal(compute_scale(w, channel), compute_scale(w, grouped).squeeze(-2))
    assert torch.equal(fake_quantize(w, channel), fake_quantize(w, grouped))


def test_a_short_tail_is_padded_into_its_own_group():
    args = QuantizationArgs(num_bits=8, strategy="group", group_size=GROUP_SIZE)
    x = torch.randn(4, 100)
    assert compute_scale(x, args).shape == (4, 1, 1)  # 100 -> one padded group
    assert fake_quantize(x, args).shape == x.shape  # and the padding is dropped again


def test_padding_is_numerically_free():
    """Zero padding cannot move a symmetric abs-max, so a padded short group quantizes
    exactly as a group of its own real length would."""
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    padded = QuantizationArgs(num_bits=4, strategy="group", group_size=128)
    exact = QuantizationArgs(num_bits=4, strategy="group", group_size=64)
    assert torch.equal(fake_quantize(x, padded), fake_quantize(x, exact))


def test_group_size_is_required_and_exclusive():
    with pytest.raises(ValueError, match="requires a positive group_size"):
        QuantizationArgs(num_bits=4, strategy="group")
    with pytest.raises(ValueError, match="only valid for strategy"):
        QuantizationArgs(num_bits=4, strategy="channel", group_size=128)


def test_rounding_is_half_to_even():
    args = preset_name_to_scheme("W8A8").input_activations
    q = quantize(torch.tensor([0.5, 1.5, 2.5, -0.5]), torch.tensor(1.0), args)
    assert q.tolist() == [0.0, 2.0, 2.0, -0.0]


def test_scale_uses_true_division_not_reciprocal_multiply():
    # `tensor / 7` is a reciprocal multiply in PyTorch and lands 1 ulp off IEEE
    # division, which a CUDA kernel cannot reproduce. Guard the bit-exact rule.
    args = QuantizationArgs(num_bits=4, strategy="group", group_size=GROUP_SIZE)
    x = torch.zeros(1, GROUP_SIZE)
    x[0, 0] = 6.2982625961
    amax = x.abs().amax()
    assert torch.equal(compute_scale(x, args).flatten(), (amax / torch.tensor(7.0)).flatten())


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        preset_name_to_scheme("W3A8")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(GROUP_SIZE, 8, bias=False)
        self.lm_head = nn.Linear(GROUP_SIZE, 4, bias=False)


def test_modifier_replaces_targets_and_respects_lm_head_scheme():
    model = QuantizationModifier(scheme="W4A8").apply(TinyModel())
    assert isinstance(model.proj, FakeQuantLinear)
    assert isinstance(model.lm_head, nn.Linear)

    model = QuantizationModifier(scheme="W4A8", lm_head_scheme="W8A8").apply(TinyModel())
    assert isinstance(model.lm_head, FakeQuantLinear)
    assert model.lm_head.scheme.weights.num_bits == 8


def test_size_accounts_for_one_scale_per_group():
    from llmquant.core.metrics import SCALE_BYTES, _linear_bytes

    # out_features is a multiple of the group, so the out-axis padding is a no-op here
    linear = nn.Linear(4 * GROUP_SIZE, GROUP_SIZE, bias=False)
    scheme = preset_name_to_scheme("W4A16")
    weight_bytes = linear.weight.numel() * 4 // 8
    assert _linear_bytes(linear, scheme) == weight_bytes + GROUP_SIZE * 4 * SCALE_BYTES
