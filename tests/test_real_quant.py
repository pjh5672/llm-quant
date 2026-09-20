"""Phase 2: the weight stays an integer and the matmul consumes it as one.

The point of this stage is to be the oracle the Phase 4 kernel is checked against, so what
is pinned here is the arithmetic, not just the output being roughly right.
"""

import pytest
import torch
import torch.nn as nn

from llmquant.core.scheme import GROUP_SIZE, QuantizationArgs, preset_name_to_scheme
from llmquant.s1_fake import FakeQuantLinear
from llmquant.s2_real import RealQuantLinear, exact_fp32_matmul, quantize_weight

K, N, M = 4 * GROUP_SIZE, 256, 32
SCHEMES = ["W8A16", "W8A8", "W4A16", "W4A8"]


@pytest.fixture
def linear():
    torch.manual_seed(0)
    return nn.Linear(K, N, bias=False)


def test_weights_are_stored_as_integers(linear):
    args = preset_name_to_scheme("W4A16").weights
    qweight, scale = quantize_weight(linear.weight.detach(), args)
    assert qweight.dtype == torch.int8
    assert qweight.shape == linear.weight.shape
    assert qweight.min() >= -8 and qweight.max() <= 7  # int4 range in an int8 container
    assert scale.shape == (N, K // GROUP_SIZE, 1)


@pytest.mark.parametrize("name", SCHEMES)
def test_real_quant_tracks_fake_quant(name, linear):
    """Same scheme, so the two must land in the same place -- they differ only in how the
    sum is accumulated, not in what is being approximated."""
    torch.manual_seed(0)
    x = torch.randn(M, K) * 2
    scheme = preset_name_to_scheme(name)
    fake = FakeQuantLinear.from_linear(linear, scheme)(x).float()
    real = RealQuantLinear.from_linear(linear, scheme)(x).float()

    exact = x @ linear.weight.detach().T
    quantization_error = (fake - exact).abs().mean()
    gap = (real - fake).abs().mean()
    assert gap < quantization_error  # closer to each other than either is to the truth


@pytest.mark.parametrize("name", SCHEMES)
def test_real_quant_is_not_worse_than_fake_quant(name, linear):
    """Integers accumulate exactly and are scaled once per group, where fake quant rounds
    both sides first -- so real should never be meaningfully worse.

    Not asserted as a strict inequality: in fp32 the two accumulation orders land within
    float noise of each other and either can come out a hair ahead. The gap only opens up
    in the setting that matters, bf16 activations on the GPU, where the exact integer
    accumulation measurably wins.
    """
    torch.manual_seed(0)
    x = torch.randn(M, K) * 2
    scheme = preset_name_to_scheme(name)
    exact = x @ linear.weight.detach().T
    fake_error = (FakeQuantLinear.from_linear(linear, scheme)(x).float() - exact).abs().mean()
    real_error = (RealQuantLinear.from_linear(linear, scheme)(x).float() - exact).abs().mean()
    assert real_error <= fake_error * 1.01


def test_bias_is_applied(linear):
    biased = nn.Linear(K, N, bias=True)
    torch.nn.init.constant_(biased.bias, 1.5)
    scheme = preset_name_to_scheme("W8A16")
    x = torch.zeros(M, K)
    assert torch.allclose(RealQuantLinear.from_linear(biased, scheme)(x), biased.bias.expand(M, N))


def test_bf16_weights_are_refused(linear):
    scheme = preset_name_to_scheme("W8A16")
    bf16_only = type(scheme)(weights=None, input_activations=scheme.input_activations)
    with pytest.raises(ValueError, match="real quant needs a weight scheme"):
        RealQuantLinear.from_linear(linear, bf16_only)


def test_shapes_survive_extra_leading_dimensions(linear):
    scheme = preset_name_to_scheme("W4A8")
    module = RealQuantLinear.from_linear(linear, scheme)
    assert module(torch.randn(2, 5, K)).shape == (2, 5, N)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_a_group_partial_sum_is_exact_in_fp32():
    """The property the whole stage rests on: 128 * 127 * 127 fits inside 2^24."""
    a = torch.full((8, GROUP_SIZE), 127.0)
    b = torch.full((GROUP_SIZE, 16), 127.0)
    b[:, ::2] = -127.0
    reference = (a.long() @ b.long()).float()
    assert reference.abs().max() < 2**24

    with exact_fp32_matmul():
        assert torch.equal(a.cuda() @ b.cuda(), reference.cuda())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_tf32_does_not_change_an_int8_product():
    """Measured rather than assumed: TF32 holds 11 mantissa bits, and |int8| <= 127 < 2^11,
    so the inputs survive and the accumulator is fp32 either way."""
    torch.manual_seed(0)
    a = torch.randint(-127, 128, (8, GROUP_SIZE), dtype=torch.float32)
    b = torch.randint(-127, 128, (GROUP_SIZE, 16), dtype=torch.float32)
    reference = (a.long() @ b.long()).float().cuda()

    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        assert torch.equal(a.cuda() @ b.cuda(), reference)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_exact_fp32_matmul_restores_the_flag():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with exact_fp32_matmul():
            assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_head_aware_and_out_grouped_schemes_round_trip():
    """o_proj groups the reduction axis by head; q/k/v share a scale per output head."""
    torch.manual_seed(0)
    module = nn.Linear(K, N, bias=False)
    for args in [
        QuantizationArgs(num_bits=4, strategy="group", group_size=GROUP_SIZE, head_dim=64),
        QuantizationArgs(num_bits=4, strategy="group", group_size=GROUP_SIZE, out_group=64),
    ]:
        qweight, scale = quantize_weight(module.weight.detach(), args)
        assert qweight.shape == module.weight.shape
        assert qweight.min() >= -8 and qweight.max() <= 7
        assert scale.numel() < module.weight.numel()
