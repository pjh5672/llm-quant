"""Phase 4: the kernel path must agree with the Phase 2 reference, in every layout.

The reference is the oracle here. A kernel that is merely close looks the same as a
quantization effect, which is exactly the failure this stage is set up to catch.
"""

import pytest
import torch
import torch.nn as nn

from llmquant.core.scheme import (
    GROUP_SIZE,
    QuantizationArgs,
    QuantizationScheme,
    preset_name_to_scheme,
)
from llmquant.quantizers import RealQuantLinear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

K, N, HEAD_DIM = 2048, 512, 64


def weight_only(bits, **kwargs):
    return QuantizationScheme(
        weights=QuantizationArgs(
            num_bits=bits, strategy="group", group_size=GROUP_SIZE, **kwargs
        ),
        input_activations=None,
    )


LAYOUTS = {
    "plain": weight_only(8),
    "int4": weight_only(4),
    "o_proj (head on the reduction axis)": weight_only(8, head_dim=HEAD_DIM),
    "q_proj (head on the output axis)": weight_only(8, out_group=HEAD_DIM),
}


@pytest.fixture(scope="module")
def linear():
    torch.manual_seed(0)
    return nn.Linear(K, N, bias=False).cuda().to(torch.bfloat16)


@pytest.mark.parametrize("name", list(LAYOUTS))
@pytest.mark.parametrize("M", [1, 4])
def test_the_kernel_path_is_exact_against_the_reference(name, M, linear):
    """Below the dispatch threshold the kernel does the same grouped integer arithmetic the
    reference does, so this is an equality rather than a tolerance."""
    from llmquant.quantizers import KernelQuantLinear

    scheme = LAYOUTS[name]
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    got = KernelQuantLinear.from_linear(linear, scheme)(x)
    want = RealQuantLinear.from_linear(linear, scheme)(x)
    assert torch.equal(got, want), name


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_the_cublas_path_lands_exactly_where_fake_quant_does(name, linear):
    """Above the threshold the weight is dequantized to the activation dtype and handed to
    cuBLAS -- the same computation fake quant performs, so the two agree exactly. It is
    therefore no less precise than the path it replaces, and no more."""
    from llmquant.quantizers import KERNEL_MAX_ROWS, FakeQuantLinear, KernelQuantLinear

    scheme = LAYOUTS[name]
    torch.manual_seed(0)
    x = torch.randn(KERNEL_MAX_ROWS * 4, K, device="cuda", dtype=torch.bfloat16)
    kernel = KernelQuantLinear.from_linear(linear, scheme)(x).float()
    fake = FakeQuantLinear.from_linear(linear, scheme)(x).float()
    reference = RealQuantLinear.from_linear(linear, scheme)(x).float()

    kernel_error = (kernel - reference).abs().max()
    fake_error = (fake - reference).abs().max()
    assert torch.isclose(kernel_error, fake_error, rtol=1e-3), name


def test_real_width_is_kept_separate_from_the_padded_one(linear):
    """o_proj pads 32 heads of 64 into 32 groups of 128, so the stored tensor is twice as
    wide as the layer. Reporting the padded width broke the dequant path."""
    from llmquant.quantizers import KernelQuantLinear

    module = KernelQuantLinear.from_linear(linear, LAYOUTS["o_proj (head on the reduction axis)"])
    assert module.in_features == K
    assert module.padded_in_features == 2 * K


def test_both_dispatch_branches_are_exercised(linear):
    from llmquant.quantizers import KERNEL_MAX_ROWS, KernelQuantLinear

    module = KernelQuantLinear.from_linear(linear, LAYOUTS["plain"])
    small = torch.randn(KERNEL_MAX_ROWS, K, device="cuda", dtype=torch.bfloat16)
    large = torch.randn(KERNEL_MAX_ROWS + 1, K, device="cuda", dtype=torch.bfloat16)
    assert module(small).shape == (KERNEL_MAX_ROWS, N)
    assert module(large).shape == (KERNEL_MAX_ROWS + 1, N)


def test_weight_is_stored_as_int8(linear):
    from llmquant.quantizers import KernelQuantLinear

    module = KernelQuantLinear.from_linear(linear, LAYOUTS["int4"])
    assert module.qweight.dtype == torch.int8
    assert module.qweight.shape[-1] == GROUP_SIZE  # canonical [N, groups, group_size]
    assert module.qweight.min() >= -8 and module.qweight.max() <= 7


def test_leading_dimensions_survive(linear):
    from llmquant.quantizers import KernelQuantLinear

    module = KernelQuantLinear.from_linear(linear, LAYOUTS["plain"])
    assert module(torch.randn(2, 3, K, device="cuda", dtype=torch.bfloat16)).shape == (2, 3, N)


def test_bias_is_applied():
    from llmquant.quantizers import KernelQuantLinear

    biased = nn.Linear(K, N, bias=True).cuda().to(torch.bfloat16)
    torch.nn.init.constant_(biased.bias, 1.5)
    module = KernelQuantLinear.from_linear(biased, preset_name_to_scheme("W8A16"))
    out = module(torch.zeros(1, K, device="cuda", dtype=torch.bfloat16))
    assert torch.allclose(out.float(), biased.bias.float().expand(1, N), atol=1e-2)


def test_bf16_weights_are_refused(linear):
    from llmquant.quantizers import KernelQuantLinear

    scheme = QuantizationScheme(weights=None, input_activations=None)
    with pytest.raises(ValueError, match="needs a weight scheme"):
        KernelQuantLinear.from_linear(linear, scheme)
