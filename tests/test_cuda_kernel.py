"""The CUDA quant-dequant kernel must be BIT-EXACT with the PyTorch reference.

Not "close": every rule the reference fixes (fp32 scale math, true IEEE division,
round-half-to-even, clamp order, round-to-nearest-even cast back) has to survive into
the kernel, because Phase 4 checks whole-model PPL for exact equality.
"""

import pytest
import torch

from llmquant.core import QuantizationArgs
from llmquant.core.observers import compute_scale
from llmquant.core.quant_ops import fake_quantize
from llmquant.core.scheme import GROUP_SIZE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@pytest.fixture(scope="module")
def cuda_fake_quantize():
    from llmquant.cuda import fake_quantize_cuda

    return fake_quantize_cuda


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape",
    [(16, GROUP_SIZE), (2048, 2048), (512, 2048), (2048, 8192), (2, 7, 2048)],
    ids=["single-group", "square", "kv-proj", "down-proj", "activation-3d"],
)
def test_matches_reference_bit_exactly(cuda_fake_quantize, bits, dtype, shape):
    torch.manual_seed(0)
    args = QuantizationArgs(num_bits=bits, strategy="group", group_size=GROUP_SIZE)
    x = torch.randn(*shape, device="cuda", dtype=dtype) * 3

    got, scale = cuda_fake_quantize(x, args, return_scale=True)
    assert torch.equal(got, fake_quantize(x, args))
    assert torch.equal(scale, compute_scale(x, args).squeeze(-1))
    assert got.dtype == x.dtype and got.shape == x.shape


@pytest.mark.parametrize("group_size", [32, 128, 256, 2048])
def test_matches_reference_for_other_group_sizes(cuda_fake_quantize, group_size):
    torch.manual_seed(0)
    args = QuantizationArgs(num_bits=4, strategy="group", group_size=group_size)
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(cuda_fake_quantize(x, args), fake_quantize(x, args))


def test_channel_strategy_uses_a_single_group(cuda_fake_quantize):
    torch.manual_seed(0)
    args = QuantizationArgs(num_bits=4, strategy="channel")
    w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(cuda_fake_quantize(w, args), fake_quantize(w, args))


@pytest.mark.parametrize(
    "name,factory",
    [
        ("all-zero group hits the eps floor", lambda: torch.zeros(4, GROUP_SIZE)),
        ("subnormal-ish values", lambda: torch.full((4, GROUP_SIZE), 1e-9)),
        ("near-overflow values", lambda: torch.full((4, GROUP_SIZE), 3e38)),
        ("exact .5 ties round half to even", lambda: torch.arange(512.0).reshape(4, 128) - 256 + 0.5),
    ],
)
def test_edge_cases(cuda_fake_quantize, name, factory):
    args = QuantizationArgs(num_bits=8, strategy="group", group_size=GROUP_SIZE)
    x = factory().cuda()
    assert torch.equal(cuda_fake_quantize(x, args), fake_quantize(x, args)), name


def test_non_contiguous_input(cuda_fake_quantize):
    torch.manual_seed(0)
    args = QuantizationArgs(num_bits=8, strategy="group", group_size=GROUP_SIZE)
    x = torch.randn(GROUP_SIZE, 64, device="cuda", dtype=torch.bfloat16).t()
    assert not x.is_contiguous()
    assert torch.equal(cuda_fake_quantize(x, args), fake_quantize(x, args))


def test_rejects_bad_inputs(cuda_fake_quantize):
    args = QuantizationArgs(num_bits=8, strategy="group", group_size=GROUP_SIZE)
    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        cuda_fake_quantize(torch.randn(4, GROUP_SIZE), args)
    with pytest.raises(RuntimeError, match="float32, float16 or bfloat16"):
        cuda_fake_quantize(torch.randn(4, GROUP_SIZE, device="cuda").double(), args)


def test_padded_shapes_still_match_the_reference(cuda_fake_quantize):
    """A last axis that is not a multiple of the group size gets the same zero padding on
    both sides, so the two must still agree bit for bit."""
    torch.manual_seed(0)
    args = QuantizationArgs(num_bits=8, strategy="group", group_size=GROUP_SIZE)
    for shape in [(4, 100), (2, 3, 64), (8, 192)]:
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        assert torch.equal(cuda_fake_quantize(x, args), fake_quantize(x, args)), shape
