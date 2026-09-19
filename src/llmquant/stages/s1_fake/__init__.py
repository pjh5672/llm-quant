"""Phase 1 -- fake quant: quantize then dequantize back to bf16.

Measures what a bit combination costs in accuracy without needing any int kernel. The
weight is dequantized once at load, so the forward is a plain bf16 matmul; that is why
timing this stage says nothing about deployed speed (see docs, "속도 측정").
"""

from llmquant.stages.s1_fake.fake_quant_cache import FakeQuantCache, make_cache_factory
from llmquant.stages.s1_fake.fake_quant_linear import FakeQuantLinear

__all__ = ["FakeQuantCache", "FakeQuantLinear", "make_cache_factory"]
