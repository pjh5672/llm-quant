"""Phase 2 -- real quant reference.

RealQuantLinear: int weights kept as integers, with the matmul done by verified
PyTorch ops. Pass criterion: its accuracy matches the fake-quant accuracy to float precision; a
larger gap means a bug rather than a quantization effect.

With group_size=128 a group's partial sum peaks near 2.06M, inside fp32's exact-integer
range, so both W4A8 and W8A8 can use a plain fp32 matmul and stay bit-exact (TF32 off).
"""

from llmquant.s2_real.real_quant_linear import (
    RealQuantLinear,
    exact_fp32_matmul,
    quantize_weight,
)

__all__ = ["RealQuantLinear", "exact_fp32_matmul", "quantize_weight"]
