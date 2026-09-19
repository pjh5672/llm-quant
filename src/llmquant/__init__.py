"""Symmetric RTN quantization, organized as core / stages / eval.

  core    everything every stage shares: config, schemes, scale computation, the
          quant-dequant math, the modifier that rewires a model, cost metrics
  stages  one folder per phase of docs/w4a8_rtn_notes.md, matching the `mode` config
          field: s1_fake -> s2_real -> s3_pack -> s4_kernel -> s5_chat
  eval    measuring a quantized model: PPL, LAMBADA, generation, GEMM speed, analysis
"""

from llmquant.core.config import DatasetArgs, ModelArgs, QuantConfig
from llmquant.core.modifier import QuantizationModifier
from llmquant.core.oneshot import oneshot
from llmquant.eval.evaluate import evaluate_ppl, generate

__all__ = [
    "DatasetArgs",
    "ModelArgs",
    "QuantConfig",
    "QuantizationModifier",
    "evaluate_ppl",
    "generate",
    "oneshot",
]
