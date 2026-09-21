"""Shared by every stage. Nothing here may import a stage at module level.

The one exception is the mode -> module class lookup, which llmquant.quantizers.dispatch owns and
core.modifier imports lazily inside apply(); see the comment there.
"""

from llmquant.core.config import BF16, DTYPE_BITS, DatasetArgs, ModelArgs, QuantConfig
from llmquant.core.metrics import bits_per_element, estimate_deployed_bytes, model_metrics
from llmquant.core.modifier import QuantizationModifier
from llmquant.core.observers import SCALE_EPS, compute_scale, group_view, pad_to_group
from llmquant.core.oneshot import oneshot
from llmquant.core.quant_ops import dequantize, fake_quantize, quantize
from llmquant.core.scheme import (
    GROUP_SIZE,
    PRESET_SCHEMES,
    QuantizationArgs,
    QuantizationScheme,
    preset_name_to_scheme,
)
from llmquant.core.selection import SelectionConfig

__all__ = [
    "BF16",
    "DTYPE_BITS",
    "GROUP_SIZE",
    "PRESET_SCHEMES",
    "DatasetArgs",
    "ModelArgs",
    "QuantConfig",
    "QuantizationArgs",
    "QuantizationModifier",
    "QuantizationScheme",
    "SCALE_EPS",
    "SelectionConfig",
    "bits_per_element",
    "compute_scale",
    "dequantize",
    "estimate_deployed_bytes",
    "fake_quantize",
    "group_view",
    "model_metrics",
    "pad_to_group",
    "oneshot",
    "preset_name_to_scheme",
    "quantize",
]
