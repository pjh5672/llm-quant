"""Phase 3 -- packing and the single-file .bin format.

Nibble packing for int4, a [magic][header][tensors] writer, and a memory-mapped loader
that rebuilds the model without ever materializing bf16 weights. Weights are stored in the
canonical layout the kernel consumes, so nothing is reshaped at load and there is no second
place that could disagree with core.layout about where a group begins.
"""

from llmquant.s3_pack.format import read_header, read_packed, write_packed
from llmquant.s3_pack.model_io import (
    packed_dtypes,
    describe_packed,
    load_packed_model,
    save_packed_model,
)
from llmquant.s3_pack.packing import (
    pack_int4,
    pack_weight,
    unpack_int4,
    unpack_weight,
)

__all__ = [
    "packed_dtypes",
    "describe_packed",
    "load_packed_model",
    "pack_int4",
    "pack_weight",
    "read_header",
    "read_packed",
    "save_packed_model",
    "unpack_int4",
    "unpack_weight",
    "write_packed",
]
