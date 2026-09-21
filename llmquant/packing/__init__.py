"""One file holding a whole quantized model.

int4 goes two to a byte, the padding and head splits are already applied, and every tensor
starts 64-byte aligned so the file can be mapped and handed to the GPU without a realigning
copy. The header records the model id and the dtype of every layer, so a packed file loads
without the config that produced it.
"""

from llmquant.packing.format import read_header, read_packed, write_packed
from llmquant.packing.model_io import (
    describe_packed,
    load_packed_model,
    packed_dtypes,
    save_packed_model,
)
from llmquant.packing.packing import pack_int4, pack_weight, unpack_int4, unpack_weight

__all__ = [
    "describe_packed",
    "load_packed_model",
    "pack_int4",
    "pack_weight",
    "packed_dtypes",
    "read_header",
    "read_packed",
    "save_packed_model",
    "unpack_int4",
    "unpack_weight",
    "write_packed",
]
