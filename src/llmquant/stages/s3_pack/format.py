"""The single-file weight format.

    [magic + version][header length, 8 bytes][JSON header][aligned raw tensor bytes]

One file holds the whole model: quantized weights in the canonical layout, their scales,
and everything that was never quantized (embeddings, norms) as bf16. The tokenizer and the
architecture come from the model id recorded in the header, so the file carries weights and
nothing else.

Two decisions worth stating, because a later reader cannot infer them from the bytes:

Weights are stored in the *canonical* layout -- [N, groups, group_size] with the reduction
axis already padded and any head split already applied by core.layout. The alternative,
storing the logical [out, in] weight and reshaping at load, would leave two places that
have to agree on padding and head boundaries. They would eventually stop agreeing, and the
symptom would be wrong numbers rather than an error.

Every tensor starts at a 64-byte boundary so the file can be memory-mapped and handed to
the GPU without a copy to realign it.
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch

MAGIC = b"LLMQUANT"
VERSION = 1
ALIGNMENT = 64
HEADER_LENGTH_BYTES = 8

_TORCH_DTYPE = {
    "uint8": torch.uint8,
    "int8": torch.int8,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _align(offset: int) -> int:
    return -(-offset // ALIGNMENT) * ALIGNMENT


def _to_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().cpu().contiguous()
    if flat.dtype == torch.bfloat16:
        # numpy has no bfloat16; the raw 16 bits round-trip through uint16 untouched
        return flat.view(torch.uint16).numpy().tobytes()
    return flat.numpy().tobytes()


def _from_buffer(buffer: memoryview, entry: dict) -> torch.Tensor:
    dtype = entry["dtype"]
    start, size = entry["offset"], entry["nbytes"]
    raw = buffer[start : start + size]
    if dtype == "bfloat16":
        array = np.frombuffer(raw, dtype=np.uint16)
        return torch.from_numpy(array.copy()).view(torch.bfloat16).reshape(entry["shape"])
    array = np.frombuffer(raw, dtype=np.dtype(dtype))
    return torch.from_numpy(array.copy()).reshape(entry["shape"])


def write_packed(path, tensors: dict, meta: dict) -> Path:
    """Write tensors plus a metadata blob. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entries, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        payload = _to_bytes(tensor)
        padding = _align(offset) - offset
        if padding:
            blobs.append(b"\0" * padding)
            offset += padding
        entries[name] = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "offset": offset,
            "nbytes": len(payload),
        }
        blobs.append(payload)
        offset += len(payload)

    header = json.dumps({"version": VERSION, "meta": meta, "tensors": entries}).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        for blob in blobs:
            handle.write(blob)
    return path


def read_header(path) -> dict:
    """Header only -- enough to inspect a file without paging in the weights."""
    with open(path, "rb") as handle:
        if handle.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path} is not a llmquant weight file")
        (length,) = struct.unpack("<Q", handle.read(HEADER_LENGTH_BYTES))
        header = json.loads(handle.read(length).decode("utf-8"))
    if header["version"] != VERSION:
        raise ValueError(f"unsupported file version {header['version']}, expected {VERSION}")
    return header


def read_packed(path):
    """(tensors, meta), memory-mapped rather than read, so nothing is copied twice."""
    header = read_header(path)
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(len(MAGIC) + HEADER_LENGTH_BYTES)[len(MAGIC):])
        data_start = len(MAGIC) + HEADER_LENGTH_BYTES + length

    mapped = np.memmap(path, dtype=np.uint8, mode="r", offset=data_start)
    buffer = memoryview(mapped)
    tensors = {name: _from_buffer(buffer, entry) for name, entry in header["tensors"].items()}
    return tensors, header["meta"]
