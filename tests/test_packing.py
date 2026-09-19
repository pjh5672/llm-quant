"""Phase 3: the single-file format, and the round trip that has to be exact.

A packed file that is nearly right is the worst failure mode in this project -- nothing
raises, the model just answers slightly differently, and it looks like a quantization
effect. So the assertions here are equalities.
"""

import json
import struct

import pytest
import torch
import torch.nn as nn

from llmquant.core.scheme import GROUP_SIZE, preset_name_to_scheme
from llmquant.stages.s3_pack import (
    describe_packed,
    pack_int4,
    pack_weight,
    read_header,
    read_packed,
    unpack_int4,
    unpack_weight,
    write_packed,
)
from llmquant.stages.s3_pack.format import ALIGNMENT, MAGIC
from llmquant.stages.s3_pack.model_io import save_packed_model
from llmquant.stages.s4_kernel import KernelQuantLinear


def test_the_documented_nibble_example():
    """-3, 5 -> 5, 13 -> 0xD5, straight out of the format note."""
    packed = pack_int4(torch.tensor([[-3, 5]], dtype=torch.int8))
    assert int(packed[0, 0]) == 0xD5


def test_every_int4_value_survives_the_round_trip():
    values = torch.arange(-8, 8, dtype=torch.int8).repeat(4, 1)
    assert torch.equal(unpack_int4(pack_int4(values)), values)


def test_int4_halves_the_bytes_and_int8_is_left_alone():
    values = torch.arange(-8, 8, dtype=torch.int8).repeat(4, 1)
    assert pack_weight(values, 4).numel() * 2 == values.numel()
    assert torch.equal(unpack_weight(pack_weight(values, 8), 8), values)


@pytest.mark.parametrize("bad", [9, -9])
def test_values_outside_the_int4_range_are_refused(bad):
    with pytest.raises(ValueError, match=r"must be in \[-8, 7\]"):
        pack_int4(torch.tensor([[bad, 0]], dtype=torch.int8))


def test_an_odd_last_axis_is_refused():
    with pytest.raises(ValueError, match="must be even"):
        pack_int4(torch.zeros(1, 3, dtype=torch.int8))


@pytest.mark.parametrize("num_bits", [2, 16])
def test_unsupported_widths_are_refused(num_bits):
    with pytest.raises(ValueError, match="unsupported num_bits"):
        pack_weight(torch.zeros(2, 2, dtype=torch.int8), num_bits)


def test_tensors_round_trip_through_the_file(tmp_path):
    tensors = {
        "packed": torch.randint(0, 255, (4, 8), dtype=torch.uint8),
        "scale": torch.rand(4, 2),
        "norm": torch.randn(16).to(torch.bfloat16),
    }
    path = write_packed(tmp_path / "w.bin", tensors, {"model_id": "test"})
    restored, meta = read_packed(path)
    assert meta["model_id"] == "test"
    for name, tensor in tensors.items():
        assert torch.equal(restored[name], tensor), name


def test_every_tensor_starts_aligned(tmp_path):
    """So the file can be mapped and handed to the GPU without a realigning copy."""
    path = write_packed(
        tmp_path / "w.bin",
        {"a": torch.zeros(3, dtype=torch.uint8), "b": torch.zeros(5, dtype=torch.uint8)},
        {"model_id": "test"},
    )
    for entry in read_header(path)["tensors"].values():
        assert entry["offset"] % ALIGNMENT == 0


def test_a_foreign_file_is_rejected(tmp_path):
    path = tmp_path / "not-ours.bin"
    path.write_bytes(b"SOMETHING" + b"\0" * 64)
    with pytest.raises(ValueError, match="not a llmquant weight file"):
        read_header(path)


def test_a_future_version_is_rejected(tmp_path):
    header = json.dumps({"version": 999, "meta": {}, "tensors": {}}).encode()
    path = tmp_path / "future.bin"
    path.write_bytes(MAGIC + struct.pack("<Q", len(header)) + header)
    with pytest.raises(ValueError, match="unsupported file version"):
        read_header(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("name", ["W4A16", "W8A16"])
def test_a_quantized_layer_round_trips_bit_exactly(tmp_path, name):
    torch.manual_seed(0)
    linear = nn.Linear(4 * GROUP_SIZE, GROUP_SIZE, bias=False).cuda().to(torch.bfloat16)
    module = KernelQuantLinear.from_linear(linear, preset_name_to_scheme(name))

    model = nn.Module()
    model.proj = module
    path = save_packed_model(model, tmp_path / "layer.bin", "test/model")

    tensors, meta = read_packed(path)
    bits = meta["layers"]["proj"]["weights"]["num_bits"]
    restored = unpack_weight(tensors["proj.qweight"], bits).reshape(
        meta["layers"]["proj"]["canonical_shape"]
    )
    assert torch.equal(restored.cuda(), module.qweight)
    assert torch.equal(tensors["proj.wscale"].cuda(), module.wscale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_derived_buffers_are_not_written(tmp_path):
    """A quantized layer carries buffers derived from its scales. Writing them out as bf16
    meant the loader overwrote a value its own constructor had just computed exactly, and
    the weights still compared bit-exact while the logits moved."""
    torch.manual_seed(0)
    linear = nn.Linear(GROUP_SIZE, GROUP_SIZE, bias=False).cuda().to(torch.bfloat16)
    model = nn.Module()
    model.proj = KernelQuantLinear.from_linear(linear, preset_name_to_scheme("W4A16"))

    path = save_packed_model(model, tmp_path / "layer.bin", "test/model")
    stored = set(read_header(path)["tensors"])
    assert stored == {"proj.qweight", "proj.wscale"}
    assert not any("_broadcast_scale" in name for name in stored)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_describe_reads_only_the_header(tmp_path):
    torch.manual_seed(0)
    linear = nn.Linear(GROUP_SIZE, GROUP_SIZE, bias=False).cuda().to(torch.bfloat16)
    model = nn.Module()
    model.proj = KernelQuantLinear.from_linear(linear, preset_name_to_scheme("W4A16"))
    path = save_packed_model(model, tmp_path / "layer.bin", "test/model")

    summary = describe_packed(path)
    assert summary["model_id"] == "test/model"
    assert summary["quantized_layers"] == 1
    assert summary["bytes"] > 0
