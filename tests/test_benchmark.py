import pytest
import torch

from llmquant.eval.benchmark import (
    INT_MM_MIN_M,
    LLAMA_1B_SHAPES,
    benchmark_shape,
    format_gemm_report,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

KEYS = {"m", "k", "n", "group_size", "bf16_ms", "int8_ms", "int8_g128_ms"}


def test_reports_every_timing(_iters=3):
    row = benchmark_shape(64, 256, 128, iters=_iters)
    assert KEYS <= set(row)
    assert row["bf16_ms"] > 0 and row["int8_ms"] > 0 and row["int8_g128_ms"] > 0


def test_small_m_is_reported_as_unsupported_not_silently_skipped():
    row = benchmark_shape(INT_MM_MIN_M - 1, 256, 128, iters=3)
    assert row["bf16_ms"] > 0
    assert row["int8_ms"] is None and row["int8_g128_ms"] is None
    assert "_int_mm" in row["note"]  # decode-sized M cannot use the int8 path at all


def test_every_llama_shape_is_group_aligned():
    # a K that is not a multiple of the group size would silently drop a tail group
    assert all(k % 128 == 0 for _, _, k in LLAMA_1B_SHAPES)


def test_format_handles_rows_with_and_without_int8():
    rows = []
    for m in (INT_MM_MIN_M - 1, 64):
        row = benchmark_shape(m, 256, 128, iters=3)
        row.update(shape="tiny", bf16_tflops=1.0, int8_speedup=None, int8_g128_speedup=None)
        if row["int8_ms"]:
            row.update(int8_speedup=1.0, int8_g128_speedup=1.0)
        rows.append(row)
    text = format_gemm_report(rows)
    assert "tiny" in text and "_int_mm" in text
