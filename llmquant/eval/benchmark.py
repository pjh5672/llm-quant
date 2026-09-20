"""Op-level GEMM benchmark: what can an int8 activation actually buy?

This deliberately does NOT time the fake-quant model. In `mode="fake"` the weights are
stored already dequantized, so the forward is a plain bf16 GEMM plus, for A8, a per-forward
activation quant -- A8 measures *slower* than A16 there, which is the opposite of the
deployed answer. Timing the GEMM primitives instead gives a number that is not polluted by
the fake-quant path and needs no Phase 4 kernel.

Read the results as:
  bf16          the baseline every Linear runs today
  int8 full-K   one int32 accumulation over the whole K -- the ceiling for A8
  int8 g128     accumulation broken every 128 elements, as the confirmed design requires.
                Composed from torch ops here, so it pays a kernel launch per group; a fused
                kernel would not. Treat it as a floor, not as the design's real speed.
"""

import time

import torch

# (label, N, K) for Llama-3.2-1B; every Linear in the model has one of these shapes
LLAMA_1B_SHAPES = (
    ("q/o_proj", 2048, 2048),
    ("k/v_proj", 512, 2048),
    ("gate/up_proj", 8192, 2048),
    ("down_proj", 2048, 8192),
)
INT_MM_MIN_M = 17  # torch._int_mm: "self.size(0) needs to be greater than 16"


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def _tflops(m, k, n, ms):
    return 2 * m * k * n / (ms * 1e-3) / 1e12


def benchmark_shape(m: int, k: int, n: int, group_size: int = 128, iters: int = 50) -> dict:
    dev = "cuda"
    a16 = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    b16 = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
    a8 = torch.randint(-127, 127, (m, k), device=dev, dtype=torch.int8)
    b8 = torch.randint(-127, 127, (k, n), device=dev, dtype=torch.int8)

    row = {"m": m, "k": k, "n": n, "group_size": group_size}
    row["bf16_ms"] = _time(lambda: torch.mm(a16, b16), iters)

    if m < INT_MM_MIN_M:
        row["int8_ms"] = None
        row["int8_g128_ms"] = None
        row["note"] = f"torch._int_mm needs M > {INT_MM_MIN_M - 1}"
        return row

    row["int8_ms"] = _time(lambda: torch._int_mm(a8, b8), iters)

    # pre-slice outside the timed region: the copies are an artifact of composing this from
    # torch ops, not something the design pays
    groups = [
        (a8[:, i : i + group_size].contiguous(), b8[i : i + group_size].contiguous())
        for i in range(0, k, group_size)
    ]
    sx = torch.rand(m, len(groups), device=dev)
    sw = torch.rand(len(groups), n, device=dev)

    def grouped():
        acc = torch.zeros(m, n, device=dev, dtype=torch.float32)
        for g, (ag, bg) in enumerate(groups):
            acc += torch._int_mm(ag, bg).float() * sx[:, g : g + 1] * sw[g]
        return acc

    row["int8_g128_ms"] = _time(grouped, max(iters // 5, 5))
    return row


def benchmark_gemms(m_values=(64, 512, 2048), shapes=LLAMA_1B_SHAPES, group_size=128, iters=50):
    rows = []
    for label, n, k in shapes:
        for m in m_values:
            row = benchmark_shape(m, k, n, group_size, iters)
            row["shape"] = label
            row["bf16_tflops"] = _tflops(m, k, n, row["bf16_ms"])
            row["int8_speedup"] = row["bf16_ms"] / row["int8_ms"] if row["int8_ms"] else None
            row["int8_g128_speedup"] = (
                row["bf16_ms"] / row["int8_g128_ms"] if row["int8_g128_ms"] else None
            )
            rows.append(row)
    return rows


def format_gemm_report(rows) -> str:
    out = [
        f"{'shape':<13} {'M':>5} {'bf16 ms':>9} {'TFLOPS':>7} {'int8 ms':>9} {'x':>6} "
        f"{'g128 ms':>9} {'x':>6}",
        "-" * 72,
    ]
    for r in rows:
        if r["int8_ms"] is None:
            out.append(
                f"{r['shape']:<13} {r['m']:>5} {r['bf16_ms']:>9.3f} {r['bf16_tflops']:>7.1f} "
                f"{'-':>9} {'-':>6} {'-':>9} {'-':>6}   ({r['note']})"
            )
            continue
        out.append(
            f"{r['shape']:<13} {r['m']:>5} {r['bf16_ms']:>9.3f} {r['bf16_tflops']:>7.1f} "
            f"{r['int8_ms']:>9.3f} {r['int8_speedup']:>5.2f}x "
            f"{r['int8_g128_ms']:>9.3f} {r['int8_g128_speedup']:>5.2f}x"
        )
    return "\n".join(out)
