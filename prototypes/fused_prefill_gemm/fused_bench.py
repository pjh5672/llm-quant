"""Does a hand-written fused GEMM get close enough to cuBLAS to be worth it?

Correctness first, then three timings per shape:
  cuBLAS bf16        what prefill runs today, after paying for the dequant separately
  proto bf16         the same GEMM hand-written -- how good is this kernel at all?
  proto fused int8   the same kernel reading int8 + fp32 group scale in the smem loader
"""

import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from llmquant.s4_kernel.build import (  # noqa: E402
    CUDA_FLAGS,
    CXX_FLAGS,
    ensure_extensions_dir,
    ensure_msvc_env,
    ensure_space_free_tmp,
)

HERE = pathlib.Path(__file__).resolve().parent
GROUP = 128


def build():
    from torch.utils.cpp_extension import load

    ensure_msvc_env()
    ensure_space_free_tmp()
    ensure_extensions_dir()
    import os

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    return load(name="fused_proto",
                sources=[str(HERE / "fused_proto.cpp"), str(HERE / "fused_proto.cu")],
                extra_cflags=CXX_FLAGS, extra_cuda_cflags=CUDA_FLAGS, verbose=False)


def timed(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ext = build()
    torch.manual_seed(0)
    M = 512
    shapes = [("gate/up", 8192, 2048), ("down", 2048, 8192), ("q/o", 2048, 2048)]

    print(f"{'shape':<9} {'N':>5} {'K':>5} {'cuBLAS':>9} {'proto16':>9} {'vs':>6} "
          f"{'fused8':>9} {'vs':>6} {'maxerr':>9}")
    for name, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        qb = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8)
        scale = (torch.rand(N, K // GROUP, device="cuda", dtype=torch.float32) * 0.01 + 1e-3)
        # the bf16 weight the unfused path would have built, so the two are comparable
        b16 = (qb.reshape(N, K // GROUP, GROUP) * scale.unsqueeze(-1)).to(torch.bfloat16).reshape(N, K)

        ref = (a @ b16.T).float()
        got = ext.run(a, b16, qb, scale, True).float()
        err = (got - ref).abs().max().item()
        scaleref = ref.abs().max().item()

        ms_cublas = timed(lambda: a @ b16.T)
        ms_proto = timed(lambda: ext.run(a, b16, qb, scale, False))
        ms_fused = timed(lambda: ext.run(a, b16, qb, scale, True))
        print(f"{name:<9} {N:>5} {K:>5} {ms_cublas:>8.3f}m {ms_proto:>8.3f}m "
              f"{ms_cublas/ms_proto:>5.2f}x {ms_fused:>8.3f}m {ms_cublas/ms_fused:>5.2f}x "
              f"{err/scaleref:>9.2e}")

    print("\n'vs' is cuBLAS/proto: 1.00x means matching cuBLAS, 0.50x means half its speed.")
    print("maxerr is relative to the largest output element; bf16 has ~8 mantissa bits.")


if __name__ == "__main__":
    main()
