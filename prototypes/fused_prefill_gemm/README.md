# Fused prefill GEMM — prototype, not shipped

**Result: rejected.** A hand-written tensor-core GEMM reaches about a third of cuBLAS here,
which is not enough to pay for what fusing saves. Kept because the finding is worth keeping
and because picking the work back up should not start from nothing.

Nothing in `src/llmquant/` imports this. It builds and runs on its own:

```bash
python prototypes/fused_prefill_gemm/fused_bench.py
```

## The problem it was built to solve

In `mode=kernel`, decode (M ≤ 4) runs the fused GEMV in `s4_kernel/csrc/wq_gemv.cu`: the
int weight is read as integers and dequantized in registers, so no dequantized copy is ever
written. Prefill (M > 4) does the opposite — `KernelQuantLinear._dequantized_weight()`
materializes the whole weight in bf16 and hands it to cuBLAS, every forward.

Measured at M=512, summed over every Linear in Llama-3.2-1B:

```
bf16 GEMMs only    23.3 ms   <- what the bf16 baseline pays
+ dequant          50.9 ms   <- what mode=kernel pays (dequant is 27.6 ms)
```

The dequant costs more than the GEMM it feeds. It is the whole TTFT regression: 48.6 ms for
bf16 against 77.8 ms for the kernel path.

A blocked dequant, keeping the fp32 intermediate small enough to stay in cache, was tried
first and did nothing: 27.6 → 28.0 ms. The cost is the bf16 output write and the int8 read,
not the fp32 temporary, so only fusing removes it.

## What this measures

Two kernels from one template, differing only in how the B tile is filled — `gemm_bf16`
reads a ready-made bf16 weight, `gemm_fused` reads int8 plus the fp32 group scale and
dequantizes into shared memory. The first says how good the hand-written GEMM is at all;
the second says what fusing costs on top.

```
shape        N     K    cuBLAS   proto16    vs    fused8    vs    maxerr
gate/up   8192  2048    0.396m   1.042m   0.38x   1.155m  0.34x  0.00e+00
down      2048  8192    0.388m   1.225m   0.32x   1.710m  0.23x  0.00e+00
q/o       2048  2048    0.109m   0.248m   0.44x   0.444m  0.25x  0.00e+00
```

`maxerr 0.00e+00` — the fused result is bit-identical to what cuBLAS produces from the
dequantized weight, so the arithmetic is right. Only the speed is wrong.

## Why that is a rejection

```
current (cuBLAS + separate dequant)   0.396 + 0.516 = 0.912 ms   (gate/up)
this prototype, fused                                 1.155 ms   <- 27% slower
bf16 baseline, the actual target                      0.396 ms
```

Beating the current path needs only 0.43x of cuBLAS. The prototype lands at 0.34x — close,
but short. Getting TTFT back to 48 ms needs roughly 1.0x, which is cuBLAS-class.

## What it deliberately leaves out

All standard, all already in CUTLASS:

- **no `cp.async` double buffering** — global loads and math serialize. The largest single
  loss, and why `down` (K=8192, the longest K loop) is worst at 0.23x.
- **64×64×32 tiles** — cuBLAS uses 128×128 and larger; arithmetic intensity and register
  reuse are both low here.
- **4 warps** — too little occupancy to hide latency.
- **padded shared memory instead of swizzling** — minor.

Implementing these by hand would plausibly reach 0.7–0.9x. That is days of work with no
guarantee of cuBLAS parity, for a win confined to prefill: TTFT 77.8 → ~48 ms, 1.6x on the
second priority, with decode unchanged. So the next step, if there is one, is CUTLASS's
mixed-input GEMM rather than more hand-tuning — with the caveat that it has to be matched
to `core/layout.py`'s canonical form (per-head padding, one fp32 scale per 128 along K).
