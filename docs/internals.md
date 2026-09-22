# Internals

The layout rules every stage agrees on, what bit-exactness means here and where it stops, and the things that were built and then rejected.

## Layout

Padding and grouping are defined once, in `core/layout.py`, and every stage reads from there
— fake quant, real quant, packing and the kernel have to agree exactly or the bit-exactness
checks between them mean nothing.

| module | reduction (in) axis | output axis |
|---|---|---|
| q/k/v_proj | pad the tail to a group | split per head, pad each head to a group |
| o_proj | split per head, pad each head | pad the tail to a group |
| mlp, lm_head | pad the tail to a group | pad the tail to a group |

q/k/v carry heads on the output axis because they produce them; o_proj carries them on the
reduction axis because it consumes them. Splitting per head keeps two heads with different
dynamic ranges out of one group.

Padding is zeros, which is numerically free for a symmetric abs-max scale — a zero cannot
move `max(|x|)`. It is not free in storage: at head_dim 64 and group 128, q/k/v stores two
slots for every one that matters, and int8 attention ends up weighing what bf16 weighs.

## Bit-exactness

The CUDA kernels are checked against the PyTorch reference for exact equality, not
tolerance. The rules that make that possible:

- scales in fp32, and **true IEEE division** — `tensor / 7` lowers to a reciprocal multiply
  in PyTorch and lands 1 ulp off, which a kernel cannot reproduce. Divide by a tensor.
- `torch.round` ↔ CUDA `rintf`, both half-to-even
- clamp after rounding, cast round-to-nearest-even, no `--use_fast_math`

fp32 accumulation over integers is exact while partial sums stay under 2^24; a group of 128
int8 products peaks at 2.06M, so it holds.

## What was tried and rejected

`prototypes/fused_prefill_gemm/` holds a working, bit-exact fused int8-weight GEMM that is
not shipped, because it runs at about a third of cuBLAS.

Decode already fuses: the GEMV reads int weights and dequantizes in registers, so no
dequantized copy is ever written. Prefill does not — it materializes the whole weight in
bf16 every forward, and that dequant costs 27.6 ms against 23.3 ms for the GEMM it feeds,
which is the entire TTFT regression. Fusing is the only way to remove it, and a hand-written
tensor-core GEMM landed at 0.34x of cuBLAS where 0.43x was break-even. The prototype's
README has the numbers and the list of optimizations it leaves out; the next attempt, if
there is one, should use CUTLASS rather than more hand-tuning.

## Layout of the repo

```
llmquant/
  core/         config, schemes, scale computation, quant-dequant math, layout, cost metrics
  quantizers/   the three implementations and the lookup that picks one:
                  fake.py    dequantize and run bf16 -- what a dtype costs in accuracy
                  real.py    keep the integers, multiply in fp32 -- the reference
                  kernel.py  hand them to CUDA -- the path that gets deployed
  cuda/         building and calling the kernels, and csrc/
  packing/      the single-file format: pack, write, read, load
  runtime/      running one: graphed decode, chat, and bf16 comparison
  eval/         measuring, analysing, reporting
examples/       run / sweep / analyze / chat / bench_gemm
configs/        recommended / sweep / smoke; CLI flags override
prototypes/     tried and rejected, kept with its numbers
tests/          318 tests
NOTES.md        the working notes, in Korean, far more detailed than this
```

`core` reaches into a quantizer in exactly one place: `modifier.apply()` asks
`quantizers.dispatch.quant_linear_for()` for the class matching `mode`, via a deferred
import so the dependency does not become a module-level cycle.
