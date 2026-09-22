# Contributing

## Setup

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

334 tests. The ones that need a GPU skip themselves without CUDA, so a CPU-only checkout
still runs most of the suite. The CUDA kernels compile on first use — one to two minutes,
cached after that. `NOTES.md` has the Windows toolchain setup if the build fails, including
the one where `nvcc` exits silently because `%TMP%` contains a space.

```bash
make test       # pytest -q
make quality    # ruff
make check      # both
```

## The rule that matters here

**A performance claim needs the measurement that produced it, and a measurement needs to be
repeated before it is believed.**

That is not a style preference. Several conclusions in this repo were wrong the first time
because a single run landed inside the run-to-run spread:

- An int8 KV cache was projected at 1.47x decode from an analytic byte count and measured
  0.65x. The estimate was right about bytes and could not see that the cache is quantized in
  PyTorch on every step while the weights are read by a kernel.
- A quantized MoE was reported at 0.97x of bf16, then 1.13x, from single runs. Repeating it
  gave a median of 50.0 tok/s against bf16's 53.6 with a **26% spread** against bf16's 2.6%.
  Both earlier numbers were inside the noise.
- A synthetic benchmark of one expert block said the batched kernel was 4.7x faster while
  the whole model was still slower. The harness synchronised every iteration, which inflated
  the baseline it was being compared against.
- CUDA-event profiling showed every decode component faster than bf16 while end-to-end
  decode was not, because events measure GPU time and the cost was a host stall.

So: report a median over repeats with the spread, say which harness produced it, and if a
component measurement disagrees with the end-to-end one, the end-to-end one wins until the
disagreement is explained.

## Changing the numerics

The CUDA kernels are checked against the PyTorch reference for **equality**, not tolerance,
and the rules that make that possible are easy to break by accident:

- scales in fp32, and **true IEEE division** — `tensor / 7` lowers to a reciprocal multiply
  in PyTorch and lands an ulp off, which a kernel cannot reproduce. Divide by a tensor.
- `torch.round` ↔ CUDA `rintf`, both half-to-even; clamp after rounding; no `--use_fast_math`
- the GEMV scales each lane's partial sum and reduces once at the end. That ordering is
  deliberate — it is what would let an int-by-int kernel be exact — so keep it if you touch
  the kernel, and note that it makes whole-model equality shape-dependent.

`docs/internals.md` has the rest, including where bit-exactness stops.

## Layout

Padding and grouping are defined once, in `core/layout.py`, and every stage reads from
there. Fake quant, real quant, packing and the kernel have to agree exactly or the
bit-exactness checks between them mean nothing — a packed file laid out one way and a
kernel expecting another does not fail loudly, it is just wrong.

## Adding an architecture

The recipe finds layers by name, in `core/modifier.py`. An unmatched layer stays bf16,
which is safe and silent, so `eval/inspect.py` measures pattern coverage and says so before
any measurement happens. If you add patterns, check that `core/layout.py`'s head rules still
describe the attention you added, and add the architecture to the table in
`docs/models.md`.

## Commits

Say what changed and why it was worth changing. If a measurement motivated it, put the
numbers in the message — several of the commits here are the only place a given result is
written down.
