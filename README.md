# llm-quant

Symmetric round-to-nearest quantization for Llama-class models, with custom CUDA kernels and
a measurement pipeline that checks its own predictions.

Weights go to int4 or int8 in groups of 128 along the reduction axis; activations and the KV
cache can be quantized independently. There is no calibration step — RTN only — so a
configuration is a few dtype choices and nothing has to be fit.

The point of the repo is not the quantizer, which is standard. It is the path from
"quantize it" to "and it is actually faster", which is where the interesting failures are.

## Install

```bash
python -m pip install -e ".[dev]"   # ninja is required: the CUDA kernels JIT-compile
```

Python ≥ 3.10, PyTorch with CUDA. The kernels compile on first use (1–2 minutes, then
cached). Built and measured on an RTX 5060 Ti (sm_120) under Windows; see
[NOTES.md](NOTES.md) for the MSVC/nvcc setup, including the
one where `nvcc` dies silently because `%TMP%` contains a space.

## Quick start

```bash
# the whole study: structure check -> sweep -> selection -> real-path verification -> report
python examples/study.py --cfg configs/sweep.yaml

# same, on a different model
python examples/study.py --cfg configs/sweep.yaml --model Qwen/Qwen2.5-1.5B-Instruct

# two-minute plumbing check
python examples/study.py --cfg configs/smoke.yaml

# one configuration, no sweep
python examples/auto_llm.py --cfg configs/recommended.yaml

# chat with a packed model
python examples/phase5_chat.py --load-packed experiments/<project>/model.bin
```

`study.py` writes `reports/<project>.md` (committable, meant to be diffed against the next
model) and `experiments/<project>/sweep.json` (re-analysable without a GPU via
`examples/phase1_analyze.py`).

## Configuration

Five independent dtype axes, `int4` | `int8` | `bf16`, plus a uniform group size of 128:

```yaml
quantization:
  attn_weight: int8     # self_attn.{q,k,v,o}_proj
  mlp_weight:  int8     # mlp.{gate,up,down}_proj
  head_weight: int8     # lm_head
  activation:  bf16     # input of every quantized Linear; int8 | bf16 only
  kv_cache:    bf16     # applied at generation time, not by rewiring; int8 | bf16 only
  group_size:  128
  mode: fake            # fake | real | kernel
```

`mode` selects which stage runs the linear layers:

| mode | what it does | use it for |
|---|---|---|
| `fake` | quantize-dequantize, weights stay bf16 | accuracy. Speed here is the simulation's, not a deployment's |
| `real` | integer weights, dequantized per group in fp32 | the exactness oracle |
| `kernel` | integer weights read directly by a CUDA kernel | speed, VRAM, and `--pack` |

A sweep expands a grid:

```yaml
sweep:
  attn_weight: [int4, int8]
  mlp_weight:  [int4, int8]
  kv_cache:    [bf16, int8]
```

## What the pipeline does

```
1 inspect    structural facts that decide what the results mean   eval/inspect.py
2 sweep      every combination, on the fake path                  eval/run.py
3 select     accuracy limit, weighted ranking, axis effects       eval/analysis.py
4 verify     the winner re-measured on the real path + ablations  eval/verify.py
5 reconcile  projection vs measurement, with attribution          eval/verify.py
6 report     terminal + reports/<project>.md                      eval/markdown.py
```

Stages 4 and 5 exist because stages 2 and 3 cannot tell the truth about speed. A fake-quant
sweep stores dequantized bf16 weights, so every combination moves the same bytes and the
ranking has to fall back on an analytic estimate of decode traffic. That estimate is a ratio
of bytes and it is correct about bytes — but it cannot price work that happens outside the
kernel, and it was wrong by more than 2x the first time it mattered. So the selected
configuration is re-measured for real, and every axis handled outside the kernel gets an
ablation run that isolates its true cost.

Selection is a hard accuracy limit plus a weighted score over what is left. The default
priority is decode throughput, then prefill latency, then accuracy — set it in the config.

## Results (Llama-3.2-1B-Instruct, group 128)

Accuracy is mean score over ARC-Easy, ARC-Challenge and OpenBookQA — generation tasks, not
perplexity, for reasons below. Speed is measured on `mode=kernel`.

| configuration | accuracy | Δacc | decode | TTFT | peak VRAM | BPV |
|---|---|---|---|---|---|---|
| bf16 | 0.4867 | — | 77.3 tok/s | 48.6 ms | 2.42 GB | 16.00 |
| W8 + int8 lm_head | 0.4700 | −3.43% | **80.6 tok/s** | 79.8 ms | 1.91 GB | 9.33 |
| W8, bf16 lm_head | 0.4722 | −2.97% | 74.3 tok/s | 80.8 ms | 1.65 GB | 10.98 |

3 of 25 combinations came within a 5% accuracy limit, and all three were int8 weights.

Where the accuracy goes, one axis at a time:

```
attn_weight: int8 -> int4    +30.08%    dBPV -1.09
kv_cache:    bf16 -> int4    +23.23%    dBPV  0.00
mlp_weight:  int8 -> int4    +13.38%    dBPV -2.61
kv_cache:    bf16 -> int8     +1.28%    dBPV  0.00
activation:  bf16 -> int8     +0.97%    dBPV  0.00
```

### Five things that were not obvious

**Perplexity is wrong in both directions.** A PPL pass never reads the KV cache back, so it
scores an int4 cache at +0.18% while generation accuracy falls 18%. In the other direction
it overstates int4 weight damage by 33 points. Accuracy here is measured on generations.

**int4 belongs in the MLP, never in attention.** Attention is 2.2x more int4-sensitive and
saves 2.4x less, so `attn=int8, mlp=int4` beats `attn=int4, mlp=int8` on accuracy *and*
size. An earlier PPL-based reading of the same question said the opposite.

**An int8 KV cache is not free, and the analytic model cannot see why.** It was projected at
1.47x decode and measured 0.65x: on a 1B model at 2k context the weights are 1.58 GB per
token against a 0.034 GB cache, so quantizing the cache saves 1.8% of the traffic — and
costs 32% of decode time, because it runs in PyTorch on every step while the weights are
read by a kernel. This is what stage 4 exists to catch.

**An int8 activation buys nothing here, and cannot.** No mode in this repo runs an
int8xint8 matmul: `fake` quantize-dequantizes and then multiplies in bf16, `real` quantizes
to integers and multiplies in fp32 for exactness, and `kernel` is weight-only -- it reads
the activation scheme only to print a label. Building an int GEMM would not change the
answer either: `torch._int_mm` measured 1.03-1.04x against bf16 on this GPU and requires
M>16, so decode cannot call it at all. And decode would not benefit in any case, because at
M=1 the matmul is a GEMV that streams the whole weight matrix to do very little arithmetic
-- the activation is a few KB against 1.6 GB of weights per token. Shrinking the small
operand does nothing when the large one is the bottleneck, which is why the activation axis
moves BPV and decode traffic by exactly 0.00.

**Quantizing the transformer weights does not speed up decode at this size.** Only ~45% of a
decode step is weight bandwidth on a 1B model, so halving the weight bytes cannot pay for
the dequant on the way in. What puts the stack above baseline is `head_weight=int8`, and it
reads oddly on disk: this model ties `lm_head` to the embedding, so quantizing it does not
shrink a tensor — it unties one and adds a 258 MB int8 copy beside the 501 MB bf16
embedding. Decode does not care, because it reads `lm_head` in full every token and only
row-indexes the embedding. Hence disk +252 MB and decode traffic −237 MB/token at once.
Stage 1 prints this before the sweep runs, so the next model does not repeat the confusion.

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

## Layout of the repo

```
llmquant/
  core/       config, schemes, scale computation, quant-dequant math, layout, cost metrics
  eval/       running, measuring, analysing, reporting
  modes.py    mode -> Linear class, the one place core reaches into a phase
  s1_fake/    mode="fake"    bf16 dequant, for measuring the accuracy cost
  s2_real/    mode="real"    int weights in exact fp32 math, the reference
  s3_pack/                   packing, the .bin format, the loader
  s4_kernel/  mode="kernel"  CUDA kernels, and the graphed decode loop
  s5_chat/                   conversation on a packed model
examples/     study.py (full pipeline), auto_llm.py (one run), phase1_analyze.py, ...
configs/      recommended / sweep / smoke; CLI flags override
prototypes/   tried and rejected, kept with its numbers
tests/        283 tests
NOTES.md      the working notes, in Korean, far more detailed than this
```

`core` reaches into a phase in exactly one place: `modifier.apply()` asks
`modes.quant_linear_for()` for the Linear class matching `mode`, via a deferred import so
the dependency does not become a module-level cycle.

## Tests

```bash
python -m pytest -q
```

GPU tests skip automatically without CUDA.
