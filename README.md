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

## The five stages, end to end

Every command below was run to produce the numbers shown, on Llama-3.2-1B-Instruct with
`configs/recommended.yaml` (attn/mlp/lm_head int8, bf16 activations and KV cache) and a
30-example task limit to keep it quick. The point of the order is that each stage answers
a question the next one depends on.

### 1. Fake quant — what does this cost in accuracy?

```bash
python examples/auto_llm.py --cfg configs/recommended.yaml --mode fake
```

```
mean_task_acc      0.4000
ppl               13.1899
bits_per_element    9.330
size_gb             1.832
```

Weights are quantized and immediately dequantized back to bf16, so the arithmetic is the
quantized arithmetic while the model still runs as a normal bf16 model. That makes this the
cheapest way to ask "how much accuracy does this dtype combination cost" — and the only
thing it is good for. **Its speed numbers are meaningless**: every combination stores the
same bf16 bytes, so decode is identical across the grid and int8 activations measure
*slower* for the quant they add. `configs/sweep.yaml` runs a grid of combinations this way;
`examples/study.py` drives that and picks a winner.

### 2. Real quant — is the quantized arithmetic right?

```bash
python examples/auto_llm.py --cfg configs/recommended.yaml --mode real
```

```
mean_task_acc      0.4111
```

Integer weights, dequantized per group and accumulated in fp32 with TF32 off. This is the
**reference**, not a deployment path — it is slower than bf16 on purpose, because it is
built to be obviously correct rather than fast. Its job is to be the thing the CUDA kernel
is checked against, bit for bit.

It does not match stage 1 exactly and is not supposed to: fake quant dequantizes and runs a
bf16 matmul, real quant keeps integers and multiplies in fp32. Two different computations
of the same thing, agreeing to about 0.006% on perplexity.

### 3. Pack — write the integers to a file

```bash
python examples/auto_llm.py --cfg configs/recommended.yaml --mode kernel --pack
# -> experiments/<project>/model.bin
```

`--pack` requires `--mode kernel`, because the packed layout is the one the kernel reads:
int4 packed two to a byte, padding and head splits already applied, one fp32 scale per
group of 128. The file carries the model id and the dtype of every layer, so it can be
loaded without the config that produced it.

### 4. Load the packed file — does it still say the same thing?

```bash
python examples/auto_llm.py --load-packed experiments/<project>/model.bin
```

```
name               attn-int8_mlp-int8_head-int8_act-bf16_g128
mean_task_acc      0.4000
```

The bf16 model is never built: the architecture comes from the model id recorded in the
file and the weights straight from the packed integers. The dtypes in the row are read
back out of the file, not from any config.

**This is the verification.** The accuracy here is identical to stage 3's, which is what
"packing is lossless" means in practice. The stricter checks live in the test suite: the
packed integers and scales compare `torch.equal` against what was packed, and a packed
model's logits are bit-identical to the same model before packing.

The kernel against the stage-2 reference is bit-identical at decode shapes for the layers
themselves, but not guaranteed for a whole model. The GEMV scales each lane's partial sum
and reduces once at the end rather than reducing per group -- the same sum in exact
arithmetic, and up to an ulp apart in floating point. On a small Mixtral that shows up as
about 4e-09 on the logits once `lm_head` is quantized, while attention and the experts stay
exactly equal. The reordering is deliberate: it is what would let an int-by-int kernel be
exact, since integer sums do not depend on summation order.

### 5. Chat — multi-turn, on the packed weights

```bash
python examples/phase5_chat.py --load-packed experiments/<project>/model.bin
```

```
you> What is the capital of France?
bot> The capital of France is Paris.
you> And of Japan?
bot> The capital of Japan is Tokyo.
```

The second answer needs the first turn to be understood, and it is: the session keeps the
conversation and its KV cache across turns rather than re-encoding the history each time.
Re-encoding would be both slower and, with a quantized cache, quietly different — entries
quantized once and kept accumulate no further error, while re-encoding requantizes
everything from scratch every turn. `/reset` clears both, `/exit` leaves.

A carried conversation has to be bounded, because nothing else bounds it: the cache grows
one entry per token forever, 32 KB of it per token on this model, and the run would end at
the model's position limit (131072 here) or at an out-of-memory, whichever came first and
neither of them legible. So the oldest turns are dropped to stay inside
`max_context_tokens`, which defaults to whatever the model says it can attend over. A trim
resets the cache — its entries are positional and the prefix they assumed is gone — so that
one turn pays a full re-encode. A single turn that cannot fit at all raises instead of
silently truncating.

The same thing from Python:

```python
from llmquant.s5_chat import ChatSession, load_for_chat

model, tokenizer, cache_factory = load_for_chat(packed_path="experiments/<project>/model.bin")
chat = ChatSession(model=model, tokenizer=tokenizer, cache_factory=cache_factory)
chat.ask("What is the capital of France?")
chat.ask("And of Japan?")      # sees the first turn
```

## Which models does this work on?

Not every HuggingFace model. The recipe finds layers by name, and those names are Llama's:

```
self_attn.{q,k,v,o}_proj      mlp.{gate,up,down}_proj      lm_head
```

plus the stacked expert weights a Mixture-of-Experts uses instead of an MLP:

```
experts.{gate_up_proj,down_proj}      (and w1/w2/w3 on older transformers)
```

That covers Llama 2/3, Mistral, Qwen2/2.5, Gemma, Phi-3 and Mixtral. It does **not** cover
architectures that name or shape things differently — GPT-2 (`attn.c_attn`, and `Conv1D`
rather than `Linear`), Falcon and GPT-NeoX (`query_key_value`), or OPT (`fc1`/`fc2` outside
any `mlp`).

A Mixture-of-Experts goes through every stage, including packing and chat. transformers
keeps Mixtral's experts as stacked `nn.Parameter`s rather than Linear modules, so there is
no Linear to swap — but there is a block to swap. `MixtralExperts.forward` already loops
over the experts a batch hit and calls `F.linear(tokens, gate_up_proj[e])` on each, and
`gate_up_proj[e]` is an ordinary `[out, in]` weight with the reduction axis last. So the
existing weight-only GEMV runs it unchanged, once per hit expert, and **no new CUDA kernel
was needed**. Fake quant does not even swap the block: it writes dequantized weights back
into the parameters and leaves the forward alone.

**The router stays bf16 deliberately.** It is one `[num_experts, hidden]` matrix whose
output is argmaxed into a discrete choice of expert, so an error there does not perturb a
value — it sends the token to a different expert. It is also far too small for quantizing
it to save anything.

Costing a MoE is different too, and the metrics know: disk pays for every expert, but a
router picks `top_k` of `num_experts`, so only that fraction is read per token. The bf16
baseline is routed the same way, since scaling top-k for the quantized run alone would
credit quantization with the router's work.

**Mixtral-8x7B itself does not fit a 16 GB card** — 21.7 GB at int4, and quantizing it needs
it in bf16 first at 87 GB. The code path is exercised end to end against a small
`MixtralForCausalLM`: real transformers routing, real expert loop, small dimensions. Running
the 8x7B checkpoint needs either a bigger GPU or layer-at-a-time quantization, which this
does not do yet.

Phi-3's fused projections work because the layout rules already describe them. `qkv_proj`
emits q, k and v from one Linear, but its output axis is still nothing but `head_dim`-sized
heads — `(num_heads + 2 * num_kv_heads)` of them — so the q/k/v boundaries land exactly on
head boundaries and splitting per head separates them for free. That makes the per-head
rule more necessary there than on a split projection, not less: without it a q head and a k
head, which have no reason to share a dynamic range, would share a scale. `gate_up_proj`
needs nothing special, since the MLP rule only pads the tail. A fused projection whose
output is *not* a multiple of `head_dim` is refused rather than split wrongly.

An unmatched layer is left in bf16, which is safe but silent — the run finishes and reports
a quantized config having quantized nothing. So stage 1 of `examples/study.py` checks and
says so before any measurement happens:

```
! NOTHING MATCHED. The recipe's patterns are Llama naming (self_attn.*_proj, mlp.*_proj,
  lm_head) and this model uses different names, so every weight stays bf16 [...]
```

or, for a partial match, which Linears were missed and how much of the model they are.
Supporting another architecture means adding its patterns in `core/modifier.py` — and, if
its attention is shaped differently, checking that `core/layout.py`'s head rules still hold.

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
