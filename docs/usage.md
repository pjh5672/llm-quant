# Usage

The five stages end to end, and what each script is for. Every number below came from running the command above it.

## The five stages, end to end

Every command below was run to produce the numbers shown, on Llama-3.2-1B-Instruct with
`configs/recommended.yaml` (attn/mlp/lm_head int8, bf16 activations and KV cache) and a
30-example task limit to keep it quick. The point of the order is that each stage answers
a question the next one depends on.

### 1. Fake quant — what does this cost in accuracy?

```bash
python examples/run.py --cfg configs/recommended.yaml --mode fake
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
`examples/sweep.py` drives that and picks a winner.

### 2. Real quant — is the quantized arithmetic right?

```bash
python examples/run.py --cfg configs/recommended.yaml --mode real
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
python examples/run.py --cfg configs/recommended.yaml --mode kernel --pack
# -> experiments/<project>/model.bin
```

`--pack` requires `--mode kernel`, because the packed layout is the one the kernel reads:
int4 packed two to a byte, padding and head splits already applied, one fp32 scale per
group of 128. The file carries the model id and the dtype of every layer, so it can be
loaded without the config that produced it.

### 4. Load the packed file — does it still say the same thing?

```bash
python examples/run.py --load-packed experiments/<project>/model.bin
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
python examples/chat.py --load-packed experiments/<project>/model.bin
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
from llmquant.runtime import ChatSession, load_for_chat

model, tokenizer, cache_factory = load_for_chat(packed_path="experiments/<project>/model.bin")
chat = ChatSession(model=model, tokenizer=tokenizer, cache_factory=cache_factory)
chat.ask("What is the capital of France?")
chat.ask("And of Japan?")      # sees the first turn
```

## The scripts

Five, each doing one thing. All of them take `--cfg` plus any flag from the config, and a
CLI flag always wins over the file.

### `run.py` — one configuration, measured

```bash
python examples/run.py --cfg configs/recommended.yaml
python examples/run.py --cfg configs/recommended.yaml --mode kernel --pack
python examples/run.py --load-packed experiments/<project>/model.bin
python examples/run.py --cfg configs/recommended.yaml --mlp-weight int4   # the flag wins
```

The workhorse. Loads a model, applies one recipe, and reports accuracy, perplexity, cost
and latency for it. `--pack` writes the packed file (and needs `--mode kernel`, since the
packed layout is the one the kernel reads); `--load-packed` skips quantizing and measures a
file that already exists.

### `sweep.py` — the whole study, one command

```bash
python examples/sweep.py --cfg configs/sweep.yaml
python examples/sweep.py --cfg configs/sweep.yaml --model Qwen/Qwen2.5-1.5B-Instruct
python examples/sweep.py --cfg configs/smoke.yaml          # two-minute plumbing check
python examples/sweep.py --cfg configs/sweep.yaml --skip-verify
```

Expands the `sweep:` grid and runs every combination through the same code path `run.py`
uses, then picks a winner and **re-measures it on the real path**. That last part is not a
formality: the analytic estimate the ranking uses is a ratio of bytes and cannot price work
done outside the kernel, and it was wrong by more than 2x the first time it mattered. Writes
`reports/<project>.md` and `experiments/<project>/sweep.json`.

### `analyze.py` — re-rank a finished sweep, no GPU

```bash
python examples/analyze.py experiments/<project>/sweep.json
python examples/analyze.py experiments/<project>/sweep.json --bpv-weight 5 --no-limit
```

The weights are what turn the same measurements into different answers, so they are flags
rather than something baked into the saved file. Useful for asking "what if I cared more
about size than accuracy" without paying for the sweep again.

### `chat.py` — talk to it

```bash
python examples/chat.py --load-packed experiments/<project>/model.bin
python examples/chat.py --cfg configs/recommended.yaml --mode kernel
python examples/chat.py --load-packed model.bin --ask "What is a prime number?"
```

Multi-turn: the conversation and its KV cache are carried across turns rather than
re-encoded, and the oldest turns are dropped when the context fills. `/reset` clears both,
`/exit` leaves, `--ask` does one question and returns.

`--compare` answers with the bf16 model as well and reports where the two part:

```bash
python examples/chat.py --load-packed model.bin --compare
python examples/chat.py --load-packed big.bin --compare --reference-device cpu
```

```
you> Name three primary colours.
  bf16      > The three primary colours are: 1. Red 2. Blue 3. Yellow
  quantized > Here are three primary colors: 1. Red 2. Blue 3. Yellow ...
  38% of tokens agree, first difference at token 0

you> Why is the sky blue?
  bf16      > ... a phenomenon called Rayleigh scattering, n
  quantized > ... a phenomenon called scattering. When
  16% of tokens agree, first difference at token 4
```

That is W4 on Llama-3.2-1B; W8 answers all three identically. Both decode greedily, so
every difference is quantization error and nothing else, and the position of the first
different token matters as much as the rate: 90% agreement starting at token 2 is a
different thing from 90% starting at token 40. `/summary` gives the totals so far.

Each model keeps its own reply in its own history by default, so divergence compounds the
way it would for someone actually using it. `--follow-reference` feeds both the bf16 reply
instead, which judges each turn on its own. `--reference-device cpu` exists because the two
models have to fit at once: Llama-3.2-1B is 2.4 GB plus 1.9 GB and fits easily, OLMoE-1B-7B
is 12.9 GB in bf16 and does not fit beside anything.

### `bench_gemm.py` — the GEMM primitives on their own

```bash
python examples/bench_gemm.py --m 1 64 512 2048
```

Times bf16 against int8 at the shapes this model actually uses, without a model around
them. This is the measurement that decided A8 was not worth having: int8 GEMM came out at
1.03-1.04x of bf16 here, and decode cannot call it at all.
