# Supported models

Which architectures the recipe reaches, how it says so when it does not, and what a Mixture-of-Experts needs that a dense model does not.

## Which models does this work on?

Not every HuggingFace model. The recipe finds layers by name, and those names are Llama's:

```
self_attn.{q,k,v,o}_proj      mlp.{gate,up,down}_proj      lm_head
```

plus the stacked expert weights a Mixture-of-Experts uses instead of an MLP:

```
experts.{gate_up_proj,down_proj}      (and w1/w2/w3 on older transformers)
```

That covers Llama 2/3, Mistral, Qwen2/2.5, Gemma, Phi-3, Mixtral and OLMoE. It does **not** cover
architectures that name or shape things differently — GPT-2 (`attn.c_attn`, and `Conv1D`
rather than `Linear`), Falcon and GPT-NeoX (`query_key_value`), or OPT (`fc1`/`fc2` outside
any `mlp`).

A Mixture-of-Experts goes through every stage, including packing and chat. transformers
keeps the experts as stacked `nn.Parameter`s rather than Linear modules, so there is no
Linear to swap — but there is a block to swap. The expert block's forward loops over the
experts a batch hit and calls `F.linear(tokens, gate_up_proj[e])` on each, and
`gate_up_proj[e]` is an ordinary `[out, in]` weight with the reduction axis last, so the
existing weight-only GEMV runs it unchanged, once per hit expert. Fake quant does not even
swap the block: it writes dequantized weights back into the parameters and leaves the
forward alone.

**The block is found by structure, not by class name.** The first version matched
`MixtralExperts`, which meant OLMoE — the same module under a different name — was silently
not swapped and reported a 1.03x packed ratio. `core/modifier.py` now looks for the
attributes an expert block has and raises naming the missing ones, so the next architecture
either works or says why.

### Why a per-expert loop was not enough

Running the GEMV once per hit expert is correct and it is what shipped first, but at decode
it launches `top_k` kernels per layer for one token each — on OLMoE that is 8 experts x 16
layers x 2 projections = 256 launches per token, none of them with enough work to cover
their own overhead. Measured, it came out level with bf16 and the run-to-run spread was
16%, which is how two earlier readings of it (0.97x, then 1.13x) both landed inside the
noise and both got reported before the measurement was repeated.

Two things fixed it:

- **`cuda/csrc/wq_gemv_batched.cu`** runs every hit expert in one launch. The host emits a
  block list so no block straddles two experts, and the arithmetic is ordered exactly as
  the dense GEMV orders it, so a row is bit-identical either way.
- **The decode step captures into a CUDA graph**, which it could not do at first: the
  routing plan called `torch.bincount`, and on CUDA that reads its maximum back to the host
  to size its output — even with `minlength` — and a host round trip is the one thing
  capture cannot contain. `scatter_add_` into a pre-sized tensor does the same counting
  without leaving the GPU.

### What it is worth (OLMoE-1B-7B-Instruct, W4 MLP + W8 attention)

Medians of 5 runs at 128 prompt tokens and 48 new tokens, on an RTX 5060 Ti. The numbers
are in [`experiments/olmoe/latency.json`](../experiments/olmoe/latency.json).

| | decode | TTFT | peak VRAM |
|---|---|---|---|
| bf16 | 52.9 tok/s | 56.8 ms | 13.01 GB |
| packed, per-expert loop | 54.4 tok/s | 516.5 ms | 6.80 GB |
| packed, batched + CUDA graph | **93.0 tok/s** | 521.1 ms | 7.51 GB |

Decode is 1.76x bf16 at 0.58x the memory — the point of the exercise. Two things are worth
saying plainly about the rest of that table.

**Prefill is 9x worse, and that is a real cost, not a measurement artifact.** Above a row
limit the batched path falls back to dequantizing each expert and calling cuBLAS, because
prefill is compute-bound rather than bandwidth-bound and a weight-only GEMV has nothing to
offer it. This is the same conclusion the dense model reached when fused prefill was tried
and rejected — see [internals](internals.md).

**bf16 cannot be CUDA-graphed here at all.** transformers' grouped-MM expert forward copies
a CPU offsets tensor into the capture, and fails with exactly the error our own path used
to fail with. So part of the 1.76x is that the quantized path is capturable and the
baseline is not — which is a property of the implementation, not of the arithmetic, and
would narrow if transformers fixed it.

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
a quantized config having quantized nothing. So stage 1 of `examples/sweep.py` checks and
says so before any measurement happens:

```
! NOTHING MATCHED. The recipe's patterns are Llama naming (self_attn.*_proj, mlp.*_proj,
  lm_head) and this model uses different names, so every weight stays bf16 [...]
```

or, for a partial match, which Linears were missed and how much of the model they are.
Supporting another architecture means adding its patterns in `core/modifier.py` — and, if
its attention is shaped differently, checking that `core/layout.py`'s head rules still hold.
