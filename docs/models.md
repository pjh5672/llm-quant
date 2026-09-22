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
a quantized config having quantized nothing. So stage 1 of `examples/sweep.py` checks and
says so before any measurement happens:

```
! NOTHING MATCHED. The recipe's patterns are Llama naming (self_attn.*_proj, mlp.*_proj,
  lm_head) and this model uses different names, so every weight stays bf16 [...]
```

or, for a partial match, which Linears were missed and how much of the model they are.
Supporting another architecture means adding its patterns in `core/modifier.py` — and, if
its attention is shaped differently, checking that `core/layout.py`'s head rules still hold.
