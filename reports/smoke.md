# Quantization study: meta-llama/Llama-3.2-1B-Instruct

Decode traffic is quoted at 2048 tokens of context: weights plus the whole KV cache, re-read every step.

## Model

| field | value |
|---|---|
| model | meta-llama/Llama-3.2-1B-Instruct |
| layers | 16 |
| hidden | 2048 |
| heads | 32 |
| kv heads | 8 |
| head_dim | 64 |
| vocab | 128256 |
| tied embeddings | True |

### What this means for the numbers below

- lm_head is TIED to the embedding, so head_weight=int8 makes the file BIGGER (about +258 MB of int8 copy beside the bf16 embedding), while decode traffic falls -- decode reads lm_head in full every token but only row-indexes the embedding. Judge head_weight on decode, not disk.
- head_dim 64 is narrower than group_size 128: every head is padded into a full group, so q/k/v and o_proj store 2x the slots they use. Quantizing attention may not shrink it at all.
- attn: stores 2.00x the weights it uses (335.5M slots for 167.8M weights)
- mlp holds 54% of the parameters (attn 11%, embedding 18%, lm_head 18%, mlp 54%) -- that is where a lower weight dtype can pay, and where it can hurt.

## Selected

**`attn-int8_mlp-int8_head-bf16_act-bf16_kv-int8_g128`** -- gives up 6.67% of the bf16 accuracy for BPV 10.98 and 1.6125 GB/token of decode traffic.

2 of 3 runs met the limit.

### Weighted ranking

| score | dacc% | BPV | dec GB/t | name |
|---|---|---|---|---|
| 136.24 | +6.67 | 10.98 | 1.6125 | `attn-int8_mlp-int8_head-bf16_act-bf16_kv-int8_g128` |
| 131.28 | +6.67 | 10.98 | 1.6418 | `attn-int8_mlp-int8_head-bf16_act-bf16_g128` |

## Verification (measured on the real path)

| run | mode | TPS | TTFT ms | peak VRAM | acc |
|---|---|---|---|---|---|
| bf16 baseline | kernel | 78.0 | 22.9 | 2.33 | 0.7500 |
| selected | kernel | 47.1 | 58.7 | 1.63 | 0.7000 |
| selected, kv_cache=bf16 | kernel | 69.6 | 47.6 | 1.63 | 0.7000 |

### Verdict

- decode: projected 1.47x, measured 0.60x -- the projection does NOT hold up (47.1 vs 78.0 tok/s).
- the selected combination decodes SLOWER than bf16. Quantized weights only pay when weight bandwidth dominates the step; on a small model it may not.
- kv_cache=int8 costs 32% of decode throughput (69.6 tok/s without it). It runs outside the weight kernel, so the analytic traffic estimate cannot see it.
- TTFT is 2.57x the baseline: prefill dequantizes weights every pass. Fusing dequant into the GEMM mainloop is what removes it.
- peak VRAM is 0.70x the baseline.

## Every combination

| attn | mlp | head | act | kv | acc | dacc% | PPL | BPV | kv KB/t | dec GB/t | TTFT ms | TPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | bf16 | bf16 | bf16 | bf16 | 0.7500 | +0.00 | - | 16.00 | 32.0 | 2.3644 | 18.2 | 77.5 |
| int8 | int8 | bf16 | bf16 | bf16 | 0.7000 | +6.67 | - | 10.98 | 32.0 | 1.6418 | 16.8 | 76.5 |
| int8 | int8 | bf16 | bf16 | int8 | 0.7000 | +6.67 | - | 10.98 | 17.0 | 1.6125 | 28.3 | 45.9 |

`dacc%` is accuracy lost against bf16 on the generation tasks. TTFT and TPS come from the sweep path; if that path is fake quant they describe the simulation, not a deployment -- see Verification.

## Where the accuracy goes

One axis changed, the rest held fixed.

| change | n | mean dacc% | min | max | dBPV |
|---|---|---|---|---|---|
| `kv_cache: bf16 -> int8` | 1 | +0.00 | +0.00 | +0.00 | +0.00 |

