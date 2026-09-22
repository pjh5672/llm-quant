# Configuration

The dtype axes, the three modes, and how a sweep grid is expanded.

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
