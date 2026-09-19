"""Deployed-cost metrics for a recipe, computed on the unmodified bf16 model.

Three numbers, because they do not agree with each other:

  deployed_bytes          what the model costs on disk
  decode_bytes_per_token  what decode reads per token, which is what decode speed tracks
                          (decode is memory bound, so this is the speed proxy)
  bits_per_element        gaia-compressor's BPV: weight bits plus the amortized scale bits,
                          parameter-weighted across the model. Normalized, so it compares
                          schemes rather than models.

lm_head is where they disagree. With tie_word_embeddings the embedding and lm_head share one
tensor; quantizing lm_head breaks the tie and *adds* a second copy, so disk grows. But decode
reads lm_head in full every token while the embedding is a row lookup, so the same change
*shrinks* decode traffic.

The KV cache is the fourth number and behaves differently again: it costs nothing on disk,
but decode re-reads the whole cache every step, so its share of decode traffic grows with the
context length. At short context the weights dominate and quantizing the cache buys almost
nothing; past a few thousand tokens it becomes the larger term.
"""

import torch.nn as nn

from llmquant.core.modifier import QuantizationModifier

SCALE_BYTES = 4  # fp32 scale
BF16_BYTES = 2
BF16_BITS = BF16_BYTES * 8


def _num_scales(linear: nn.Linear, args) -> int:
    """One scale per output channel, times the number of groups along in_features."""
    if args.strategy == "group":
        if linear.in_features % args.group_size:
            raise ValueError(f"in_features {linear.in_features} not divisible by group_size {args.group_size}")
        return linear.out_features * (linear.in_features // args.group_size)
    return linear.out_features


def bits_per_element(num_bits: int | None, group_size: int | None) -> float:
    """Weight bits plus the scale bits amortized over a group; bf16 when num_bits is None."""
    if num_bits is None:
        return float(BF16_BITS)
    return num_bits + SCALE_BYTES * 8 / group_size


def _linear_bytes(linear: nn.Linear, scheme):
    if scheme is None or scheme.weights is None:
        return linear.weight.numel() * BF16_BYTES
    args = scheme.weights
    return linear.weight.numel() * args.num_bits // 8 + _num_scales(linear, args) * SCALE_BYTES


def _linear_bpv(linear: nn.Linear, scheme) -> float:
    if scheme is None or scheme.weights is None:
        return float(BF16_BITS)
    args = scheme.weights
    group = args.group_size if args.strategy == "group" else linear.in_features
    return bits_per_element(args.num_bits, group)


def _scheme_of(recipe, name, module):
    if recipe is None or type(module).__name__ not in recipe.targets:
        return None
    return recipe.scheme_for(name)  # already applies `ignore` and the attn / mlp overrides


def model_metrics(model: nn.Module, recipe: QuantizationModifier | None) -> dict:
    """All three cost metrics in one walk. Call on the unmodified bf16 model."""
    lm_head = model.get_output_embeddings()
    embed = model.get_input_embeddings()
    tied = lm_head.weight is embed.weight
    lm_head_scheme = recipe.scheme_for("lm_head") if recipe else None

    counted = set()
    disk = 0
    decode = 0  # read in full for every generated token
    weighted_bits, elements = 0.0, 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name == "lm_head":
            continue
        scheme = _scheme_of(recipe, name, module)
        nbytes = _linear_bytes(module, scheme)
        disk += nbytes
        decode += nbytes
        weighted_bits += _linear_bpv(module, scheme) * module.weight.numel()
        elements += module.weight.numel()
        counted.add(id(module.weight))

    head_bytes = _linear_bytes(lm_head, lm_head_scheme)
    decode += head_bytes
    weighted_bits += _linear_bpv(lm_head, lm_head_scheme) * lm_head.weight.numel()
    elements += lm_head.weight.numel()

    for p in model.parameters():
        if id(p) in counted or (not tied and p is lm_head.weight) or p is embed.weight:
            continue
        nbytes = p.numel() * BF16_BYTES
        disk += nbytes
        decode += nbytes  # norms are read every token too, though they are tiny
    if lm_head_scheme is not None:
        disk += head_bytes + embed.weight.numel() * BF16_BYTES  # quantizing lm_head unties them
    elif tied:
        disk += head_bytes  # one shared tensor, counted once
    else:
        disk += head_bytes + embed.weight.numel() * BF16_BYTES

    kv_bits = recipe.kv_cache_bits if recipe else None
    return {
        "deployed_bytes": disk,
        "decode_bytes_per_token": decode,
        "bits_per_element": weighted_bits / elements,
        "kv_bytes_per_token": kv_cache_bytes_per_token(model, kv_bits),
    }


def kv_cache_bytes_per_token(model, num_bits: int | None, group_size: int | None = None) -> int:
    """Bytes the KV cache costs per token of context, for one sequence.

    Keys and values, every layer, every KV head. The group is the head dimension: KV
    quantization is per token and per head, and head_dim (64 here) is smaller than the
    weight group size (128), so the weight grouping cannot be reused.
    """
    config = model.config
    layers = config.num_hidden_layers
    kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    elements = 2 * layers * kv_heads * head_dim  # 2 = keys and values

    if num_bits is None:
        return elements * BF16_BYTES
    group = min(group_size or head_dim, head_dim)
    scales = 2 * layers * kv_heads * (head_dim // group)
    return elements * num_bits // 8 + scales * SCALE_BYTES


def decode_bytes_at_context(metrics: dict, context_tokens: int) -> int:
    """Weights read every step, plus the whole KV cache re-read every step."""
    return metrics["decode_bytes_per_token"] + context_tokens * metrics["kv_bytes_per_token"]


def estimate_deployed_bytes(model: nn.Module, recipe: QuantizationModifier | None) -> int:
    """Theoretical on-disk size of the quantized model."""
    return model_metrics(model, recipe)["deployed_bytes"]
