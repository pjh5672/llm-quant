"""What about this model changes how its sweep should be read.

A sweep reports the same columns for every model, but two structural facts decide what
those columns mean, and both caught us out on Llama-3.2-1B:

  tied embeddings   when lm_head shares storage with the embedding, quantizing it does not
                    shrink a tensor -- it unties one and adds an int8 copy beside the bf16
                    embedding. The file grows while decode traffic falls, and the disk and
                    speed criteria look like they conflict when they are simply describing
                    different tensors.
  head_dim padding  a head narrower than the group size is padded into a full group, so
                    those weights store more slots than they use. At head_dim 64 and group
                    128, int8 q/k/v weighs exactly what bf16 weighs and the whole point of
                    quantizing attention disappears.

Neither is visible in a results table, so a run that does not print them invites the same
two days of confusion on the next model.
"""

import torch.nn as nn

from llmquant.core.layout import layout_for
from llmquant.core.metrics import SCALE_BYTES

MB = 1024**2


def _config_of(model):
    return getattr(model, "config", None)


def _head_dim(config) -> int | None:
    if config is None:
        return None
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and getattr(config, "num_attention_heads", None):
        head_dim = config.hidden_size // config.num_attention_heads
    return head_dim


def _group_of(name: str) -> str:
    if ".self_attn." in name:
        return "attn"
    if ".mlp." in name:
        return "mlp"
    if "lm_head" in name:
        return "lm_head"
    return "other"


def padding_overhead(model, recipe) -> dict:
    """Stored weight slots vs slots that carry data, per target group.

    A ratio above 1.0 is padding: storage paid for nothing. It is what makes int8
    attention weigh as much as bf16 attention on a narrow-headed model.
    """
    recipe = recipe.resolve(model) if recipe is not None else None
    totals = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        group = _group_of(name)
        if group == "other":
            continue
        args = None
        if recipe is not None:
            scheme = recipe.scheme_for(name)
            args = scheme.weights if scheme is not None else None
        real = module.out_features * module.in_features
        if args is None:
            stored = real
        else:
            stored = layout_for(module.out_features, module.in_features, args).stored_elements
        entry = totals.setdefault(group, {"real": 0, "stored": 0, "modules": 0})
        entry["real"] += real
        entry["stored"] += stored
        entry["modules"] += 1
    for entry in totals.values():
        entry["overhead"] = entry["stored"] / entry["real"] if entry["real"] else 1.0
    return totals


def pattern_coverage(model, recipe) -> dict:
    """Which Linears the recipe's patterns actually reach.

    The patterns are Llama naming -- `self_attn.*_proj`, `mlp.*_proj`, `lm_head`. On a
    model that names things differently the recipe matches nothing and every weight stays
    bf16, which is safe and silent: the run finishes, reports a quantized config, and has
    quantized nothing. This is what makes that visible.
    """
    import torch.nn as nn

    recipe = recipe.resolve(model) if recipe is not None else None
    matched = unmatched = 0
    missed = []
    fused_qkv = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        params = module.weight.numel()
        covered = recipe is not None and recipe.scheme_for(name) is not None
        if covered:
            matched += params
            leaf = name.rsplit(".", 1)[-1]
            # a fused qkv projection is handled -- its output axis is still head_dim-sized
            # heads -- but only if head_dim actually divides it. Anything else in attention
            # is a shape this project has not reasoned about.
            if ".self_attn." in name and leaf not in (
                "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"
            ):
                fused_qkv.append(name)
        else:
            unmatched += params
            missed.append(name)
    total = matched + unmatched
    return {
        "matched_params": matched,
        "unmatched_params": unmatched,
        "matched_fraction": matched / total if total else 0.0,
        "unmatched_modules": missed,
        "unexpected_attn_modules": fused_qkv,
    }


def parameter_split(model) -> dict:
    """Where the parameters actually are, which is where quantizing them can pay."""
    split = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key = _group_of(name)
        elif isinstance(module, nn.Embedding):
            key = "embedding"
        else:
            continue
        split[key] = split.get(key, 0) + module.weight.numel()
    return split


def model_facts(model, recipe=None, group_size=None) -> dict:
    """Structural facts worth printing beside any sweep of this model."""
    config = _config_of(model)
    head_dim = _head_dim(config)
    tied = bool(getattr(config, "tie_word_embeddings", False)) if config else False
    facts = {
        "model_id": getattr(config, "_name_or_path", None) if config else None,
        "hidden_size": getattr(config, "hidden_size", None) if config else None,
        "num_layers": getattr(config, "num_hidden_layers", None) if config else None,
        "vocab_size": getattr(config, "vocab_size", None) if config else None,
        "num_attention_heads": getattr(config, "num_attention_heads", None) if config else None,
        "num_key_value_heads": getattr(config, "num_key_value_heads", None) if config else None,
        "head_dim": head_dim,
        "tie_word_embeddings": tied,
        "parameters": parameter_split(model),
    }
    if recipe is not None:
        facts["padding"] = padding_overhead(model, recipe)
        facts["coverage"] = pattern_coverage(model, recipe)
        if group_size is None:
            for target in ("attn_scheme", "mlp_scheme", "lm_head_scheme"):
                scheme = getattr(recipe, target, None)
                if scheme is not None and scheme.weights is not None:
                    group_size = scheme.weights.group_size
                    break
        facts["group_size"] = group_size
        if head_dim and group_size:
            facts["head_fits_group"] = head_dim % group_size == 0 or group_size % head_dim == 0
            facts["heads_per_group"] = group_size / head_dim
    return facts


def fact_warnings(facts: dict) -> list[str]:
    """The consequences that a results table will not show on its own."""
    notes = []
    coverage = facts.get("coverage")
    if coverage is not None:
        fraction = coverage["matched_fraction"]
        if fraction < 0.01:
            notes.append(
                "NOTHING MATCHED. The recipe's patterns are Llama naming "
                "(self_attn.*_proj, mlp.*_proj, lm_head) and this model uses different "
                "names, so every weight stays bf16 and the run will report a quantized "
                "config having quantized nothing. Add patterns in core/modifier.py."
            )
        elif fraction < 0.95:
            missed = coverage["unmatched_modules"]
            shown = ", ".join(missed[:3]) + ("..." if len(missed) > 3 else "")
            notes.append(
                f"only {fraction:.0%} of Linear parameters match the recipe's patterns; "
                f"{len(missed)} Linears stay bf16 ({shown}). Accuracy will look better "
                "than the dtype suggests and the size saving will be smaller."
            )
        if coverage["unexpected_attn_modules"]:
            names = ", ".join(coverage["unexpected_attn_modules"][:3])
            notes.append(
                f"attention Linears that are not q/k/v/qkv/o_proj: {names}. These are "
                "quantized, but the head rules in core/layout.py were written for those "
                "names and have not been checked against this shape."
            )
    if facts.get("tie_word_embeddings"):
        vocab, hidden = facts.get("vocab_size"), facts.get("hidden_size")
        extra = ""
        if vocab and hidden:
            added = (vocab * hidden + vocab * hidden / 128 * SCALE_BYTES) / MB
            extra = f" (about +{added:.0f} MB of int8 copy beside the bf16 embedding)"
        notes.append(
            "lm_head is TIED to the embedding, so head_weight=int8 makes the file BIGGER"
            f"{extra}, while decode traffic falls -- decode reads lm_head in full every "
            "token but only row-indexes the embedding. Judge head_weight on decode, not disk."
        )
    head_dim, group_size = facts.get("head_dim"), facts.get("group_size")
    if head_dim and group_size and head_dim < group_size:
        notes.append(
            f"head_dim {head_dim} is narrower than group_size {group_size}: every head is "
            f"padded into a full group, so q/k/v and o_proj store {group_size / head_dim:.0f}x "
            "the slots they use. Quantizing attention may not shrink it at all."
        )
    for group, entry in sorted((facts.get("padding") or {}).items()):
        if entry["overhead"] > 1.01:
            notes.append(
                f"{group}: stores {entry['overhead']:.2f}x the weights it uses "
                f"({entry['stored'] / 1e6:.1f}M slots for {entry['real'] / 1e6:.1f}M weights)"
            )
    params = facts.get("parameters") or {}
    total = sum(params.values())
    if total:
        biggest = max(params, key=params.get)
        notes.append(
            f"{biggest} holds {params[biggest] / total:.0%} of the parameters "
            f"({', '.join(f'{k} {v / total:.0%}' for k, v in sorted(params.items()))}) "
            "-- that is where a lower weight dtype can pay, and where it can hurt."
        )
    return notes
