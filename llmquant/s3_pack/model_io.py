"""Save a quantized model to one file, and load it back without touching bf16 weights.

The point of loading from a packed file is that the bf16 model never exists: the
architecture comes from the model id's config, the quantized layers are built straight from
the packed integers, and only the tensors that were never quantized are read as bf16.

What is stored per quantized layer is the canonical layout the kernel already consumes --
[N, groups, group_size], padded and head-split -- plus its scales. Nothing is reshaped at
load, so there is no second place that could disagree with core.layout about where a group
begins.
"""

import torch
import torch.nn as nn

from llmquant.core.scheme import QuantizationArgs, QuantizationScheme
from llmquant.s3_pack.format import read_header, read_packed, write_packed
from llmquant.s3_pack.packing import pack_weight, unpack_weight
from llmquant.s4_kernel.quant_linear import KernelQuantLinear

def _args_to_dict(args: QuantizationArgs | None):
    if args is None:
        return None
    return {
        "num_bits": args.num_bits,
        "strategy": args.strategy,
        "group_size": args.group_size,
        "head_dim": args.head_dim,
        "out_group": args.out_group,
        "symmetric": args.symmetric,
        "dynamic": args.dynamic,
    }


def _args_from_dict(payload):
    return None if payload is None else QuantizationArgs(**payload)


def save_packed_model(
    model: nn.Module,
    path,
    model_id: str,
    kv_cache_bits: int | None = None,
    extra: dict | None = None,
):
    """Write every weight of a kernel-mode model into one file."""
    tensors, layers = {}, {}

    for name, module in model.named_modules():
        if not isinstance(module, KernelQuantLinear):
            continue
        args = module.scheme.weights
        tensors[f"{name}.qweight"] = pack_weight(module.qweight, args.num_bits)
        tensors[f"{name}.wscale"] = module.wscale
        layers[name] = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "canonical_shape": list(module.qweight.shape),
            "weights": _args_to_dict(args),
            "input_activations": _args_to_dict(module.scheme.input_activations),
            "bias": module.bias is not None,
        }
        if module.bias is not None:
            tensors[f"{name}.bias"] = module.bias

    # everything that was never quantized travels as bf16, so one file loads the whole model.
    # A quantized layer owns more than qweight and wscale -- it also carries buffers derived
    # from them, and writing those out as bf16 meant the loader overwrote a value its own
    # constructor had just computed exactly. So the whole layer is excluded by prefix, which
    # also means adding another derived buffer later cannot reintroduce the bug.
    owned = tuple(f"{layer}." for layer in layers)
    state = model.state_dict()
    for name, tensor in state.items():
        if name.startswith(owned) or name in tensors:
            continue
        tensors[name] = tensor.detach().to(torch.bfloat16)

    # Non-persistent buffers are absent from state_dict, and the loader builds the model on
    # meta and calls to_empty(), which hands back *uninitialized* memory for them. RoPE's
    # inv_freq is one, so skipping them leaves the position encoding reading garbage --
    # weights load bit-exact and the logits still come out wrong. They are tiny, and kept in
    # their own dtype because inv_freq in bf16 is not precise enough to encode a position.
    non_persistent = []
    for name, buffer in model.named_buffers():
        if name.startswith(owned) or name in state or name in tensors:
            continue
        tensors[name] = buffer.detach()
        non_persistent.append(name)

    meta = {
        "model_id": model_id,
        "layers": layers,
        "non_persistent_buffers": non_persistent,
        # the KV cache is quantized at generate time rather than stored, so nothing in the
        # weights records it; without this a packed file would silently chat in bf16 cache
        "kv_cache_bits": kv_cache_bits,
        **(extra or {}),
    }
    return write_packed(path, tensors, meta)


def load_packed_model(path, device="cuda", dtype=torch.bfloat16):
    """Rebuild the model from a packed file. The bf16 weights are never materialized."""
    from transformers import AutoConfig, AutoModelForCausalLM

    tensors, meta = read_packed(path)
    config = AutoConfig.from_pretrained(meta["model_id"])
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    model = model.to_empty(device=device)

    for name, spec in meta["layers"].items():
        args = _args_from_dict(spec["weights"])
        qweight = unpack_weight(tensors[f"{name}.qweight"], args.num_bits)
        qweight = qweight.reshape(spec["canonical_shape"]).to(device)
        scheme = QuantizationScheme(
            weights=args, input_activations=_args_from_dict(spec["input_activations"])
        )
        bias = tensors.get(f"{name}.bias")
        module = KernelQuantLinear(
            qweight,
            tensors[f"{name}.wscale"].to(device),
            nn.Parameter(bias.to(device=device, dtype=dtype)) if bias is not None else None,
            scheme,
            spec["in_features"],
        )
        _replace(model, name, module)

    non_persistent = set(meta.get("non_persistent_buffers", []))
    owned = tuple(f"{layer}." for layer in meta["layers"])
    remaining = {
        name: tensor.to(device=device, dtype=dtype)
        for name, tensor in tensors.items()
        if name not in non_persistent and not name.startswith(owned)
    }
    _, unexpected = model.load_state_dict(remaining, strict=False, assign=True)
    unexpected = [n for n in unexpected if not n.startswith(owned)]
    if unexpected:
        raise ValueError(f"packed file has tensors the model does not want: {unexpected[:5]}")

    # assigned directly: load_state_dict would call these unexpected, and to_empty() left
    # them pointing at uninitialized memory
    for name in non_persistent:
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent.register_buffer(attribute, tensors[name].to(device), persistent=False)
    return model.eval()


def _replace(model: nn.Module, name: str, module: nn.Module):
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, module)


def describe_packed(path) -> dict:
    """Header only: what is in the file, without paging in the weights."""
    header = read_header(path)
    entries = header["tensors"]
    return {
        "model_id": header["meta"]["model_id"],
        "quantized_layers": len(header["meta"]["layers"]),
        "tensors": len(entries),
        "bytes": sum(e["nbytes"] for e in entries.values()),
    }
