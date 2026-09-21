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
from llmquant.s4_kernel.expert_linear import KernelQuantExperts
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

    # A Mixture-of-Experts block is one module holding every expert, so it is written as
    # one entry with its two stacked tensors rather than as E separate layers. pack_weight
    # works on the last axis, which is the group axis either way, so the [E, out, groups,
    # gs] stack packs exactly as a [out, groups, gs] weight does.
    experts = {}
    for name, module in model.named_modules():
        if not isinstance(module, KernelQuantExperts):
            continue
        args = module.scheme.weights
        tensors[f"{name}.qgate_up"] = pack_weight(module.qgate_up, args.num_bits)
        tensors[f"{name}.sgate_up"] = module.sgate_up
        tensors[f"{name}.qdown"] = pack_weight(module.qdown, args.num_bits)
        tensors[f"{name}.sdown"] = module.sdown
        experts[name] = {
            "num_experts": module.num_experts,
            "hidden_dim": module.hidden_dim,
            "intermediate_dim": module.intermediate_dim,
            "gate_up_shape": list(module.qgate_up.shape),
            "down_shape": list(module.qdown.shape),
            "weights": _args_to_dict(args),
            "input_activations": _args_to_dict(module.scheme.input_activations),
        }

    # everything that was never quantized travels as bf16, so one file loads the whole model.
    # A quantized layer owns more than qweight and wscale -- it also carries buffers derived
    # from them, and writing those out as bf16 meant the loader overwrote a value its own
    # constructor had just computed exactly. So the whole layer is excluded by prefix, which
    # also means adding another derived buffer later cannot reintroduce the bug.
    owned = tuple(f"{layer}." for layer in (*layers, *experts))
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
        "experts": experts,
        "non_persistent_buffers": non_persistent,
        # the KV cache is quantized at generate time rather than stored, so nothing in the
        # weights records it; without this a packed file would silently chat in bf16 cache
        "kv_cache_bits": kv_cache_bits,
        **(extra or {}),
    }
    return write_packed(path, tensors, meta)


def _materialize_remaining(model, device, dtype, provided=()):
    """Give storage to whatever is still on meta and will not be assigned from the file.

    torch.nn.Module.to_empty() would do this for the whole model, including the quantized
    modules that already hold real int weights -- it would hand them fresh uninitialized
    memory and silently undo the load.

    `provided` is skipped because load_state_dict(assign=True) replaces those tensors
    outright. Allocating them first means the empty tensor and the loaded one are both
    resident for a moment, and on a large vocabulary the embedding alone made that
    half a gigabyte.
    """
    provided = set(provided)
    for prefix, module in model.named_modules():
        for name, param in list(module.named_parameters(recurse=False)):
            full = f"{prefix}.{name}" if prefix else name
            if param is not None and param.is_meta and full not in provided:
                empty = torch.empty(param.shape, dtype=dtype, device=device)
                setattr(module, name, nn.Parameter(empty, requires_grad=param.requires_grad))
        for name, buffer in list(module.named_buffers(recurse=False)):
            full = f"{prefix}.{name}" if prefix else name
            if buffer is not None and buffer.is_meta and full not in provided:
                persistent = name not in getattr(module, "_non_persistent_buffers_set", set())
                module.register_buffer(
                    name,
                    torch.empty(buffer.shape, dtype=buffer.dtype, device=device),
                    persistent=persistent,
                )


def load_packed_model(path, device="cuda", dtype=torch.bfloat16):
    """Rebuild the model from a packed file.

    Peak memory stays near the size of the
    packed weights: nothing is ever allocated at bf16 width, not even briefly.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    tensors, meta = read_packed(path)
    config = AutoConfig.from_pretrained(meta["model_id"])
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # The quantized modules are built and swapped in FIRST, while everything around them is
    # still on meta and costs nothing. to_empty() here instead would size every bf16
    # parameter before replacing it -- 12.95 GB of peak to load OLMoE's 6.76 GB, and no way
    # at all to load a packed model bigger than the card.
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

    for name, spec in (meta.get("experts") or {}).items():
        args = _args_from_dict(spec["weights"])
        scheme = QuantizationScheme(
            weights=args, input_activations=_args_from_dict(spec["input_activations"])
        )
        qgate_up = unpack_weight(tensors[f"{name}.qgate_up"], args.num_bits)
        qdown = unpack_weight(tensors[f"{name}.qdown"], args.num_bits)
        # act_fn is architecture, not weights: take it from the block being replaced
        module = KernelQuantExperts(
            qgate_up.reshape(spec["gate_up_shape"]).to(device),
            tensors[f"{name}.sgate_up"].to(device),
            qdown.reshape(spec["down_shape"]).to(device),
            tensors[f"{name}.sdown"].to(device),
            scheme,
            spec["hidden_dim"],
            spec["intermediate_dim"],
            model.get_submodule(name).act_fn,
        )
        _replace(model, name, module)

    _materialize_remaining(model, device, dtype, provided=tensors)

    non_persistent = set(meta.get("non_persistent_buffers", []))
    owned = tuple(f"{layer}." for layer in (*meta["layers"], *(meta.get("experts") or {})))
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


def packed_dtypes(path) -> dict:
    """The dtype each axis actually has in a packed file.

    Without this a run started with --load-packed reports whatever the config happened to
    say, which for a bare --load-packed is bf16 on every axis -- a quantized model
    described as unquantized. The file knows; it records num_bits per layer.
    """
    meta = read_header(path)["meta"]
    groups = {"attn_weight": set(), "mlp_weight": set(), "head_weight": set()}
    for name, spec in (meta.get("layers") or {}).items():
        if ".self_attn." in name:
            key = "attn_weight"
        elif ".mlp." in name:
            key = "mlp_weight"
        elif "lm_head" in name:
            key = "head_weight"
        else:
            continue
        bits = (spec.get("weights") or {}).get("num_bits")
        if bits:
            groups[key].add(bits)

    def name_for(bits_set):
        if not bits_set:
            return "bf16"
        if len(bits_set) > 1:
            return "mixed"
        return f"int{next(iter(bits_set))}"

    kv_bits = meta.get("kv_cache_bits")
    return {
        **{k: name_for(v) for k, v in groups.items()},
        "kv_cache": f"int{kv_bits}" if kv_bits else "bf16",
        "activation": "bf16",  # the kernel path is weight-only; see s4_kernel
    }


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
