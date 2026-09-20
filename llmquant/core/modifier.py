import re
from dataclasses import dataclass, replace

import torch.nn as nn

from llmquant.core.scheme import QuantizationScheme, preset_name_to_scheme

# Llama naming. attn_scheme / mlp_scheme are what the config's attn_weight / mlp_weight map to.
ATTN_PATTERN = r"re:.*\.self_attn\..*_proj$"
MLP_PATTERN = r"re:.*\.mlp\..*_proj$"
# o_proj is the only Linear whose input carries head structure: it consumes the
# concatenated attention output, so its K axis is heads x head_dim.
O_PROJ_PATTERN = r"re:.*\.self_attn\.o_proj$"
# q/k/v are the mirror image: their *output* axis is heads x head_dim.
#
# A fused projection belongs here too. Phi-3's qkv_proj emits q, k and v from one Linear,
# and its output axis is still nothing but head_dim-sized heads -- (num_heads +
# 2 * num_kv_heads) of them -- so the q/k/v boundaries fall exactly on head boundaries and
# splitting per head separates them for free. That makes the per-head rule more necessary
# here than on a split projection, not less: without it a q head and a k head, which have
# no reason to share a dynamic range, would share a scale.
QKV_PATTERN = r"re:.*\.self_attn\.(q|k|v|qkv)_proj$"


def _resolve(scheme):
    if scheme is None or isinstance(scheme, QuantizationScheme):
        return scheme
    return preset_name_to_scheme(scheme)


def _matches(name, patterns):
    for p in patterns:
        if p.startswith("re:") and re.match(p[3:], name):
            return True
        if name == p:
            return True
    return False


def _set_module(model, name, module):
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


@dataclass
class QuantizationModifier:
    scheme: str | QuantizationScheme | None = None  # default for targeted Linears
    targets: tuple[str, ...] = ("Linear",)
    ignore: tuple[str, ...] = ("lm_head",)
    lm_head_scheme: str | QuantizationScheme | None = None  # None keeps lm_head bf16
    mode: str = "fake"
    attn_scheme: str | QuantizationScheme | None = None  # overrides `scheme` for self_attn
    mlp_scheme: str | QuantizationScheme | None = None  # overrides `scheme` for mlp
    # KV cache bit width, already resolved; None keeps the cache in bf16. Not a scheme:
    # the cache is grouped by head_dim, not by the weight group size.
    kv_cache_bits: int | None = None
    # filled in from the model by resolve(); drives the o_proj grouping
    head_dim: int | None = None
    # one scale per q/k/v output head instead of per output row. Matches a format that
    # stores a scale per output tile, and costs accuracy: measured 2.6x the weight error
    # on q_proj. Turn off to keep the finer per-row scales.
    qkv_out_scale_per_head: bool = True

    def resolve(self, model) -> "QuantizationModifier":
        """Read the head size off the model. Called before apply() and before costing."""
        config = getattr(model, "config", None)
        if config is not None and self.head_dim is None:
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None and getattr(config, "num_attention_heads", None):
                head_dim = config.hidden_size // config.num_attention_heads
            self.head_dim = head_dim
        return self

    def _head_aware(self, scheme):
        """Same scheme, but grouped inside each head instead of across two of them."""
        if scheme is None or not self.head_dim:
            return scheme
        return QuantizationScheme(
            weights=replace(scheme.weights, head_dim=self.head_dim) if scheme.weights else None,
            input_activations=(
                replace(scheme.input_activations, head_dim=self.head_dim)
                if scheme.input_activations
                else None
            ),
        )

    def _out_head_aware(self, scheme):
        """One scale per output head. Only the weight: an activation has no output axis."""
        if scheme is None or not self.head_dim or scheme.weights is None:
            return scheme
        return QuantizationScheme(
            weights=replace(scheme.weights, out_group=self.head_dim),
            input_activations=scheme.input_activations,
        )

    def scheme_for(self, name: str) -> QuantizationScheme | None:
        """Scheme that applies to the Linear called `name`, or None to keep it bf16."""
        if name == "lm_head":
            return _resolve(self.lm_head_scheme)
        if _matches(name, self.ignore):
            return None
        if self.attn_scheme is not None and _matches(name, (ATTN_PATTERN,)):
            scheme = _resolve(self.attn_scheme)
            if _matches(name, (O_PROJ_PATTERN,)):
                return self._head_aware(scheme)  # head structure on the reduction axis
            if _matches(name, (QKV_PATTERN,)) and self.qkv_out_scale_per_head:
                return self._out_head_aware(scheme)  # head structure on the output axis
            return scheme
        if self.mlp_scheme is not None and _matches(name, (MLP_PATTERN,)):
            return _resolve(self.mlp_scheme)
        return _resolve(self.scheme)

    def apply(self, model: nn.Module) -> nn.Module:
        # imported here, not at module level, so core never depends on a stage until a
        # recipe is actually applied; see llmquant/__init__.py
        from llmquant.modes import quant_linear_for

        quant_cls = quant_linear_for(self.mode)
        self.resolve(model)

        replacements = []
        for name, module in model.named_modules():
            if type(module).__name__ not in self.targets:
                continue
            scheme = self.scheme_for(name)
            if scheme is not None:
                replacements.append((name, quant_cls.from_linear(module, scheme)))
        for name, new_module in replacements:
            _set_module(model, name, new_module)
        return model
