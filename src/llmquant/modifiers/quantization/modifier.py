import re
from dataclasses import dataclass

import torch.nn as nn

from llmquant.modifiers.quantization.scheme import QuantizationScheme, preset_name_to_scheme
from llmquant.modules.fake_quant_linear import FakeQuantLinear

QUANT_LINEAR_BY_MODE = {
    "fake": FakeQuantLinear,
}

# Llama naming. attn_scheme / mlp_scheme are what the config's attn_weight / mlp_weight map to.
ATTN_PATTERN = r"re:.*\.self_attn\..*_proj$"
MLP_PATTERN = r"re:.*\.mlp\..*_proj$"


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

    def scheme_for(self, name: str) -> QuantizationScheme | None:
        """Scheme that applies to the Linear called `name`, or None to keep it bf16."""
        if name == "lm_head":
            return _resolve(self.lm_head_scheme)
        if _matches(name, self.ignore):
            return None
        if self.attn_scheme is not None and _matches(name, (ATTN_PATTERN,)):
            return _resolve(self.attn_scheme)
        if self.mlp_scheme is not None and _matches(name, (MLP_PATTERN,)):
            return _resolve(self.mlp_scheme)
        return _resolve(self.scheme)

    def apply(self, model: nn.Module) -> nn.Module:
        if self.mode not in QUANT_LINEAR_BY_MODE:
            raise ValueError(f"unsupported mode {self.mode!r}, expected one of {list(QUANT_LINEAR_BY_MODE)}")
        quant_cls = QUANT_LINEAR_BY_MODE[self.mode]

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
