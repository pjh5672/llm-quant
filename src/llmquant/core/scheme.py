from dataclasses import dataclass

GROUP_SIZE = 128  # elements per scale along the reduction axis (K), for weights and activations


@dataclass(frozen=True)
class QuantizationArgs:
    num_bits: int
    # "group": one scale per group_size elements of the last (reduction) axis
    # "channel": one scale per weight output channel | "token": one scale per activation token
    strategy: str
    symmetric: bool = True
    dynamic: bool = False
    group_size: int | None = None
    # Set only for o_proj, whose K axis is heads x head_dim rather than a flat hidden
    # size. Grouping then happens inside each head instead of straddling two of them.
    head_dim: int | None = None
    # Output channels that share one scale. None means one scale per channel, the finest
    # option. Set to head_dim for q/k/v_proj, whose out axis is heads x head_dim, when the
    # target format stores one scale per output tile rather than per row.
    out_group: int | None = None

    def __post_init__(self):
        if self.strategy == "group":
            if not self.group_size or self.group_size <= 0:
                raise ValueError("strategy='group' requires a positive group_size")
        elif self.group_size is not None:
            raise ValueError(f"group_size is only valid for strategy='group', got {self.strategy!r}")


@dataclass(frozen=True)
class QuantizationScheme:
    weights: QuantizationArgs | None
    input_activations: QuantizationArgs | None


def _weight(bits, group_size=GROUP_SIZE):
    return QuantizationArgs(num_bits=bits, strategy="group", group_size=group_size)


def _activation(bits, group_size=GROUP_SIZE):
    # per-token AND per-group: each token's scales depend only on that token, so the
    # scheme stays causal and batch-invariant while aligning with the weight groups.
    return QuantizationArgs(num_bits=bits, strategy="group", group_size=group_size, dynamic=True)


PRESET_SCHEMES = {
    "W4A8": QuantizationScheme(weights=_weight(4), input_activations=_activation(8)),
    "W8A8": QuantizationScheme(weights=_weight(8), input_activations=_activation(8)),
    "W4A16": QuantizationScheme(weights=_weight(4), input_activations=None),
    "W8A16": QuantizationScheme(weights=_weight(8), input_activations=None),
}


def preset_name_to_scheme(name: str) -> QuantizationScheme:
    try:
        return PRESET_SCHEMES[name.upper()]
    except KeyError:
        raise ValueError(f"unknown scheme {name!r}, expected one of {list(PRESET_SCHEMES)}")
