"""How a weight is divided, padded and scaled -- defined once, for every stage.

Fake quant, real quant, packing and the kernel all have to agree on this exactly, or the
bit-exact checks between phases mean nothing: a packed file laid out one way and a kernel
expecting another will not disagree loudly, it will just be wrong. So the geometry lives
here and every stage asks this module rather than re-deriving it.

The rules, by where the head structure sits:

    module       reduction (in) axis             output axis
    q/k/v_proj   pad the tail up to a group      split per head, pad each head to a group
    o_proj       split per head, pad each head   pad the tail up to a group
    mlp, lm_head pad the tail up to a group      pad the tail up to a group

q/k/v have heads on the output axis because they produce them; o_proj has heads on the
reduction axis because it consumes them. Splitting per head keeps two heads with different
dynamic ranges out of the same group.

Padding is zeros, which is free for a symmetric abs-max scale -- a zero cannot move
max(|x|), so the real elements quantize exactly as they would without it. What padding does
cost is storage, and that is not free at all: a 64-wide head padded into a 128 group stores
two slots for every one that matters, so int8 q/k/v ends up the same size as bf16.
"""

from dataclasses import dataclass


def round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


@dataclass(frozen=True)
class WeightLayout:
    """Padded geometry and scale count for one Linear weight of shape [out, in]."""

    out_features: int
    in_features: int
    group_size: int
    in_head_dim: int | None = None  # set for o_proj
    out_head_dim: int | None = None  # set for q/k/v_proj

    def __post_init__(self):
        for axis, head_dim, size in (
            ("in", self.in_head_dim, self.in_features),
            ("out", self.out_head_dim, self.out_features),
        ):
            if head_dim and size % head_dim:
                raise ValueError(
                    f"{axis}_features {size} is not a multiple of head_dim {head_dim}"
                )

    @property
    def padded_in(self) -> int:
        """Reduction-axis length once padded; per head when the axis carries heads."""
        if self.in_head_dim:
            heads = self.in_features // self.in_head_dim
            return heads * round_up(self.in_head_dim, self.group_size)
        return round_up(self.in_features, self.group_size)

    @property
    def padded_out(self) -> int:
        if self.out_head_dim:
            heads = self.out_features // self.out_head_dim
            return heads * round_up(self.out_head_dim, self.group_size)
        return round_up(self.out_features, self.group_size)

    @property
    def in_groups(self) -> int:
        return self.padded_in // self.group_size

    @property
    def scale_rows(self) -> int:
        """Rows that share a scale: one per output head, else one per output channel."""
        if self.out_head_dim:
            return self.out_features // self.out_head_dim
        return self.out_features

    @property
    def num_scales(self) -> int:
        return self.scale_rows * self.in_groups

    @property
    def stored_elements(self) -> int:
        """Weight slots actually written, padding included."""
        return self.padded_out * self.padded_in

    def weight_bytes(self, num_bits: int) -> int:
        return self.stored_elements * num_bits // 8

    def bits_per_real_element(self, num_bits: int, scale_bytes: int) -> float:
        """Bits paid per weight that matters, so padding shows up instead of hiding.

        Counting only the slots that carry real data would report the same number whether
        or not a head was padded, while the file size moved.
        """
        real = self.out_features * self.in_features
        bits = self.weight_bytes(num_bits) * 8 + self.num_scales * scale_bytes * 8
        return bits / real


def layout_for(out_features: int, in_features: int, args, group_size=None) -> WeightLayout:
    """Build the layout a QuantizationArgs implies for one Linear."""
    return WeightLayout(
        out_features=out_features,
        in_features=in_features,
        group_size=group_size or args.group_size,
        in_head_dim=args.head_dim,
        out_head_dim=args.out_group,
    )
