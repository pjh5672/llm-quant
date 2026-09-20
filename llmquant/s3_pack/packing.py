"""Bit packing for sub-byte weights.

int8 is stored as-is. int4 goes two values to a byte: each value is shifted by +8 into
0..15, the even index takes the low nibble and the odd index the high one, so -3, 5 becomes
5, 13 becomes 0xD5.

The +8 shift matters more than it looks. int4 here is symmetric and signed, -8..7, and
writing a negative number into a nibble needs a convention; biasing into unsigned 0..15
means pack and unpack agree without either side reasoning about sign extension, which is
the sort of thing that silently costs half the value range when a kernel is written months
later against the file rather than against this code.
"""

import torch

INT4_BIAS = 8
INT4_MIN, INT4_MAX = -8, 7


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """[..., n] int8 in -8..7 -> [..., n // 2] uint8. The last axis must be even."""
    if values.shape[-1] % 2:
        raise ValueError(f"last dim {values.shape[-1]} must be even to pack into nibbles")
    if values.numel() and (values.min() < INT4_MIN or values.max() > INT4_MAX):
        raise ValueError(
            f"values must be in [{INT4_MIN}, {INT4_MAX}], got "
            f"[{int(values.min())}, {int(values.max())}]"
        )
    biased = (values.to(torch.int16) + INT4_BIAS).to(torch.uint8)
    low = biased[..., 0::2]
    high = biased[..., 1::2]
    return (low | (high << 4)).contiguous()


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """[..., n // 2] uint8 -> [..., n] int8. The inverse of pack_int4."""
    low = (packed & 0x0F).to(torch.int16) - INT4_BIAS
    high = ((packed >> 4) & 0x0F).to(torch.int16) - INT4_BIAS
    out = torch.stack((low, high), dim=-1)
    return out.reshape(*packed.shape[:-1], -1).to(torch.int8)


def pack_weight(qweight: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Storage form for an integer weight: packed for int4, unchanged for int8."""
    if num_bits == 8:
        return qweight.contiguous()
    if num_bits == 4:
        return pack_int4(qweight)
    raise ValueError(f"unsupported num_bits {num_bits}, expected 4 or 8")


def unpack_weight(stored: torch.Tensor, num_bits: int) -> torch.Tensor:
    if num_bits == 8:
        return stored.to(torch.int8)
    if num_bits == 4:
        return unpack_int4(stored)
    raise ValueError(f"unsupported num_bits {num_bits}, expected 4 or 8")
