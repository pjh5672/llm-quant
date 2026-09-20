"""Run config: one data type per target, one group size shared by all of them.

    attn_weight: int4      # self_attn.{q,k,v,o}_proj
    mlp_weight:  int4      # mlp.{gate,up,down}_proj
    head_weight: bf16      # lm_head
    activation:  int8      # the input of every quantized Linear above (int8 or bf16 only)
    kv_cache:    bf16      # K/V cache -- slot reserved, not implemented yet

Activations share the weights' group size on purpose: the int GEMM only decomposes into
per-group partial sums if both sides split the K axis on the same boundaries.
"""

from dataclasses import dataclass

from llmquant.core.modifier import QuantizationModifier
from llmquant.core.scheme import (
    GROUP_SIZE,
    QuantizationArgs,
    QuantizationScheme,
)

@dataclass
class ModelArgs:
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass
class DatasetArgs:
    dataset: str = "wikitext2"
    seqlen: int = 2048


BF16 = "bf16"
DTYPE_BITS = {"int4": 4, "int8": 8, BF16: None}
WEIGHT_TARGETS = ("attn_weight", "mlp_weight", "head_weight")
# Activations are int8 or bf16 only. int4 activations are outside the confirmed design:
# the output formula and the packing format assume an int8 activation, and an int4 x int4
# GEMM is a separate kernel path that neither CUTLASS nor QQQ gives us for free.
ACTIVATION_DTYPES = ("int8", BF16)
# The KV cache is int8 or bf16 only. int4 was measured and is not a trade worth offering:
# on its own it costs +23.23% of generation accuracy, and combined with int4 MLP weights it
# takes the model below random guessing (0.0989 against 0.25 for a four-way choice) while
# perplexity reports the damage as +0.18%, because a perplexity pass never reads the cache
# back. int8 costs +1.28% and halves cache traffic, so the useful range is int8 to bf16.
KV_CACHE_DTYPES = ("int8", BF16)


def normalize_dtype(dtype) -> str:
    """YAML `null`, a missing key and "bf16" all mean "leave this target in bf16"."""
    if dtype is None:
        return BF16
    key = str(dtype).strip().lower()
    if key not in DTYPE_BITS:
        raise ValueError(f"unknown dtype {dtype!r}, expected one of {sorted(DTYPE_BITS)} or null")
    return key


def dtype_to_args(dtype, group_size: int, dynamic: bool) -> QuantizationArgs | None:
    """None for bf16, otherwise symmetric group-wise int args over the last (K) axis."""
    bits = DTYPE_BITS[normalize_dtype(dtype)]
    if bits is None:
        return None
    return QuantizationArgs(
        num_bits=bits, strategy="group", group_size=group_size, dynamic=dynamic
    )


@dataclass(frozen=True)
class QuantConfig:
    attn_weight: str = BF16
    mlp_weight: str = BF16
    head_weight: str = BF16
    activation: str = BF16
    kv_cache: str = BF16
    group_size: int = GROUP_SIZE
    # q/k/v store one scale per output head rather than per output row. Matches a format
    # with one scale per output tile; measured cost is 2.6x the q_proj weight error.
    qkv_out_scale_per_head: bool = True

    def __post_init__(self):
        if not isinstance(self.group_size, int) or self.group_size <= 0:
            raise ValueError(f"group_size must be a positive int, got {self.group_size!r}")
        for name in (*WEIGHT_TARGETS, "activation", "kv_cache"):
            normalize_dtype(getattr(self, name))  # validate every field up front
        if normalize_dtype(self.activation) not in ACTIVATION_DTYPES:
            raise ValueError(
                f"activation={self.activation!r}: activations support only "
                f"{list(ACTIVATION_DTYPES)}; see docs/w4a8_rtn_notes.md."
            )
        if normalize_dtype(self.kv_cache) not in KV_CACHE_DTYPES:
            raise ValueError(
                f"kv_cache={self.kv_cache!r}: the KV cache supports only "
                f"{list(KV_CACHE_DTYPES)}; see docs/w4a8_rtn_notes.md."
            )

    def scheme(self, target: str) -> QuantizationScheme | None:
        """Scheme for one weight target, or None to leave it bf16.

        A bf16 weight means the whole target stays bf16, activation included: feeding an
        int8 activation to a bf16 weight buys nothing, there is no int GEMM to run.
        """
        if target not in WEIGHT_TARGETS:
            raise ValueError(f"unknown target {target!r}, expected one of {list(WEIGHT_TARGETS)}")
        weights = dtype_to_args(getattr(self, target), self.group_size, dynamic=False)
        if weights is None:
            return None
        activations = dtype_to_args(self.activation, self.group_size, dynamic=True)
        return QuantizationScheme(weights=weights, input_activations=activations)

    @property
    def kv_cache_bits(self) -> int | None:
        """Bit width for the KV cache, or None to keep it in bf16.

        Not part of a QuantizationScheme: the cache is grouped per token and per head, so
        its group is head_dim (64 here), which is smaller than the weight group size (128).
        """
        return DTYPE_BITS[normalize_dtype(self.kv_cache)]

    def to_modifier(self, mode: str = "fake") -> QuantizationModifier:
        return QuantizationModifier(
            scheme=None,  # every Linear is covered by the attn / mlp / lm_head targets
            attn_scheme=self.scheme("attn_weight"),
            mlp_scheme=self.scheme("mlp_weight"),
            lm_head_scheme=self.scheme("head_weight"),
            kv_cache_bits=self.kv_cache_bits,
            qkv_out_scale_per_head=self.qkv_out_scale_per_head,
            mode=mode,
        )

    def describe(self) -> str:
        """Short, filesystem-safe label used for experiment directories and result rows."""
        parts = [f"attn-{normalize_dtype(self.attn_weight)}", f"mlp-{normalize_dtype(self.mlp_weight)}",
                 f"head-{normalize_dtype(self.head_weight)}", f"act-{normalize_dtype(self.activation)}"]
        if normalize_dtype(self.kv_cache) != BF16:
            parts.append(f"kv-{normalize_dtype(self.kv_cache)}")
        return "_".join(parts) + f"_g{self.group_size}"
