"""How to pick a winner once several bit combinations are measured.

The old rule was "within 5% PPL, take the smallest model". It cannot see speed at all, and
disk size is the wrong proxy for it: decode is memory bound, so decode speed tracks the bytes
read per token, and lm_head makes those two disagree (see llmquant.core.metrics).

So: keep accuracy as an optional hard limit, then rank by a weighted score over three
quantities, each expressed as a percentage relative to the bf16 baseline.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionConfig:
    ppl_limit_ratio: float | None = 1.05  # None disables the hard limit
    accuracy_weight: float = 1.0
    bpv_weight: float = 2.0
    decode_speed_weight: float = 2.0
    generation_weight: float = 1.0  # used once generation metrics exist

    def __post_init__(self):
        if self.ppl_limit_ratio is not None and self.ppl_limit_ratio < 1.0:
            raise ValueError(f"ppl_limit_ratio must be >= 1.0, got {self.ppl_limit_ratio}")
        for name in ("accuracy_weight", "bpv_weight", "decode_speed_weight", "generation_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
