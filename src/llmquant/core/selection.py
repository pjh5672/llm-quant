"""How to pick a winner once several bit combinations are measured.

The original rule was "within 5% PPL, take the smallest model". Two things were wrong with
it. It cannot see speed at all, and disk size is the wrong proxy for speed anyway: decode is
memory bound, so decode speed tracks the bytes read per token, and lm_head makes those two
disagree in opposite directions (see llmquant.core.metrics).

So: accuracy stays a hard limit, but it is measured on generation tasks rather than
perplexity, and the ranking inside the limit is a weighted score over the costs that
actually matter -- bits per element, decode traffic, prefill latency.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionConfig:
    # hard limit on the accuracy drop, in percent of the bf16 score. None disables it.
    acc_drop_limit_pct: float | None = 5.0

    accuracy_weight: float = 1.0
    bpv_weight: float = 2.0
    decode_speed_weight: float = 2.0
    prefill_speed_weight: float = 1.0
    generation_weight: float = 1.0  # stage-2 agreement, when no task score exists

    def __post_init__(self):
        if self.acc_drop_limit_pct is not None and self.acc_drop_limit_pct < 0:
            raise ValueError(f"acc_drop_limit_pct must be >= 0, got {self.acc_drop_limit_pct}")
        for name in (
            "accuracy_weight",
            "bpv_weight",
            "decode_speed_weight",
            "prefill_speed_weight",
            "generation_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
