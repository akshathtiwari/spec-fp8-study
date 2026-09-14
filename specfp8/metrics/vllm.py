"""vLLM Prometheus counter mapping.

vLLM counter names (as of v0.20+):
- vllm:spec_decode_draft_acceptance_rate — but this is a gauge, not raw counts
- vllm:num_spec_tokens_total — total draft tokens proposed
- vllm:num_accepted_tokens_total — total accepted tokens
- vllm:num_spec_decode_steps_total — total verification steps

Counter names may vary across versions. If the expected counters are
missing, parse() returns None and τ is recorded as null.
"""

from __future__ import annotations

from specfp8.metrics.base import SpecStats


class VllmMetricsScraper:
    """Maps vLLM Prometheus counters to SpecStats."""

    # Primary counter names (vLLM v0.20+)
    DRAFT_TOKENS = "vllm:num_spec_tokens_total"
    ACCEPTED_TOKENS = "vllm:num_accepted_tokens_total"
    VERIFICATION_STEPS = "vllm:num_spec_decode_steps_total"

    # Fallback names (older versions)
    FALLBACK_NAMES = {
        "draft": [
            "vllm:num_spec_tokens_total",
            "vllm_num_spec_tokens_total",
            "num_spec_tokens_total",
        ],
        "accepted": [
            "vllm:num_accepted_tokens_total",
            "vllm_num_accepted_tokens_total",
            "num_accepted_tokens_total",
        ],
        "steps": [
            "vllm:num_spec_decode_steps_total",
            "vllm_num_spec_decode_steps_total",
            "num_spec_decode_steps_total",
        ],
    }

    def parse(self, raw: dict[str, float]) -> SpecStats | None:
        draft = self._find(raw, self.FALLBACK_NAMES["draft"])
        accepted = self._find(raw, self.FALLBACK_NAMES["accepted"])
        steps = self._find(raw, self.FALLBACK_NAMES["steps"])

        if draft is None or accepted is None or steps is None:
            return None

        return SpecStats(
            draft_tokens=int(draft),
            accepted_tokens=int(accepted),
            verification_steps=int(steps),
        )

    @staticmethod
    def _find(raw: dict[str, float], names: list[str]) -> float | None:
        for name in names:
            if name in raw:
                return raw[name]
        return None
