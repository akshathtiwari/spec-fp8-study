"""SGLang Prometheus counter mapping.

SGLang counter names (as of v0.4+):
- sglang:spec_draft_tokens_total
- sglang:spec_accepted_tokens_total
- sglang:spec_verify_steps_total

Counter names may vary. If missing, parse() returns None → τ = null.
"""

from __future__ import annotations

from specfp8.metrics.base import SpecStats


class SglangMetricsScraper:
    """Maps SGLang Prometheus counters to SpecStats."""

    FALLBACK_NAMES = {
        "draft": [
            "sglang:spec_draft_tokens_total",
            "sglang_spec_draft_tokens_total",
            "spec_draft_tokens_total",
            # SpecForge-era naming
            "sglang:num_spec_tokens",
            "num_spec_tokens",
        ],
        "accepted": [
            "sglang:spec_accepted_tokens_total",
            "sglang_spec_accepted_tokens_total",
            "spec_accepted_tokens_total",
            "sglang:num_accepted_tokens",
            "num_accepted_tokens",
        ],
        "steps": [
            "sglang:spec_verify_steps_total",
            "sglang_spec_verify_steps_total",
            "spec_verify_steps_total",
            "sglang:num_spec_steps",
            "num_spec_steps",
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
