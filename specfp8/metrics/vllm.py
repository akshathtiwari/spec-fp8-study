"""vLLM Prometheus counter mapping.

Counter names verified against vLLM 0.29.0 by scraping /metrics from a
live server (cloud/modal_probe.py --engine diagnose):

    vllm:spec_decode_num_draft_tokens_total      total draft tokens proposed
    vllm:spec_decode_num_accepted_tokens_total   accepted *draft* tokens
    vllm:spec_decode_num_drafts_total            drafts == verification steps

The earlier `vllm:num_spec_tokens_total` family does not exist in 0.29 and
silently yielded τ = null. Older spellings are kept as fallbacks, but the
0.29 names are tried first.

`num_accepted_tokens_total` excludes the bonus token: an observed run had
650 drafts at 3 speculative tokens each (1950 draft tokens) and only 185
accepted, which is below the 650 floor that including the bonus would
imply. SpecStats.tau accounts for this — see metrics/base.py.
"""

from __future__ import annotations

from specfp8.metrics.base import SpecStats


class VllmMetricsScraper:
    """Maps vLLM Prometheus counters to SpecStats."""

    # Primary counter names (verified on vLLM 0.29.0)
    DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
    ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens_total"
    VERIFICATION_STEPS = "vllm:spec_decode_num_drafts_total"

    # Tried in order; earlier entries win.
    FALLBACK_NAMES = {
        "draft": [
            "vllm:spec_decode_num_draft_tokens_total",
            "vllm:num_spec_tokens_total",
            "vllm_num_spec_tokens_total",
            "num_spec_tokens_total",
        ],
        "accepted": [
            "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:num_accepted_tokens_total",
            "vllm_num_accepted_tokens_total",
            "num_accepted_tokens_total",
        ],
        "steps": [
            "vllm:spec_decode_num_drafts_total",
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
