"""GSM8K workload — grade-school maths with numeric ground truth.

Draws from the same 256 verified problems the quality measurement uses, so a
Phase 2 accuracy number is directly comparable to findings/F016 rather than
being scored against a different set.

Sampling deliberately draws **without replacement** up to the size of the
pool. Repeating a prompt inside one measurement would let prefix caching serve
the repeat from cache, inflating throughput and understating latency — the
opposite of what a serving benchmark should report. Requesting more prompts
than the pool holds is an error rather than a silent wrap-around.
"""

from __future__ import annotations

import random

from specfp8.correctness import extract_gsm8k_answer
from specfp8.quality import QUALITY_MAX_TOKENS, load_quality_prompts


class Gsm8kWorkload:
    name = "gsm8k"

    def __init__(self) -> None:
        # limit=None: Phase 2 draws from the full pool so repeats can take
        # disjoint prompt sets. The quality measurement stays pinned to the
        # first 256 (see specfp8.quality.QUALITY_N).
        self._prompts, self._answers = load_quality_prompts(limit=None)
        self._order: list[int] = []

    @property
    def size(self) -> int:
        return len(self._prompts)

    def prompts(self, n: int, seed: int) -> list[str]:
        if n > len(self._prompts):
            raise ValueError(
                f"gsm8k has {len(self._prompts)} prompts, {n} requested. "
                "Repeating prompts would let prefix caching serve them from "
                "cache and inflate throughput; add problems instead "
                "(python -m specfp8.workloads.fetch_gsm8k --n N)."
            )
        rng = random.Random(seed)
        self._order = rng.sample(range(len(self._prompts)), n)
        return [self._prompts[i] for i in self._order]

    def sampling(self) -> dict:
        return {
            "temperature": 0,
            "max_tokens": QUALITY_MAX_TOKENS,
            # Qwen3 reasons in plain text with thinking disabled; see
            # findings/F014 for why the budget is not the gate's 256.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def validate(self, index: int, output: str) -> bool | None:
        if index >= len(self._order):
            return None
        want = self._answers[self._order[index]]
        got = extract_gsm8k_answer(output)
        return None if got is None else abs(got - want) < 0.01
