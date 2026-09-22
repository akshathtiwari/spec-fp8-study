"""MT-Bench workload — open-ended chat, no ground truth.

Included because acceptance and throughput depend on what is being generated,
and maths reasoning is not representative of chat serving. It has no
validator: `validate` returns None, and Phase 2 reports task success as
unavailable rather than inventing a criterion.

The pool is small (16 prompts, the same ones the correctness gate uses), so
this workload caps concurrency sweeps well below gsm8k. Sampling without
replacement is enforced for the same prefix-caching reason.
"""

from __future__ import annotations

import random

from specfp8.correctness import load_correctness_prompts


class MtBenchWorkload:
    name = "mtbench"

    def __init__(self) -> None:
        mt, _ = load_correctness_prompts()
        self._prompts = [p["prompt"] for p in mt]

    @property
    def size(self) -> int:
        return len(self._prompts)

    def prompts(self, n: int, seed: int) -> list[str]:
        if n > len(self._prompts):
            raise ValueError(
                f"mtbench has only {len(self._prompts)} prompts, {n} "
                "requested. Repeating them would let prefix caching inflate "
                "throughput; use gsm8k for larger request counts."
            )
        rng = random.Random(seed)
        return [self._prompts[i]
                for i in rng.sample(range(len(self._prompts)), n)]

    def sampling(self) -> dict:
        return {
            "temperature": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def validate(self, index: int, output: str) -> bool | None:
        return None
