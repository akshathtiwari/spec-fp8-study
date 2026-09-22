"""Workload protocol.

A workload supplies prompts, the sampling settings they should be run with,
and optionally a validator that says whether an output is correct. Phase 2
measures serving behaviour, so the validator is used to report task success
alongside throughput rather than to gate anything.
"""

from __future__ import annotations

from typing import Protocol


class Workload(Protocol):
    """One source of prompts with its own sampling settings."""

    name: str

    def prompts(self, n: int, seed: int) -> list[str]:
        """Return `n` prompts, deterministically for a given seed."""
        ...

    def sampling(self) -> dict:
        """Sampling parameters this workload should be served with."""
        ...

    def validate(self, index: int, output: str) -> bool | None:
        """Was this output correct? None when the workload has no ground truth.

        `index` is the prompt's position in the list returned by `prompts`,
        not a dataset index, so a workload that samples must keep its own
        mapping back to references.
        """
        ...
