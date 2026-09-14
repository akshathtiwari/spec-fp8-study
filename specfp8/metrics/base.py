"""Prometheus metrics scraping and speculation stats.

Scrape /metrics before and after each RunCell. Delta-based, so server
reuse across cells is safe. If a mechanism exposes no counters, τ is
recorded as null — never estimated.
"""

from __future__ import annotations

import re
from typing import Protocol

import httpx
from pydantic import BaseModel


class SpecStats(BaseModel):
    """Speculative decoding statistics from Prometheus counters."""

    draft_tokens: int
    accepted_tokens: int
    verification_steps: int

    @property
    def tau(self) -> float | None:
        """Mean accepted tokens per verification step (τ)."""
        if self.verification_steps == 0:
            return None
        return self.accepted_tokens / self.verification_steps


class MetricsScraper(Protocol):
    """Interface for engine-specific metrics scraping."""

    def parse(self, raw: dict[str, float]) -> SpecStats | None:
        """Parse engine-specific counter names into SpecStats.
        Returns None if the required counters are not present."""
        ...


def scrape_prometheus(url: str) -> dict[str, float]:
    """Fetch and parse Prometheus text format from /metrics.

    Returns a dict of metric_name -> value (counters only, no histograms).
    """
    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
    except (httpx.HTTPError, httpx.ConnectError):
        return {}

    metrics: dict[str, float] = {}
    for line in resp.text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Parse: metric_name{labels} value
        # or: metric_name value
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\b(?:\{[^}]*\})?\s+(\S+)$", line)
        if match:
            name = match.group(1)
            try:
                value = float(match.group(2))
                metrics[name] = value
            except ValueError:
                continue

    return metrics


def delta(before: SpecStats, after: SpecStats) -> SpecStats:
    """Compute delta between two snapshots."""
    return SpecStats(
        draft_tokens=after.draft_tokens - before.draft_tokens,
        accepted_tokens=after.accepted_tokens - before.accepted_tokens,
        verification_steps=after.verification_steps - before.verification_steps,
    )


def scrape_spec_stats(
    base_url: str, scraper: MetricsScraper
) -> SpecStats | None:
    """Scrape the /metrics endpoint and parse into SpecStats."""
    raw = scrape_prometheus(f"{base_url}/metrics")
    if not raw:
        return None
    return scraper.parse(raw)
