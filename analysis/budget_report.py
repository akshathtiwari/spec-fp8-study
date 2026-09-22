"""What this study cost, by phase, from results/budget.log.

A reproducibility claim includes the price. "Re-derivable by a third party
on rented hardware" is only actionable if the reader knows whether that
means five dollars or five hundred, and the number should come from the
recorded log rather than from memory.

Reads `session_gpu_s` and ignores the stored `cumulative_gpu_s`: entries
written before that field was fixed restarted it on every invocation, and
`results/` is append-only, so the stored totals cannot be corrected in
place. Summing the per-entry figures repairs the history instead.

The result is a **floor**, twice over. Container start and image pull sit
outside the timed region, and four GPU entrypoints were unbilled until
2026-09-23 -- the boot-variance and concurrency runs behind F020 and F021
are among them. Both gaps are reported rather than silently absorbed.

    python analysis/budget_report.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "budget.log"

#: Modal L4 on-demand, USD per GPU-second, as quoted 2026-09.
L4_USD_PER_S = 0.000222

#: Entries before this date predate the `billed` decorator, so bespoke GPU
#: probes in that window are absent from the log entirely.
BILLING_COMPLETE_FROM = "2026-09-23"


def main() -> int:
    if not LOG.exists():
        print(f"no budget log at {LOG}", file=sys.stderr)
        return 1

    by_phase: dict[str, list[float]] = defaultdict(list)
    first = last = None
    malformed = 0

    for line in LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            secs = float(e.get("session_gpu_s", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            malformed += 1
            continue
        by_phase[e.get("phase", "?")].append(secs)
        ts = e.get("ts", "")
        first = min(first, ts) if first else ts
        last = max(last, ts) if last else ts

    total = sum(sum(v) for v in by_phase.values())

    print(f"budget from {LOG.relative_to(ROOT)}")
    if first and last:
        print(f"window: {first[:16]} -> {last[:16]}")
    print()
    print(f"{'phase':<16}{'runs':>6}{'GPU s':>12}{'hours':>9}{'USD':>9}")
    print("-" * 52)
    for phase in sorted(by_phase, key=lambda p: -sum(by_phase[p])):
        secs = sum(by_phase[phase])
        print(f"{phase:<16}{len(by_phase[phase]):>6}{secs:>12,.0f}"
              f"{secs / 3600:>9.2f}{secs * L4_USD_PER_S:>9.2f}")
    print("-" * 52)
    n = sum(len(v) for v in by_phase.values())
    print(f"{'TOTAL':<16}{n:>6}{total:>12,.0f}{total / 3600:>9.2f}"
          f"{total * L4_USD_PER_S:>9.2f}")

    print()
    print("This is a floor, not the billed amount:")
    print("  - container start and image pull are outside the timed region")
    print(f"  - four GPU entrypoints were unbilled before "
          f"{BILLING_COMPLETE_FROM}; the boot-variance and concurrency runs")
    print("    behind F020/F021 are not in this total")
    if malformed:
        print(f"  - {malformed} malformed log line(s) skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
