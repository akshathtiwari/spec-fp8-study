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

#: Modal on-demand GPU rates, USD per hour, as published 2026-09.
#: Entries written before 2026-09-30 carry no `gpu_type` and are priced as
#: L4, which is what they ran on.
GPU_USD_PER_HOUR = {
    "T4": 0.59, "L4": 0.80, "A10G": 1.10, "L40S": 1.95,
    "A100-40GB": 2.10, "A100-80GB": 2.50, "RTX PRO 6000": 3.03,
    "H100": 3.95, "H200": 4.54, "B200": 6.25, "B300": 7.10,
}
DEFAULT_GPU = "L4"

#: budget.log now stores the name the driver reports ("NVIDIA A10"), while the
#: rate table is keyed by Modal's flag ("A10G"). Mapping one to the other here
#: keeps the log machine-truthful and the pricing correct. Unknown cards fall
#: back to L4 and are counted, so a new card shows up as an undercount rather
#: than vanishing.
DRIVER_TO_RATE_KEY = {
    "NVIDIA A10": "A10G", "NVIDIA A10G": "A10G", "NVIDIA L4": "L4",
    "NVIDIA L40S": "L40S", "NVIDIA T4": "T4", "NVIDIA H100": "H100",
    "NVIDIA H200": "H200", "NVIDIA A100-SXM4-40GB": "A100-40GB",
    "NVIDIA A100-SXM4-80GB": "A100-80GB",
}


#: Cards seen in budget.log that no rate could be found for. Collected rather
#: than silently defaulted: falling back to the L4 rate undercounts a B300 run
#: by 9x, and an accounting error that reports itself as a clean total is the
#: failure this whole file exists to avoid.
UNPRICED: set[str] = set()


def _rate_key(recorded: str | None) -> str:
    """Rate-table key for whatever budget.log recorded.

    Entries before 2026-09-30 carry no gpu_type and ran on an L4. Entries
    after it carry the name the driver reports, which is not Modal's flag
    name -- "NVIDIA A10" against "A10G" -- hence the mapping.
    """
    if not recorded:
        return DEFAULT_GPU
    if recorded in GPU_USD_PER_HOUR:
        return recorded
    key = DRIVER_TO_RATE_KEY.get(recorded)
    if key is None:
        UNPRICED.add(recorded)
        return DEFAULT_GPU
    return key
L4_USD_PER_S = GPU_USD_PER_HOUR["L4"] / 3600.0

#: Measured 2026-09-30 against `modal billing report`: this ledger's floor was
#: $16.78 where Modal billed $22.54, so wall-clock inside the container misses
#: about a quarter of the bill. Container start, image pull, CPU and memory all
#: sit outside the timed region. Reported rather than silently corrected: the
#: ratio depends on how much container churn a phase does, and inflating the
#: floor by a constant would make it look like a measurement.
FLOOR_CHECKED_ON = "2026-09-30"
FLOOR_OBSERVED = 16.78
BILLED_OBSERVED = 22.54
OBSERVED_FLOOR_RATIO = BILLED_OBSERVED / FLOOR_OBSERVED

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
            rate = GPU_USD_PER_HOUR[_rate_key(e.get("gpu_type"))] / 3600.0
        except (json.JSONDecodeError, TypeError, ValueError):
            malformed += 1
            continue
        by_phase[e.get("phase", "?")].append((secs, rate))
        ts = e.get("ts", "")
        first = min(first, ts) if first else ts
        last = max(last, ts) if last else ts

    total = sum(sum(x for x, _ in v) for v in by_phase.values())
    total_usd = sum(sum(x * r for x, r in v) for v in by_phase.values())

    print(f"budget from {LOG.relative_to(ROOT)}")
    if first and last:
        print(f"window: {first[:16]} -> {last[:16]}")
    print()
    print(f"{'phase':<16}{'runs':>6}{'GPU s':>12}{'hours':>9}{'USD':>9}")
    print("-" * 52)
    for phase in sorted(by_phase, key=lambda p: -sum(x for x, _ in by_phase[p])):
        secs = sum(x for x, _ in by_phase[phase])
        usd = sum(x * r for x, r in by_phase[phase])
        print(f"{phase:<16}{len(by_phase[phase]):>6}{secs:>12,.0f}"
              f"{secs / 3600:>9.2f}{usd:>9.2f}")
    print("-" * 52)
    n = sum(len(v) for v in by_phase.values())
    print(f"{'TOTAL':<16}{n:>6}{total:>12,.0f}{total / 3600:>9.2f}"
          f"{total_usd:>9.2f}")

    print()
    print("This is a floor, not the billed amount:")
    print("  - container start and image pull are outside the timed region")
    print("  - CPU and memory are billed alongside GPU and are not counted here")
    print(f"  - measured {FLOOR_CHECKED_ON} against `modal billing report`: this "
          f"floor read ${FLOOR_OBSERVED:.2f} where Modal billed "
          f"${BILLED_OBSERVED:.2f},")
    print(f"    so the real bill runs about {OBSERVED_FLOOR_RATIO:.2f}x this figure")
    print(f"  - four GPU entrypoints were unbilled before "
          f"{BILLING_COMPLETE_FROM}; the boot-variance and concurrency runs")
    print("    behind F020/F021 are not in this total")
    if UNPRICED:
        print()
        print("  !! UNPRICED GPU(S) -- counted at the L4 rate, so this total "
              "is WRONG:")
        for name in sorted(UNPRICED):
            print(f"     {name!r} has no entry in GPU_USD_PER_HOUR or "
                  f"DRIVER_TO_RATE_KEY")
        print("     Add it before trusting any figure above.")
    if malformed:
        print(f"  - {malformed} malformed log line(s) skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
