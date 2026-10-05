# v1.3.0 release notes — paste-ready

Tag `v1.3.0` is **already pushed** and points at `59b3727`. Verified: the
retracted capacity claim is absent and F037 is present.

Go to:

    https://github.com/akshathtiwari/spec-fp8-study/releases/new?tag=v1.3.0

It will show "Existing tag". Paste the two blocks below, leave the label on
**Latest**, and publish. Zenodo mints a new version DOI under the same concept
DOI (`10.5281/zenodo.23033168`) within a couple of minutes.

---

## Title

```
v1.3.0 — Capacity reading retracted; power hypothesis measured and refuted
```

## Body

```
Two corrections, the second to a hypothesis this release was built to test.

**The capacity reading is retracted.** v1.2.0 attributed the weak CUDA-graph
deficit on the A10 to memory capacity, claiming it has 24 GiB against the L4's
22.03. Both cards report **22.5 GiB**. The field was in every record and was
never read before the claim was written, and it reached Section 4, F035 and a
public vLLM issue first.

**The replacement hypothesis was power, and it is also wrong.** The L4 is a
72 W part at 2040 MHz, the A10 a 150 W part at 1695 MHz, so throttling under
the faster execution path looked like a good fit. This release adds
GpuSampler, which polls achieved SM clock, power draw and the driver's own
clocks_throttle_reasons across each measurement window, and then ran it
against the claim.

One L4 session, vLLM 0.29.0, concurrency 1:

| arm | tok/s | power-capped | SM clock | power draw |
|---|---|---|---|---|
| default | 56.6 | 84% of samples | 1991 MHz | 67.2 W |
| eager | 78.1 | 95% of samples | 1645 MHz | 69.8 W |

Thermal throttling is 0.0% throughout. The faster arm is capped more, draws
more and clocks lower. Power capping is near-saturated in both arms, so it
cannot distinguish them (F037).

What the measurement shows instead: the slow arm holds a **higher** boost
clock while producing **fewer** tokens. A card clocking up and delivering less
work is idle or stalled between bursts rather than at a ceiling. That is a
third hypothesis, cheap to test with `utilization.gpu`, and F037 states it
without claiming it.

Three mechanisms are now eliminated by evidence rather than left unexamined:
graph mode and backend selection (F032), and power throttling (F037). The
deficit ratio also replicates a third time at 0.725, against 0.487 and 0.768.

Also in this release: F036, where the previous fix for a declared-versus-
measured defect was itself a declared value — budget.log recorded gpu_type
"L4" for a run on an A10, because the selecting environment variable is read
locally and the remote container never sees it. Billing now reads the card
from nvidia-smi.

37 findings, 8 of them retractions. 302 numeric claims, 0 untraceable,
2 disclosed provenance defects.

This is a preprint. It has not been peer reviewed.
```

---

## Why this one matters more than the others

Until you publish it, the concept DOI resolves to v1.2.0, which still contains
the capacity claim you retracted four days ago. That is the only part of this
project where a known-false statement is sitting behind a citable identifier.
