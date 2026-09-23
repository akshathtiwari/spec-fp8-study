---
id: F025
title: Every inter-token latency recorded before 2026-09-23 is corrupt, so every goodput figure in the study is 0.0 and the SLO axis was never measured
kind: retraction
status: retracted
confidence: high
date: 2026-09-23
evidence:
  analysis: [analysis/out/tables/goodput.md]
  code_sha: pending
supersedes: []
superseded_by: []
---

## What was believed

That the harness recorded per-request inter-token latency, and that
`analysis/goodput.py` therefore computed goodput under a declared SLO
across the concurrency sweep. `requirements.md` §9 names that as the
study's novelty against prior art: nobody reports goodput under an SLO,
which is where lost KV capacity actually binds.

## What is true

**Every `itl_ms` array recorded before 2026-09-23 is numerically
meaningless after its first element**, and every goodput figure the study
has produced is `0.0`.

`specfp8/client.py` accumulated latencies into `token_times` and then read
`token_times[-1]` as the previous *timestamp*:

```python
token_times.append((now - (token_times[-1] if token_times else first_token_time)) * 1000)
```

After the first append, `token_times[-1]` is a latency in **milliseconds**,
not a monotonic timestamp in **seconds**. The expression subtracts one from
the other and feeds the result back in, so the sequence alternates sign and
grows by roughly 1000x per token. Against a synthetic 20 ms/token stream:

```
buggy : 20.0, 80040.0, -79939940.0, 79940040080.0, ...   max 7.99e+31
fixed : 20.0, 20.0, 20.0, 20.0, ...
```

Real records reach `±Infinity` within about 100 tokens. p95 ITL is
therefore `Infinity`, no request ever meets an ITL target, and
`goodput.md` is a table of zeros — which is exactly what it has always
shown, unread, since the day it was generated.

## How it went unnoticed

Three reasons, all worth recording.

1. **The first value is correct.** `itl_ms[0]` is a true latency, so a
   spot-check of the field's shape looks fine.
2. **Nothing consumed it until now.** The paper's framing moved to the
   three confounds, and every headline result — throughput, tau, accuracy,
   KV capacity — is computed from wall clock, engine counters, or stored
   text. None touches `itl_ms`.
3. **The failure output is a plausible number.** `0.0` goodput reads as
   "the SLO was not met", which is a legitimate experimental outcome, not
   as "the metric is broken". A NaN or an exception would have been caught
   immediately; a confident zero was not.

## What is NOT affected

Checked field by field, because the instinct to assume broad damage is as
wrong as the instinct to assume none:

- **Throughput (tok/s)** — computed in `sweep.py` from wall clock and token
  counts. Unaffected. Every number in F020, F021, F023, F024 stands.
- **tau** — read from the engine's own counters over HTTP. Unaffected.
- **TTFT** — computed from `first_token_time`, not from `token_times`.
  Unaffected.
- **e2e latency, token counts, accuracy, KV capacity, backend selection** —
  all independent of this path.

The damage is confined to `itl_ms` and to everything derived from it, which
is goodput and the SLO sweep.

## What this costs

The SLO/goodput axis — R7, and the study's stated novelty against prior art
— **has no data at all**. Not weak data: none. The stored arrays cannot be
repaired offline because only the (corrupt) latencies were written, never
the raw arrival timestamps, so there is nothing to recompute from.

Recovering it requires re-running the grid with the fixed client. That is
about 1.9 GPU-hours, roughly $1.50, and would also yield a second
independent set of boots for the bimodality statistics.

## What this does NOT establish

- **That the goodput *analysis* is wrong.** `analysis/goodput.py` appears
  correct; it was fed corrupt input. It has never been exercised on valid
  data, so it is unverified rather than known-good.
- **That other stored arrays are fine.** Only `itl_ms` was inspected
  element-by-element. Fields that are scalars are harder to corrupt this
  way, but "harder" is not "checked".

## Lesson

The defect is a variable used for two purposes — an accumulator and a
cursor — and the type system did not object because both are floats. The
guard that would have caught it is not a checker but a **range assertion**:
an inter-token latency above, say, a second on a local server is not a slow
token, it is a bug. No such assertion existed anywhere in the harness.

This is the study's thesis in its purest form. `goodput.md` has been
committed, regenerated and shipped repeatedly as a table of zeros, and the
zeros were never read because nothing depended on them. **An unread number
is an unchecked number**, regardless of how carefully the pipeline that
produced it was built.

## How to reproduce

```bash
python -c "
times=[100.0+0.020*i for i in range(12)]
old=[]; first=None
for now in times:
    if first is None: first=now
    else: old.append((now-(old[-1] if old else first))*1000)
print(max(abs(v) for v in old))"
```
