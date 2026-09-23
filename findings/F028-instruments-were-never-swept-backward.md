---
id: F028
title: Every checker built in this study was run forward on new questions and never swept backward over settled claims, which is how a headline survived a test designed to catch it
kind: method
status: established
confidence: high
date: 2026-09-24
evidence:
  analysis: [analysis/boot_modes.py, analysis/check_provenance.py, analysis/check_paper.py]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

This study built five instruments: a provenance checker, a paper-number
checker, a cross-session guard, a validity filter, and a bimodality
separation test. **Each was run forward on the question that motivated it
and never pointed back at claims already marked `established`.**

The cost was concrete. `analysis/boot_modes.py` implements a separation
criterion specifically to distinguish two tight modes from one wide spread,
because confusing them is the error F020 made and F021 claimed to correct.
It was written to check whether n-gram speculation is bimodal. **It was never
run on the DFlash data whose bimodality is the paper's §4 headline.** When
finally pointed there it returns *not separable*: separation ratio 0.53x
against a 3.0x threshold, and a BIC comparison of one- versus two-component
fits gives $\Delta$BIC = +0.29, which is indistinguishable from no preference.

## Evidence

| Instrument | Built to answer | Ever run on prior claims? |
|---|---|---|
| `check_provenance.py` | do findings cite real records | partly — found F009/F014/F021 gaps |
| `check_paper.py` | do paper figures trace to records | yes, by construction |
| cross-session guard (`cudagraph_arms.py`) | are arm pairs comparable | no — added after the error it prevents |
| `tokens_from_usage` filter | are token counts valid | **no** — recorded since day one, consumed only on 2026-09-23 |
| `boot_modes.py` separation test | is n-gram bimodal | **no** — never pointed at DFlash until 2026-09-24 |

Two of the five were never swept backward at all, and both concealed real
defects: a 7.26 tok/s record polluting a published mean, and an unsupported
bimodality claim in the paper's central section.

## Why it happened

The mechanism is mundane and probably general. A checker gets written *in
response to* a specific doubt. Once that doubt is resolved the checker feels
spent, and attention returns to the open question. Claims already marked
`established` are treated as settled inputs to the next step rather than as
standing hypotheses that a new instrument might now be able to test.

The asymmetry is the point: **new evidence is routinely applied forward to
open questions and almost never backward to closed ones**, because closed
questions do not present themselves as work.

This study did do it correctly once. When F020 produced a boot-to-boot noise
floor for tau, that floor was carried back to F006 and withdrew a standing
claim. That worked, it was the right instinct, and it was not generalised
into a habit.

## What this does NOT establish

- **That the bimodality claim is false.** The test says the data does not
  *support* two components at n=11, which is not the same as showing one.
  The distinction is exactly the one this study keeps insisting on. A
  larger sample settles it; a failed separation test does not.
- **That the other three instruments are now exhausted.** They were swept
  backward once, on 2026-09-23 and 2026-09-24. Whether they should be re-run
  after every subsequent finding is an open process question.
- **That this is unusual.** No comparable study publishes its internal
  checks, so there is no base rate for how often instruments get pointed
  backward elsewhere.

## Consequence

The practice worth adopting is narrow enough to be actionable:

> **When a new instrument is built, run it over every prior claim it could
> test, not only the claim that motivated building it.**

That is one line, it is cheap, it is offline, and in this study it would
have caught a headline before it reached a PDF.

It also suggests a weakness in how `findings/` is organised. `status:
established` reads as *settled*, when what it means is *supported by the
evidence available when it was written*. A field recording which instruments
a finding has been checked against would make the gap visible instead of
leaving it to memory.

## How to reproduce

```bash
python analysis/boot_modes.py
```

The DFlash default-arm values at concurrency 1 are in `results/sweep.jsonl`;
running the separation test over them is the check that was missing.
