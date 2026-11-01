---
id: F041
title: Tested directly on the same sessions, calendar date predicts the regime better than the engine release does (3/3 deficit before 2026-10-31 against 0/3 on it, p=0.050, versus 0.238 for the version split); the confound F040 named as unresolvable is the stronger of the two hypotheses, and if the regime has stopped occurring on this fleet the mechanism cannot be closed by spending more
kind: result
status: established
confidence: low
date: 2026-11-01
evidence:
  runs: [2026-10-31T10-37Z_boot_variance, 2026-10-31T09-11Z_boot_variance,
         2026-10-31T07-30Z_boot_variance, 2026-10-04T20-04Z_boot_variance,
         2026-09-29T06-29Z_boot_variance, 2026-09-23T19-28Z_boot_variance]
  analysis: [analysis/session_regimes.py, analysis/out/tables/session_regimes.md]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

F040 withdrew the engine-release reading of the two regimes and named
calendar date as an entangled confound it could not separate. Tested
directly on the same nine sessions, **date is the better predictor of the
two**:

| predictor | split | one-sided Fisher p |
|---|---|---|
| engine release (0.29.0 vs later) | 3/6 vs 0/3 | 0.238 |
| **date within 0.29.0** (before 10-31 vs on/after) | **3/3 vs 0/3** | **0.050** |
| date, all engines (before 10-31 vs on/after) | 3/5 vs 0/4 | 0.119 |

Holding the engine release fixed at 0.29.0, every session run on or before
4 October was in the deficit regime and every session run on 31 October was
at parity. No session contradicts the split.

This was computed from records already in the repo. It cost no GPU time, and
it was available on 31 October had anyone asked the question in this form.

## Why confidence is `low` despite the p-value

Three reasons, and they matter more than the number.

1. **The cut date is chosen post hoc.** 31 October is the obvious gap in our
   calendar, picked after seeing which sessions were deficient. A threshold
   selected against the data it is then tested on is not a test. The
   p-value is optimistic by an amount this design cannot quantify.
2. **0.050 is the floor, not a strong result.** With 3 against 3,
   1/C(6,3) = 1/20 is the smallest p obtainable. A perfect split scores the
   minimum available and nothing better can be reached at this sample size,
   so "significant at 0.05" here means only "as separated as six sessions
   can be".
3. **The three 31 October sessions are not independent draws of fleet
   state.** They ran within four hours of each other. Whatever a "fleet
   state" is, three samples of it inside one morning plausibly share it, so
   the effective n on the parity side is closer to 1 than to 3.

## What this changes about the plan

The question this study has been trying to close is the mechanism. The plan
for November was to keep running sessions until one lands in the deficit
regime, now with the non-speculative control attached, and to capture it
with utilisation sampling.

**That plan assumes the deficit regime still occurs.** If what changed is
the provider's fleet rather than anything we control, the regime may simply
not recur, and the expected value of further sessions collapses:

| assumed deficit rate | P(at least one deficit in 10 sessions) |
|---|---|
| 3/9, the all-time rate | 98% |
| 3/5, the pre-31-October rate | 100% |
| 1/10, if the rate has dropped | 65% |

Ten 2x2 sessions is about $28.60, which is one month's credit. The decision
this finding forces is whether to spend it on a lottery whose odds we cannot
estimate, and the honest answer is that we cannot estimate them.

## Prediction, recorded before the result

A 2x2 session was launched on 2026-11-01 at 17:52 IST, before this finding
was committed, and its outcome was not known when this was written.

- **If it lands at parity:** the date reading goes to 0/4 after the cut,
  p = 0.029, and the case that the regime has stopped occurring strengthens.
- **If it lands in the deficit regime:** this finding is refuted outright,
  the date reading dies, and we get the first non-speculative control ever
  measured inside a deficit container, which is the measurement the whole
  2x2 was built for.

Either outcome is informative, which is the only defensible reason to have
launched it. The prediction is recorded first because F038 did the same
thing and that is the only reason its successor reads as a test.

## What this does NOT establish

- **That the engine release is irrelevant.** Two weak hypotheses on nine
  sessions. Date scoring better than version does not make version wrong,
  and neither reaches a standard that would survive review.
- **That anything about Modal's fleet changed.** "Date" is not a mechanism.
  It is a label for everything that varies with time and that we do not
  record: host, rack, neighbour tenancy, driver rollout, firmware, cooling.
  We record none of these, so the finding names a gap in instrumentation,
  not a cause.
- **That the regime will not recur.** Three sessions on one morning cannot
  establish that a sometimes-occurring condition has stopped.
- **A mechanism.** Unchanged from F040. Graph mode, backend selection and
  power throttling remain eliminated; the stall reading remains untested.
- **That earlier sessions were wrong.** They reproduce from stored records.

## How to reproduce

```bash
python analysis/session_regimes.py
```

The predictor comparison and the prediction above are regenerated into
`analysis/out/tables/session_regimes.md`; neither is typed by hand.
