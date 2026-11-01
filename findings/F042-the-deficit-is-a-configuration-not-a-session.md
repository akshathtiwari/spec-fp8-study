---
id: F042
title: Every deficit session shares one configuration -- L4, NVIDIA driver 580.95.05, vLLM 0.29.0, DFlash -- and every session differing from it in exactly one factor is at parity; the host driver was recorded in every record all along, the "date" in F041 is a driver rollout seen through time, and the regime is better described as a property of a configuration than of a session
kind: result
status: established
confidence: medium
date: 2026-11-01
evidence:
  runs: [2026-09-23T19-28Z_boot_variance, 2026-09-29T06-29Z_boot_variance,
         2026-10-04T20-04Z_boot_variance, 2026-09-29T04-48Z_boot_variance,
         2026-09-29T17-16Z_boot_variance, 2026-09-23T07-20Z_boot_variance,
         2026-09-23T11-08Z_boot_variance, 2026-09-30T03-16Z_boot_variance,
         2026-10-31T07-30Z_boot_variance, 2026-10-31T09-11Z_boot_variance,
         2026-10-31T10-37Z_boot_variance, 2026-11-01T12-30Z_boot_variance]
  analysis: [analysis/session_regimes.py, analysis/out/tables/session_regimes.md]
  code_sha: pending
supersedes: [F041]
superseded_by: []
---

## Claim

Every session that measured both arms, on every card and mechanism, with the
host driver each record stores under `env.driver`:

| gpu | driver | engine | mechanism | session | ratio | regime |
|---|---|---|---|---|---|---|
| A10 | 580.95.05 | 0.29.0 | dflash | 09-30 | 0.959 | parity |
| **L4** | **580.95.05** | **0.29.0** | **dflash** | 09-23 | **0.487** | **deficit** |
| **L4** | **580.95.05** | **0.29.0** | **dflash** | 09-29 a | **0.768** | **deficit** |
| **L4** | **580.95.05** | **0.29.0** | **dflash** | 10-04 | **0.724** | **deficit** |
| L4 | 580.95.05 | 0.29.0 | ngram | 09-23 07:20 | 1.000 | parity |
| L4 | 580.95.05 | 0.29.0 | ngram | 09-23 11:08 | 1.025 | parity |
| L4 | 580.95.05 | 0.30.0 | dflash | 09-29 b | 1.014 | parity |
| L4 | 580.95.05 | 0.30.0 | dflash | 09-29 c | 1.017 | parity |
| L4 | 610.57.04 | 0.29.0 | dflash | 10-31 a/b/c, 11-01 | 1.003--1.015 | parity |
| L4 | 610.57.04 | 0.29.0 | none | 11-01 | 1.034 | parity |
| L4 | 610.57.04 | 0.32.0 | dflash | 10-31 d | 1.039 | parity |

**Every deficit session is the same four-way configuration: L4, driver
580.95.05, vLLM 0.29.0, DFlash.** That cell holds three paired sessions and
all three are deficient (a fourth, 10-04 b, ran no eager arm and carries the
deficit signature by level and clock, F040).

Every session that differs from that cell in **exactly one** factor is at
parity:

| factor changed | to | sessions | deficient |
|---|---|---|---|
| engine | 0.30.0 | 2 | 0 |
| driver | 610.57.04 | 4 | 0 |
| GPU | A10 | 1 | 0 |
| mechanism | n-gram | 2 | 0 |

> **Corrected by F043 (2026-11-01).** The n-gram row is not a one-factor
> neighbour. Every n-gram log shows the engine falling back to the V1 model
> runner and disabling async scheduling, neither of which happens for DFlash.
> That row changed three factors, so its parity cannot localise the deficit
> to DFlash. The engine, driver and GPU rows are confirmed one-factor changes
> by the same logs. The DFlash entry in "Within one driver, each original
> reading holds" below is withdrawn for the same reason.

That is the structure of a one-factor-at-a-time factorial design. Nobody
designed it; it was assembled by accident across six weeks of sessions, and
it was sitting in the records the whole time.

## How this was missed, and by whom

F041 listed "driver rollout" among the things this study does not record.
**That was false.** `EnvInfo` has carried a `driver` field since the first
session, and every record in `results/probes.jsonl` stores it. The
comparison that concluded otherwise queried `driver_version`, `cuda_version`
and `torch_version`, got nothing back, and read the empty result as absence.

This is the third time the same error has reached a finding. F035 claimed
the A10 had more memory than the L4 when `env.vram_gb` showed both at 22.5
GiB. F031 found `engine_version` read and printed but never stored. Here, a
wrong key returned silence and silence was read as "not recorded". In each
case the evidence was in every record and the claim was written without
reading it. The lesson F031 drew -- a field that is checked must be checked
under the name it is actually stored under -- did not transfer.

## What this changes

### F041 is superseded

"Date predicts the regime" was a driver rollout seen through time. Within
these records driver and date are perfectly collinear: every session on or
before 4 October ran 580.95.05, every session on or after 31 October ran
610.57.04. **No test on this data can separate them**, and F041's p-values
apply to the driver partition exactly as they did to the date one, because
it is the same partition.

The driver is preferred anyway, and on grounds that are not statistical. A
date cannot cause anything; it is a label for everything that changed with
it. A driver can: CUDA graphs are captured, instantiated and replayed
through the driver API, so a driver revision changing graph execution is a
mechanism rather than a correlation with the calendar. It is also the only
recorded variable that changed between the two periods (CUDA runtime 13.0
and torch 2.13.0+cu130 are identical throughout).

### F040's three withdrawals were made across drivers

F040 withdrew three readings: that the deficit is an engine-release effect
(F034), that it is L4-specific (F035), and that it localises to draft-model
speculation (F018). All three withdrawals compared sessions on driver
580.95.05 against sessions on 610.57.04 without knowing it.

Re-read within one driver, each original reading holds:

| original reading | within driver 580.95.05 |
|---|---|
| version: 0.29.0 deficient, 0.30.0 not (F034) | 3/3 against 0/2 |
| card: L4 deficient, A10 not (F035) | 3/3 against 0/1 |
| mechanism: DFlash deficient, n-gram not (F018) | 3/3 against 0/2 |

F040 corrected an over-confident claim with an over-confident withdrawal,
because it changed a variable it did not know it was changing. **This
finding does not reinstate those three readings as established.** It
reinstates them as conditional on a driver, with the sample sizes in the
table, which are small.

### "Session-scoped" is probably the wrong description

F040's central reframe was that the regime is a property of the container
session, drawn at random. With the driver in hand, the regime is better
described as a property of a configuration: given (L4, 580.95.05, 0.29.0,
DFlash) the deficit appeared 3 times in 3; outside it, 0 times in 9.
"Session-scoped" was what a configuration effect looks like when one of its
factors changes over time and is not being watched.

What F040 established about the regime's *structure* still stands: within
the deficit configuration the magnitude varies from session to session
(ratio 0.487--0.768) and boots disperse widely (CV 3.73--13.92%), the
dispersion and the deficit partition sessions identically, and power
throttling is refuted. Those are measurements, and none of them depended on
the reading this finding revises.

### The spending plan changes

F041 framed further Modal sessions as a lottery for a deficit session whose
odds could not be estimated. **Under this finding there is no lottery.**
Modal's L4 fleet now runs 610.57.04, and the prediction is that every
(L4, 610.57.04, 0.29.0, DFlash) session will be at parity; 4 of 4 have been.
Further sessions on the current fleet are not expected to reproduce the
deficit at any price, and running them to catch one would spend money
testing a prediction this finding already makes.

## The experiment this calls for

Every result above is observational: the driver changed on the provider's
side, and the study watched. The decisive test is an **intervention**:

1. Rent an L4 on a host where the driver can be installed, rather than one
   supplied by the platform (a cloud VM with an L4 attached, as opposed to
   a serverless container).
2. Install 580.95.05. Run the 2x2 on vLLM 0.29.0. **Prediction: dflash
   deficient, none at parity.**
3. On the same VM and the same physical card, install 610.57.04. Run it
   again. **Prediction: both at parity.**

Same card, same host, same software, one variable changed by hand. It
answers three questions at once: whether the driver causes the deficit,
whether the deficit needs speculation (the question the 2x2 was built for,
which Modal can no longer answer because its fleet no longer produces the
regime), and -- because the card is held fixed -- whether the effect is a
property of the card or of its driver, which is the session-versus-card
confound no session on Modal could break. It needs an account on a provider
that allows driver installation, which this study does not currently have.

## What this does NOT establish

- **Causation.** Every factor varied by accident, not by design. The
  driver changed on the provider's side alongside whatever else rolled out
  with it -- host kernel, firmware, container runtime -- none of which is
  recorded. "The driver" means "the only recorded variable that changed",
  not "the variable that was shown to cause it".
- **That driver and date are distinguishable.** They are perfectly
  collinear in these records. The preference for the driver is mechanistic,
  not statistical.
- **That the conjunction is complete or minimal.** Four factors fit the
  data. The deficit could need a fifth that never varied (the model, the
  attention backend, concurrency 1), or not need one of the four if a cell
  we never measured would also be deficient. 0.30.0 on 610.57.04 and A10 on
  610.57.04 were never run.
- **That the one-factor neighbours are well sampled.** 2, 4, 1 and 2
  sessions. The A10 cell is a single session, and the n-gram sessions ran
  4 and 8 boots against the 12 used elsewhere.
- **A p-value for the conjunction.** The three deficit sessions are the
  three sessions in one particular cell, and the chance of that under
  random assignment among the ten paired DFlash L4 sessions is 1/C(10,3) =
  1/120. That number is not reported as a test, because the cell was
  identified after searching recorded variables for one that separated the
  data. Engine release, date and driver have now each been tried. The
  one-factor neighbour structure is the argument, not the arithmetic.
- **What it is about driver 580.95.05 and vLLM 0.29.0 together.** Neither
  is sufficient alone: 0.30.0 on 580.95.05 is at parity, and so is 0.29.0
  on 610.57.04. Something in 0.29.0's DFlash path interacts with something
  in that driver, and either a later engine or a later driver removes it.
  What it is remains unidentified.

## How to reproduce

```bash
python analysis/session_regimes.py
```

The configuration table and the one-factor neighbours are regenerated into
`analysis/out/tables/session_regimes.md` from `results/probes.jsonl`;
neither is typed by hand.
