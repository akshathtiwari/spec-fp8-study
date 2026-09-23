---
id: F022
title: Every substantive error in this study was a comparison holding a third variable fixed by assumption rather than by measurement
kind: method
status: established
confidence: high
date: 2026-09-23
evidence:
  runs: []
  cells: []
  analysis: [analysis/check_provenance.py]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

This study has produced six retractions and several near-misses. They look
like a varied collection of bugs — a bad regex, a truncated budget, a leaked
process, a wrong ground truth. They are not varied. **Every one is the same
error**: two numbers were compared while a third thing was assumed fixed,
and the third thing was not fixed.

The assumption was never stated, because an assumption you would have
written down is one you would have checked.

## Evidence

Every correction in `findings/`, classified by what was assumed constant:

| # | Compared | Assumed fixed | Actually varied | Cost if unreported |
|---|---|---|---|---|
| F007 | FP8 vs BF16 outputs | the prompt set | different prompts per arm | a false equivalence gate |
| F011 | backends | that the parsed string was a backend | it was a comms setting (`AG_RS`) | a fabricated axis |
| F012 | configurations | that the GPU was free | 20GB held by orphans | 7 cells called incompatible |
| F013 | answers vs ground truth | that ground truth was right | paraphrases kept originals' answers | accuracy off by ~14 pts |
| F015 | extracted vs reference | that extraction worked | 58.6% of correct answers dropped | accuracy off by ~59 pts |
| F018/F019 | attention backends | one backend per server | target and draft select independently | a wrong public correction |
| F006 | tau across backends | that 2.25% was signal | the noise floor is 1.32% | a claimed effect at 1.7x noise |
| F020/F021 | speculative throughput | that one boot represents the config | bimodal, 1.64x between boots | any speedup from 1.16x to 2.71x |
| perf_tables | repeats of a cell | that repeats shared a boot | 4 groups spanned hours | cross-boot spread called prompt variance |
| budget.log | cumulative spend | that "cumulative" accumulated | it restarted per session | spend understated ~4.5x |
| cell_id | configurations | that the schema could grow | any new field rehashed all 30 | every stored result orphaned |

Eleven instances, one shape. Three more arrived the day this was written;
all are below.

### The twelfth, found the same day this was written

"What this does NOT establish" below predicted that a twelfth instance
existed and was not on the list. It was found within the hour:

| # | Compared | Assumed fixed | Actually varied | Cost if unreported |
|---|---|---|---|---|
| 12 | findings vs their evidence | that every probe recorded its measurements | four GPU entrypoints wrote nothing to `results/` | F009, F014 and F021 rest on terminal output |

The shape holds exactly. The claim being checked was "this finding is
supported"; the thing assumed fixed was "the harness wrote the data down";
and it had not, for the bespoke probes. F021 — the study's strongest result
and the paper's central confound — had `cells: []` and `analysis: []`.

Worth noting how it was found, because it was not by inspection.
`check_provenance.py` validated every citation a finding *made* and was
silent about citations a finding *omitted*. An absence is not a malformed
value, so nothing looked wrong. The check that catches it had to be written
as a separate rule, and the same asymmetry — verifying what is present,
never noticing what is missing — is worth suspecting elsewhere.

That the prediction was confirmed this quickly is weak evidence that the
rate of undiscovered instances is higher than the eleven above suggest, not
that the list is now complete at twelve.

### The thirteenth, in the act of diagnosing a failure

A boot failed with `oom` on the perf_v2 grid. Diagnosing it, I matched the
failure record against prior results on
`(mechanism, weight_precision, kv_cache_dtype)` and concluded "this exact
cell booted fine before, so the OOM is a host condition", then raised a
memory threshold to fix it.

Every step was wrong, in the same shape:

| # | Compared | Assumed fixed | Actually varied | Cost if unreported |
|---|---|---|---|---|
| 13 | failing cell vs prior results | that the two shared a configuration | `enforce_eager`, the grid's own axis | a real result (F023) filed as a host fault, and a threshold changed for nothing |

The matched tuple omitted `enforce_eager`. The failing cell was
`4ae0a860802f247c` (eager on); the cell with 12 prior measurements was
`5824814b68901229` (eager off). Different cells, different `cell_id`, and
the field that distinguishes them is the one the grid exists to vary.

Two things make this the sharpest instance so far:

1. It is the **same bug** fixed in `analysis/perf_tables.py` earlier the
   same day, where a hand-listed grouping tuple would have averaged the two
   arms together. The fix there was to key on `cell_id` because "a
   hand-listed key fails the same way every time: add an axis, forget the
   key". Hours later I hand-wrote the tuple again, in a throwaway
   diagnostic script rather than in committed code — which is precisely
   where the discipline lapses.
2. The wrong diagnosis **produced action**: a real finding was nearly
   recorded as a host fault, and `MIN_FREE_GPU_MIB` was raised to fix
   something it had not caused. The failing boot had 22561 MiB free on a
   22.03 GiB card. The comment at that constant now says so.

What caught it was not care. It was `boot_failures.jsonl`, added an hour
earlier, recording `gpu_free_mib_before_boot` alongside the verbatim error.
A clean card in the record contradicted the residual-memory story outright.
The lesson is the recurring one: the defence that works is a record that
can contradict you, not an intention to be careful.

### The fourteenth, and the pattern within the pattern

Two hours after instance 13, the same axis again. An early cross-session
"consistency check" read tonight's first cell against the prior grid's
`none|bf16|fp8_e4m3` rows and flagged a 1.15x gap at c=64 as a possible
instability in F020's control. The cell was `28971d07`,
`enforce_eager=True`; the prior rows were `enforce_eager=False`.

| # | Compared | Assumed fixed | Actually varied | Cost if unreported |
|---|---|---|---|---|
| 14 | tonight's cell 1 vs prior grid | that both were the same arm | `enforce_eager` (again) | a false doubt cast on the control that isolates F021 |

The interesting part is not the error, it is its **distribution**. Instances
13 and 14, plus the `perf_tables` grouping bug fixed the same morning, are
all the same axis: `enforce_eager`. Three in one day, on the one field added
that day.

That is not coincidence and it is not carelessness in the usual sense. A
newly added axis is exactly the variable that every habit, script and
mental shortcut still treats as constant, because all of them were formed
when it was. The risk window for a new dimension is not when it is being
designed — it is the hours afterwards, when everything around it is still
written for a world with one fewer dimension.

Two practical consequences, both now implemented:

- The sweep's progress label printed both arms identically, so the screen
  could not distinguish them. It now carries `graphs=on|off` and the
  `cell_id` prefix.
- `perf_tables.py` keys on `cell_id` rather than a hand-listed tuple. The
  lapses in 13 and 14 were both in *throwaway* analysis, not committed
  code, which is where the discipline actually fails.

What caught instance 14 was the label fix from instance 13 — the first run
after it printed `graphs=off` next to a cell whose numbers had already been
interpreted as graphs-on. The fix for one instance surfaced the next.

### The fifteenth, in this study's own headline

F021 reported that `--enforce-eager` makes speculative decoding
2.38x / 1.67x / 1.37x faster at c = 1 / 16 / 64. The perf_v2 grid measured
both arms of that configuration inside a single run and got
**1.13x / 1.06x / 1.05x**.

| # | Compared | Assumed fixed | Actually varied | Cost if unreported |
|---|---|---|---|---|
| 15 | eager vs default throughput | that the default arm was representative | the default arm is bimodal — this finding's own claim | the paper's section 4 headline overstated by ~2x |

The comparison arm was one draw from a distribution F021 itself
characterises. Both default boots in the concurrency test happened to land
slow; tonight's landed fast. Nothing was measured incorrectly — the ratio
simply is not a constant, and reporting it as one embeds a coin flip in a
headline.

This is the most uncomfortable instance and the most instructive. The error
is not in a throwaway script or a stale table: it is in the study's
strongest finding, in the number the paper leads with, committed by someone
who had spent the day cataloguing exactly this mistake and had written the
bimodality down himself three sections earlier.

Knowing the pattern does not confer immunity from it. Every defence that
worked today was mechanical — an append-only record, a cross-session guard,
a checker that fails the build. None of them was vigilance.

### The sixteenth, inside the guard written to prevent the fifteenth

F025's lesson was that the missing defence is a **range assertion**, and
one was added: inter-token latencies above 60 seconds raise, on the
reasoning that "an inter-token latency above a second on a local server is
not a slow token, it is a bug".

That reasoning is an assumption about the system, stated without measuring
it. Under concurrency vLLM preempts and recomputes requests when KV
pressure is high, and a preempted request genuinely waits. At concurrency
64 with 256 queued requests it waited **253 seconds**, the guard raised,
and a paid run died on a correct measurement.

| # | Compared | Assumed fixed | Actually varied | Cost if unreported |
|---|---|---|---|---|
| 16 | observed ITL vs "possible" ITL | that 60s exceeds any real gap | preemption stalls reach 253s | a killed run, and valid data rejected as corrupt |

The guard was replaced with one that tests **impossibility** rather than
magnitude: negative or non-finite only. Time does not run backwards and a
gap is not infinite, so those are always defects, and they are the F025
signature — that bug alternated sign within three tokens. Large positive
gaps are now counted and reported, never rejected.

Two things make this instance worth its own entry.

**The error was inside the fix for the previous one.** F025 concluded that
assumptions need assertions; the assertion then encoded a fresh unmeasured
assumption. Being right about the *category* of defence says nothing about
whether a particular instance of it is calibrated.

**It failed in the safe direction, and that is the only reason it was
cheap.** A guard that raises stops a run; a guard that silently filters
would have deleted the 253-second stall from the record and left a goodput
table that quietly excluded the worst-behaved requests — which is the
precise shape of F025 itself, one layer up. The run cost $0.99 and lost
nothing, because `boot_variance` appends per measurement and commits
periodically: all eight default-arm boots were already on disk when the
exception fired, and they are the boots the experiment needed.

Choosing loud failure over silent correction is what made a wrong guard
a fifteen-minute problem instead of a finding built on filtered data.

## Reasoning

The pattern is not carelessness about *measurement*. Each individual number
was measured correctly. It is carelessness about **what else changed**, and
it recurs because the third variable is by definition the one not being
thought about.

Two structural features of this study made the instances findable, and both
are worth keeping:

1. **Raw data is committed and append-only.** Eight of the eleven were
   caught by re-reading stored records against a hypothesis formed later.
   F021's corroboration came out of a grid collected before the hypothesis
   existed. None of that is possible if results are summarised at write time.
2. **Derived outputs are regenerated by script, never hand-edited.** When
   the scoring or grouping rule was wrong, the fix was one script and a
   re-run — no GPU, no re-measurement. F015 re-scored 58.6% of answers from
   stored text at zero cost.

The generalisable defence is narrow and cheap: **measure the noise floor of
the thing being compared before comparing it.** F006's backend claim moved
three times over two days and no argument resolved it, because every
argument was about the effect and none was about the floor. One four-boot
run settled it permanently. The same holds for F020: the boot term had to be
measured before any speculative speedup meant anything.

## What this does NOT establish

- **That the list is complete.** It enumerates errors that were *caught*.
  The same blindness that produced them applies to finding them, and the
  honest expectation is that the twelfth instance exists and is not on this
  list.
- **That the rate is unusual.** No comparable study reports its retractions,
  so there is no base rate. This is not evidence that this work is more or
  less error-prone than usual — only that it is more legible.
- **That checking is free.** Measuring a noise floor costs GPU. F020's boot
  variance cost about an hour; the eager axis in `perf_v2` costs eight extra
  boots. The claim is that these are cheap *relative to publishing a wrong
  number*, not that they are free.
- **Anything about the engine, GPU, or models.** This is a claim about the
  study's own method, not a measurement result.

## Consequence

This is the paper's thesis (`docs/paper-outline.md`). The three confounds
reported there — dtype-conditioned backend selection, bimodal CUDA-graph
boot modes, and the acceptance noise floor — are the three instances of this
pattern that are large enough to change a published number, and each is
individually sufficient to account for the headline result the study set out
to report.

It also argues for the artifact rules in `docs/data-model.md` being
*enforced* rather than intended. Writing F020's mandatory "What this does NOT
establish" section is what surfaced that reading a 2.16x range as continuous
variance was the error F021 later corrected. The section had been required
since rev 1 and unwritten since rev 1.

## How to reproduce

The classification is over `findings/`; re-derive it by reading them. The
structural checks that now catch instances automatically:

```bash
python analysis/check_provenance.py
```

```bash
python analysis/perf_tables.py
```
