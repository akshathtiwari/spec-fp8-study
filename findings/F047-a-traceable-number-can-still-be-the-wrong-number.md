---
id: F047
title: v1.5.0 shipped a wrong deficit dispersion band (3.73--10.38% for 3.73--13.92%) and stale session counts, all of which passed check_paper because it asks only whether a number exists somewhere in the evidence; a clean-room recomputation that requires the paper to contain each recomputed value now gates the build
kind: method
status: established
confidence: high
date: 2026-11-03
evidence:
  analysis: [analysis/recompute_cleanroom.py, analysis/out/tables/recompute_cleanroom.md,
             analysis/check_paper.py, analysis/check_provenance.py,
             analysis/out/tables/session_regimes.md]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

The v1.5.0 paper (Zenodo version of 2026-11-02) stated that the graph-arm
boot-to-boot CV of the deficit sessions spans **3.73--10.38%**. The records
give **3.73--13.92%**: the 2026-09-23 session, the deepest deficit (ratio
0.487), has a graph-arm CV of 13.92%, and it sits in the paper's own table.
The band was mistyped in F040 and copied from there into F044, F045, the
paper and the Zenodo description.

`check_paper` passed it. That checker extracts every number in the paper and
asks whether it occurs in some derived table. 10.38 does: it is the CV of the
2026-10-04 session. **A number can be traceable and still be the wrong
number for the sentence it is in.** This is the failure F028 described for a
different checker, in a new form: the instrument tested existence, and the
claim needed correspondence.

The same sweep found that v1.5.0's section 4 text had not been brought up to
date with the 2026-11-01 parity session (vLLM 0.29.0, ratio 1.012), which was
in the records and in the regime table generated from them:

| claim | v1.5.0 | records |
|---|---|---|
| Modal L4 paired DFlash sessions | nine | ten |
| deficit graph-arm CV band | 3.73--10.38% | 3.73--13.92% |
| parity graph-arm CV band | 0.86--1.95% | 0.82--1.95% |
| partition probability | 1/84 = 0.012 | 1/C(10,3) = 1/120 = 0.008 |
| pooled boots | 116 | 128 |
| parity boot range | 75.5--83.5 | 75.5--83.9 |
| mean tau range | 4.0702--4.1594 (2.2%) | 4.0702--4.1675 (2.4%) |
| thr/tau dispersion factor, deficit | 12--22 | 13--22 (12.6--22.4) |

None of these changes a conclusion. The two regimes stay disjoint in ratio,
in CV and in pooled boots; the partition is still one partition and is now
less likely under random assignment (0.008 against 0.012); tau CV stays below
0.88% in every session. The corrections are in v1.5.1.

## Evidence

`analysis/recompute_cleanroom.py` reads `results/probes.jsonl` with the
standard library alone. It imports nothing from `analysis/` or `specfp8/`,
so a bug shared by the analysis scripts cannot hide itself. It rebuilds the
session table, the regime partition, the pooled boots, the tau statistics
and the GCP ratios with their bootstrap intervals, then requires that
`paper/paper.tex` contain the **exact string** each recomputed value would be
typeset as (for example `3.73--13.92\%`, `$n = 128$`, `\binom{10}{3}`).

- On the corrected paper: **19 of 19** strings present, exit 0.
- On the v1.5.0 text: **9 of 19**, exit 1. It names the deficit band first.

The check is now a blocking step of `paper/build.sh`, before `check_paper`.

The same audit asked whether the record store itself could hide a defect.
`check_provenance` skipped malformed JSONL lines silently. It now reports
malformed lines, blank lines and byte-identical duplicate records as defects
(check 8). Planted in a scratch copy, a truncated record and a duplicated
record were both caught. In the real store there are 0 malformed, 0 blank and
0 duplicate lines, so no earlier result was affected.

## Reasoning

`check_paper` is still useful: it catches a number that exists nowhere. It
cannot catch a number that exists somewhere else, and in a study whose tables
are full of CVs between 0.8 and 14, almost any plausible CV is "somewhere".
The fix does not make `check_paper` smarter. It adds an independent path from
raw records to the specific sentences that carry the paper's argument, so
that the claim and the value have to agree, not just co-occur.

The clean room matters as much as the comparison. Had the recomputation
imported `analysis/session_regimes.py`, it would have agreed with the
generated tables by construction, and the tables were already right. The
error lived in hand-written prose. Only a check that ends at the prose can
see it.

## What this does NOT establish

- That every number in the paper is now correct. The gate covers the 19
  headline values of sections 1 and 4 and the abstract. Other sections are
  still covered only by `check_paper`'s existence test.
- That the generated tables were wrong. They were right throughout. The
  errors were in prose and in a finding's summary line.
- Anything about the mechanism. No measurement changed.
- That 3.73--10.38% was ever the right band. It was wrong when F040 was
  written: the 13.92% session is in F040's own table.

## How to reproduce

```bash
python -I analysis/recompute_cleanroom.py      # exit 0, 19/19
git show v1.5.0:paper/paper.tex > /tmp/old.tex
SPECFP8_PAPER=/tmp/old.tex python -I analysis/recompute_cleanroom.py   # exit 1, 9/19
python analysis/check_provenance.py            # check 8: 0 malformed, 0 blank, 0 duplicate
```
