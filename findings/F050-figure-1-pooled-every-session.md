---
id: F050
title: From v1.1.0 to v1.5.2 Figure 1 plotted every boot of every session on both cards (48 to 164 graph-arm boots) while its caption described the 2026-09-23 session alone (n=12, CV 13.92%); the script never filtered by session, so the figure drifted as sessions were added and no check compared it with its caption
kind: method
status: established
confidence: high
date: 2026-11-03
evidence:
  analysis: [analysis/fig_bimodality.py, analysis/out/tables/figure1_data.md,
             analysis/recompute_cleanroom.py, analysis/out/tables/recompute_cleanroom.md]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

Figure 1's caption says it plots "the 2026-09-23 session only, which is a
deficit session": DFlash's default arm at CV 13.92% over 12 boots, its eager
arm and both n-gram arms tight (1.44%, 2.67%, 0.85%), and every eager boot
above every default boot. That is what v1.0.0 plotted, when 09-23 was the
only day with boot-variance records.

`analysis/fig_bimodality.py` selected boot-variance records by mechanism and
arm and **never by session**. Each later session joined the figure on
regeneration:

| release | DFlash default boots plotted | caption says 09-23 only |
|---|---|---|
| v1.0.0 | 12 | no |
| v1.1.0 | 48 | yes |
| v1.2.0 | 60 | yes |
| v1.3.0 | 80 | yes |
| v1.4.0 | 128 | yes |
| v1.5.0-v1.5.2 | 164 | yes |

By v1.5.2 the plotted default arm spanned 29.5 to 137.3 tok/s, including the
A10 session near 135 tok/s. The "4.66x" spread drawn on it mixed cards, and
the caption's "every eager boot exceeds every default boot" was false for
the plotted points (pooled default boots near 83 tok/s against an eager
minimum of 63.8). The caption's numbers were right for the session it named.
The figure was not that session.

## Evidence

The table above is `git show <tag>:analysis/out/tables/figure1_data.md` for
each tag. v1.6.0 restricts the figure to `FIGURE_DAY = "2026-09-23"`: DFlash
19-28Z (default n=12, CV 13.92%; eager n=8, 1.44%) and the two n-gram
sessions of that day (default n=12, 2.67%; eager n=7, 0.85%). Those match
the caption exactly.

## Reasoning

This is F047's failure in a third form. The caption was traceable (13.92 is a
real CV), and the figure was generated from records, never drawn by hand.
Both checks passed, and neither asked whether the figure showed what the
caption said it showed. A generated figure is only as specific as its
selection, and a selection by mechanism and arm alone widens without notice
as the record store grows.

`analysis/recompute_cleanroom.py` now recomputes the 2026-09-23 DFlash
session from raw records, then requires both the caption string
(`CV 13.92\%, $n=12$`) and the figure's own data table (n and CV) to match.
Given v1.5.2's figure data, the gate fails (38/39, exit 1).

## What this does NOT establish

- That any conclusion depended on the figure. The text cites session tables,
  not the figure, for every number. The figure illustrated a claim the
  tables carry.
- That other figures are correct. There is one figure.

## How to reproduce

```bash
for t in v1.0.0 v1.1.0 v1.4.0 v1.5.2; do git show $t:analysis/out/tables/figure1_data.md | grep "DFlash default" | cut -c1-60; done
python analysis/fig_bimodality.py     # n=12, CV 13.92% for DFlash default
python -I analysis/recompute_cleanroom.py
```
