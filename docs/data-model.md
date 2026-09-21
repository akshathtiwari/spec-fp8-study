# Data model — how results, analyses and findings are recorded

**rev 1 · 2026-09-21**

This document specifies where every artifact lives, what may change it, and how a
claim in the paper is traced back to the bytes that justify it. It exists because
the study's value depends on a reader being able to check the work, and because
an earlier phase of this project kept its conclusions in commit messages and
terminal scrollback, where they could not be audited.

---

## 1. Four layers

| Layer | Path | Contract |
|---|---|---|
| **Raw** | `results/` | Produced by a run. Append-only. Never edited, never regenerated. |
| **Derived** | `analysis/out/` | Produced by a script from raw. Regenerable, disposable, never hand-edited. |
| **Findings** | `findings/` | Interpreted claims. Append-only; a wrong claim is superseded, not deleted. |
| **Narrative** | `docs/` | Requirements, design, tasks. Freely edited. |

The direction of dependency is strictly one way:

```
raw  ->  derived  ->  findings  ->  narrative
```

Nothing downstream may be the only record of something upstream. If a finding
cites a number, that number must be reproducible by running a script in
`analysis/` against `results/`.

### Why raw data is committed

`results/` is ~1.5 MB. The design originally gitignored it and deferred
publication to a Zenodo asset. At this size that trade is not worth making: a
reviewer who has to download a separate archive to see a log usually does not.
Committing it means every claim is one click from its evidence. Revisit only if
the directory approaches tens of megabytes, at which point per-request bodies
are the first thing to move out.

---

## 2. Raw layer — `results/`

```
results/
├── README.md                    provenance and regeneration instructions
├── cells.jsonl                  one record per probed cell, append-only
├── budget.log                   GPU seconds per phase
├── runs.json                    run manifests (see below)
├── requests/<cell_id>.jsonl     one record per generated request
└── logs/<cell_id>.log           engine stdout/stderr for that cell
```

**`cells.jsonl` is append-only and may contain superseded records.** A re-run
appends rather than overwrites, so the same `cell_id` can appear several times.
**The last record for a `cell_id` wins.** Every consumer must deduplicate; any
that does not is a bug. This is deliberate — the history of what a cell returned
before a harness fix is itself evidence, and three of this study's corrections
depend on being able to see it.

**`cell_id` is content-addressed**: `sha256` of the canonical JSON of the
`ServerCell`, truncated to 16 hex characters. Identical configuration produces an
identical id on any machine, which is what lets results from different sessions
merge. Two fields guard against a stale record being mistaken for a current one:

- `argv` and `argv_fingerprint` — the exact launch command, with port and
  download directory normalised out so the hash is stable across hosts.
- `sampling_params` — changes tau and accuracy without changing argv.

**`runs.json`** groups cells into the invocations that produced them:

```jsonc
{
  "run_id": "2026-09-21T07-41Z_backend",
  "sweep": "sweeps/backend.yaml",
  "engine": "vllm", "engine_version": "0.29.0",
  "gpu": "NVIDIA L4", "compute_cap": "8.9",
  "started": "2026-09-21T07:41:34Z", "ended": "2026-09-21T08:55:19Z",
  "cell_ids": ["..."],
  "gpu_seconds": 4973,
  "code_sha": "…",
  "reconstructed": true   // true when inferred after the fact, not recorded live
}
```

`reconstructed: true` is load-bearing honesty: the first three runs predate run
manifests and were partitioned afterwards from timestamps, so their boundaries
are inferred. Future runs record this live.

---

## 3. Derived layer — `analysis/out/`

```
analysis/out/
├── README.md          which script regenerates what
└── tables/*.md|csv    one file per analysis
```

Every file carries a header naming the script, the input, and the code SHA. These
files are committed so the paper's tables are diffable across revisions, but they
are **never authored by hand** — if a number here is wrong, the script is wrong.

---

## 4. Findings layer — `findings/`

One file per claim: `F###-short-slug.md`, numbered in the order established.

```markdown
---
id: F007
title: One-line statement of the claim
kind: result | method | retraction | gap
status: established | provisional | retracted | superseded
confidence: high | medium | low
date: 2026-09-21
evidence:
  runs: [2026-09-21T07-41Z_backend]
  cells: [5824814b68901229]
  analysis: [analysis/out/tables/backend_matrix.md]
  code_sha: abc1234
supersedes: [F003]
superseded_by: []
---

## Claim
## Evidence
## Reasoning
## What this does NOT establish
## How to reproduce
```

Four rules:

1. **`kind: retraction` is a first-class finding.** When a claim turns out to be
   wrong, the original is marked `status: retracted` and a new finding records
   what was believed, why it was wrong, and what caught it. This study has
   already produced several; hiding them would misrepresent how the results were
   arrived at, and they are among the more useful things a reader can learn from.
2. **"What this does NOT establish" is mandatory.** Most damage in empirical
   work comes from a true claim being read more broadly than its evidence
   supports. Stating the boundary is part of stating the claim.
3. **`confidence` is independent of `status`.** A finding can be established and
   low-confidence (measured once, on one GPU).
4. **Findings never cite terminal output.** If evidence is not in `results/` or
   `analysis/out/`, the finding cannot be written until it is.

---

## 5. Provenance chain

A number in the paper resolves like this:

```
paper claim
  -> findings/F0xx-*.md        what is claimed, and its limits
    -> analysis/out/tables/*   the computed number
      -> results/cells.jsonl   the cell records behind it
        -> results/logs/*.log  what the engine actually printed
          -> runs.json         which invocation, which GPU, which code SHA
```

Any break in that chain is a defect, and reviewing it is part of accepting a
finding.

---

## 6. Naming and time

- Run ids: `YYYY-MM-DDTHH-MMZ_<sweep>`, UTC, sortable.
- All timestamps in records are UTC with explicit `Z`.
- Finding ids are permanent. Numbers are never reused, even after retraction.
