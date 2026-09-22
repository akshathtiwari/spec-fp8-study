# Data model — how results, analyses and findings are recorded

**rev 2 · 2026-09-23**

> Changes since rev 1: the cell-id rule is now a versioned schema (§2.1), the
> budget log derives its own running total and covers every GPU entrypoint
> (§2.2), and the chain in §5 is enforced by a script rather than by
> intention (§6).

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

### 2.1 The cell-id rule is a versioned schema

`cell_id` is content-addressed, which makes the hashing rule part of the
on-disk format rather than an implementation detail. Hashing the whole model
dump made the schema unextendable: any new field rehashed every cell,
orphaning committed results and breaking the ids cited in findings. The
effect was that knobs worth measuring got tested *outside* the record —
the boot-variance probes deliberately bypassed `ServerCell` — so the store's
design was pushing evidence out of the store.

The rule now has two tiers:

- **v1 fields** (`_ID_SCHEMA_V1` in `specfp8/cells.py`) are always hashed.
  This reproduces every pre-existing id byte for byte.
- **Post-v1 fields** are hashed only when set to a non-default value. A cell
  that does not use a new knob keeps its old id, which is correct — it is
  the same configuration. A cell that does use it gets a new one.

Simply omitting every field at its default would not work: `attn_backend`
defaults to `"auto"` and is already inside the existing hashes, so omitting
it would rewrite the ids the rule exists to protect.

**Post-v1 defaults may not be changed.** Editing one silently re-partitions
ids, which orphans results without any individual record looking wrong.
`analysis/check_provenance.py` recomputes every stored id from its stored
config and fails if the rule drifts.

### 2.2 `budget.log`

One line per GPU invocation. `session_gpu_s` is that invocation;
`cumulative_gpu_s` is the running total, **derived by the writer** from the
sum of session figures on file rather than supplied by the caller — both
callers previously passed the session figure for both fields, so the total
restarted on every run and read an order of magnitude low.

Consumers should sum `session_gpu_s` rather than read the last
`cumulative_gpu_s`, because entries written before 2026-09-23 carry the
restarted value and `results/` is append-only, so they cannot be corrected
in place. `analysis/budget_report.py` does this.

The total is a **floor**: container start and image pull fall outside the
timed region, and four GPU entrypoints were unbilled before 2026-09-23 —
including the runs behind F020 and F021. The report states both gaps rather
than presenting a number that looks more precise than it is.

**`runs.json`** groups cells into the invocations that produced them. It is
**derived** by `analysis/build_runs.py`, not appended to — the only regenerated
file in `results/`, because a hand-maintained manifest went stale as soon as
another sweep landed and an audit found it covering 18 of 22 cells.

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

`reconstructed: true` is load-bearing honesty: no run so far stamped a run id on
its cells, so every boundary here is inferred from timestamp gaps. Stamping it
live is the proper fix and is not yet done; until it is, the field says so.

---

## 3. Derived layer — `analysis/out/`

```
analysis/out/
├── README.md          which script regenerates what
└── tables/*.md|csv    one file per analysis
```

`provenance.md` is the table to read first: provenance fields were added during
the study, each after finding a concrete way results could be silently mixed, so
it records which cells carry which guarantees rather than leaving an absent
field to be discovered.

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

## 6. The chain is checked, not merely asserted

`analysis/check_provenance.py` walks the chain a reader would walk and exits
with the number of defects, so it can gate a commit:

1. Every `analysis/out/...` path cited by a finding exists.
2. Every cited `cell_id` appears in `results/cells.jsonl`.
3. Those cells have a persisted engine log — backend selection and verbatim
   failure text live only there (F012, F017).
4. Frontmatter is present and uses the declared vocabulary.
5. `supersedes` / `superseded_by` are symmetric and resolve; a one-way link
   is how a retracted claim keeps getting cited.
6. Every stored `cell_id` still recomputes from its stored config (§2.1).

Rev 1 stated the rule in §5 and nothing enforced it. On first run the script
found seven defects. Five were retractions missing "What this does NOT
establish", which is the wrong rule for them — a withdrawal states what was
wrong rather than making a claim — so retractions are exempt and the script
says why. Two were real.

Writing one of the two paid for itself immediately: stating F020's limits
made clear that reading a 2.16x range as *continuous* variance is what made
its hypothesis look reasonable, when six boots cannot separate a wide
unimodal spread from two tight modes. That is the error F021 corrected, and
it had been sitting in an unwritten section the whole time. This is the
argument for rule 2 of §4 being enforced rather than aspirational.

A check that has never fired is not known to work, so rule 6 was verified
against a deliberate break (dropping one v1 field reports all 30 cells as
orphaned). F013 was a check that keyed on the wrong thing and looked fine.

---

## 7. Naming and time

- Run ids: `YYYY-MM-DDTHH-MMZ_<sweep>`, UTC, sortable.
- All timestamps in records are UTC with explicit `Z`.
- Finding ids are permanent. Numbers are never reused, even after retraction.
