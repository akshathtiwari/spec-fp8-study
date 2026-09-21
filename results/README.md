# Raw results

Produced by runs; **never edited by hand, never regenerated**. Everything in
`analysis/out/` and every claim in `findings/` derives from these files.

| file | what |
|---|---|
| `cells.jsonl` | one record per probed cell — config, env, status, tau, capacity, verbatim error, `argv`, `argv_fingerprint`, `sampling_params` |
| `requests/<cell_id>.jsonl` | every generated request: timings, token counts, and the full output text |
| `logs/<cell_id>.log` | the engine's own stdout/stderr, including its backend-selection line |
| `budget.log` | GPU seconds per phase |
| `runs.json` | run manifests — which cells, which GPU, which engine version |

## Two things that will bite a consumer

**`cells.jsonl` is append-only and contains superseded records.** A re-run
appends; the same `cell_id` can appear several times and **the last one wins**.
Any consumer that does not deduplicate is wrong. This is deliberate: the history
of what a cell returned *before* a harness fix is itself evidence, and several
findings depend on it (see `findings/F012`).

**Not every cell has a log.** `logs/` only contains cells run after log
persistence was added, so the earliest runs have none. `backend_attribution.md`
reports `None` for those rather than guessing.

## Provenance

`cell_id` is `sha256(canonical_json(ServerCell))[:16]` — content-addressed, so
the same configuration yields the same id on any machine and results from
separate sessions merge. `runs.json` entries are marked `reconstructed: true`
where boundaries were inferred from timestamps after the fact.

## Regenerating the derived tables

```bash
python analysis/build_tables.py    # -> analysis/out/tables/
python analysis/build_index.py     # -> findings/README.md
```

Both are deterministic, need no GPU, and no network.

## Pulling fresh results off Modal

```bash
modal volume get specfp8-results / results --force
```
