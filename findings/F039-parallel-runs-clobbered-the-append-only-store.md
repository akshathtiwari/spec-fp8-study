---
id: F039
title: Running two Modal jobs concurrently silently destroyed 49 measurement records, because the append-only store is read-modify-written per container and the later commit wins; recovered only because two snapshots had been taken for an unrelated reason
kind: method
status: established
confidence: high
date: 2026-10-31
evidence:
  analysis: [specfp8/store.py, cloud/modal_probe.py]
  runs: [2026-10-31T06-59Z_boot_variance, 2026-10-31T07-30Z_boot_variance]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

Two boot-variance jobs were launched concurrently today, on the reasoning
that the GPU-hour quota expired at midnight and the Modal plan permits ten
concurrent GPUs. Both wrote to `results/probes.jsonl` on the shared Modal
volume.

The file is described throughout this repository as **append-only**. It is
not, under concurrency. Each container reads the file, appends its own
records, and commits the whole thing back. Two containers doing that from the
same starting snapshot produce a last-writer-wins result, and the earlier
writer's records are gone.

The 0.32.0 job finished first and wrote 72 records. The 0.29.0 job had
started from a snapshot taken before those existed, and its commit replaced
the file. **49 records were destroyed.**

```
local 564 / volume 587 / lost 49
```

The loss was visible only because the sync step prints how many local records
are absent from the volume. That check exists for a different reason -- to
confirm the volume is a superset before overwriting local -- and it is the
only thing that caught this.

## Recovery

Complete, and by luck rather than design. Two snapshots had been pulled
during the session, one after each job, for the unrelated purpose of
analysing each result as it landed. Their union restores everything:

```
p1 564  p2 587  union 636
  records from p1 missing after merge: 0
  records from p2 missing after merge: 0
```

Had both jobs been left to finish before a single sync, the 49 records would
have been unrecoverable and nothing would have reported their absence. The
analysis would simply have been computed over a 0.32.0 session with 8 default
boots instead of 12, and the shortfall would have looked like a truncated run.

## Why it happened

The store was designed for sequential use and its guarantees were never
stated as conditional. `append_budget_log` derives its cumulative total by
reading the file, which is the same pattern and the same exposure. Nothing in
`store.py` or in the Modal runner mentions concurrency, so there was no
warning to ignore.

The decision to parallelise was mine, made to fit two jobs into an expiring
quota window, without asking what the two jobs shared. They shared the only
mutable state in the system.

This is the same shape as F030 and F036: a property the repository asserts
about itself -- "append-only", "the tarball is built fresh", "gpu_type is
recorded" -- that holds under the conditions it was written for and silently
stops holding under a condition nobody checked.

## What this does NOT establish

- **That any published number was wrong.** The loss was detected and reversed
  within the same session, before any analysis was committed. The 636-record
  file is a strict superset of both snapshots.
- **That earlier results are affected.** Every prior run in this study was
  sequential. This is the first time two GPU jobs ran at once.
- **That the volume is now safe.** No locking has been added. The repository
  simply must not run concurrent jobs, and that constraint is now written
  down rather than assumed.

## Consequence

Immediate rule, and the reason the two follow-up sessions today were chained
rather than parallelised:

> **Never run two GPU jobs concurrently against the shared results volume.**
> The sequencing is not about cost or rate limits. It is the only thing
> preventing silent data loss.

Three cheaper safeguards, none yet implemented, in the order they would have
helped:

1. Have `append_record` refuse to write when the file it is about to replace
   is longer than the one it read. One comparison, catches exactly this.
2. Record a monotonic write counter per commit so a regression in record
   count is visible in the file rather than only in a sync message.
3. Give each container its own shard (`probes.<probe_run>.jsonl`) and
   concatenate at analysis time, which removes the shared mutable file
   entirely and is the correct fix.

Until one of those exists, the constraint is procedural, and procedural
constraints are the kind this study keeps documenting the failure of.

## How to reproduce

Do not. The conditions are: two `modal run` invocations overlapping in time,
both writing `probes.jsonl`. The symptom is a non-zero count in:

```bash
python - <<'PY'
import json
old=[json.loads(l) for l in open('results/probes.jsonl') if l.strip()]
new=[json.loads(l) for l in open('/path/to/fresh/volume/copy.jsonl') if l.strip()]
k=lambda r: json.dumps(r, sort_keys=True)
print("local records absent from volume:", len(set(map(k,old)) - set(map(k,new))))
PY
```
