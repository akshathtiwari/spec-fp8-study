---
id: F031
title: The engine version was read from the server, printed to the terminal and never stored, so every version claim in the study rests on a declaration; and because engine_ref is part of cell_id, a version replication would have silently superseded its own baseline
kind: method
status: established
confidence: high
date: 2026-09-29
evidence:
  analysis: [specfp8/probe.py, specfp8/launchers/vllm.py, cloud/modal_probe.py, docs/reproducibility.md]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

Two defects, found together while preparing a vLLM 0.30.0 replication, with
the same root cause: **a field that is declared and never verified.**

**First: the measured engine version was never stored.**
`VllmLauncher.engine_version()` queries the server's `/version` endpoint, and
`probe_one_cell` prints the result. The key is absent from **all 394 records**
in `results/`:

```
results/sweep.jsonl:  n=278  key absent=278  present-but-null=0  with-value=0
results/probes.jsonl: n=116  key absent=116  present-but-null=0  with-value=0
```

Absent, not null. `specfp8/sweep.py` never captures it at all, and the line in
`probe.py` that would record it postdates the entire corpus. So the version
that every result in this study is attributed to rests on two declarations:
the `VLLM_VERSION` pin in `cloud/modal_probe.py`, and
`engine_ref: vllm-0.29.0` hardcoded in all thirteen sweep files.

`docs/reproducibility.md` asserted the opposite. Its environment table read
*"engine vLLM 0.29.0, read from the running server over HTTP"* — in the
document written specifically to state what is and is not pinned.

**Second, and worse: `engine_ref` is part of `cell_id`.** It sits in
`_ID_SCHEMA_V1`. A declared value that nothing verifies, feeding the hash that
identifies a measurement, in an append-only store where a repeated `cell_id`
supersedes the earlier record.

The consequence was one command away. Running a 0.30.0 image against
`sweeps/boot_stability.yaml` — which declares `vllm-0.29.0` — would have
produced records carrying the **0.29.0 cell_ids**, and the store would have
treated them as re-runs. A version comparison would have destroyed the
baseline it was comparing against, reported no error, and left a corpus in
which 0.30.0 measurements were labelled 0.29.0.

## Evidence

| Field | Declared where | Verified against server? | In `cell_id`? |
|---|---|---|---|
| `engine_ref` | 13 sweep YAMLs, hardcoded `vllm-0.29.0` | **no** (until now) | **yes** |
| `VLLM_VERSION` | `cloud/modal_probe.py` constant | n/a, controls the wheel | no |
| `engine_version` | measured from `/version` | read, printed, **not stored** | no |

The author had already noticed half of it. `specfp8/probe.py` carries the
comment *"`engine_ref` in the sweep is a declared intent nothing verifies;
this is measured"* — written directly above a measurement that was then
discarded. Naming a gap is not closing it.

## Why it happened

`launcher.engine_version()` ends in `except Exception: return None`. That
converts every failure — a 404, a renamed endpoint, a timeout — into the same
value as "not applicable". Nothing downstream asserted it was non-None,
because nothing consumed it: the study only ever ran one engine version, so
the field had no reader, and a field with no reader is not exercised by
anything.

The deeper pattern is the one F028 named. An instrument gets built for the
question in front of it. `engine_version` was built for a cross-engine
comparison that never happened, so it was never wired into the store, and the
single-version study never noticed because a constant does not look like a
missing measurement. The defect only became reachable the moment a second
version was introduced — which is also the moment it would have done damage.

`_ID_SCHEMA_V1` compounds it. Putting `engine_ref` in the identity hash was
right: two engine versions genuinely are different cells. But an identity hash
built from an unverified declaration means the id is only as trustworthy as
the YAML, and the id is what the store uses to decide whether to supersede.

## What this does NOT establish

- **That any published number is wrong.** The image pins `vllm==0.29.0`, so
  0.29.0 is what ran. No result changes. What was wrong is the stated basis
  for the version attribution, not the attribution.
- **That the corpus is corrupted.** No 0.30.0 run was ever launched. The
  defect was found while preparing one, before any GPU time was spent.
- **That `/version` works.** The key is absent rather than null, so the HTTP
  call was never reached on the paths that wrote these records. Whether the
  endpoint answers on vLLM 0.29.0 is untested; the first run after this fix
  is what establishes it, and `engine_check: unverified` is the outcome if it
  does not.
- **That other declared axes are verified.** `model_revision` is declared as
  `main`, a mutable ref, and resolved post hoc (`docs/reproducibility.md`).
  That is the same defect shape in a field that is also in `_ID_SCHEMA_V1`,
  and it is not fixed by this change.

## Consequence

Three changes, all in this commit:

1. **The measured version is persisted**, alongside an `engine_check` field
   taking `honoured` / `mismatch` / `unverified`. `unverified` is not a pass;
   it records that the cell does not establish which engine served it.
2. **A mismatch aborts.** Both `probe_one_cell` and `boot_variance` raise
   before writing any record if the measured version disagrees with the
   declared `engine_ref`. The backend check records a mismatch and continues;
   this one cannot, because continuing overwrites another version's data.
3. **`docs/reproducibility.md` states the correction** rather than quietly
   dropping the sentence.

The practice worth generalising:

> **A declared field that feeds `cell_id` must be verified against the system
> at run time, or it is not an identity, it is an assumption with a hash
> attached.**

Two fields in `_ID_SCHEMA_V1` are still declarations: `engine_ref` is now
verified, `model_revision` is not.

## How to reproduce

The absence, on the corpus as committed before this change:

```bash
python - <<'EOF'
import json
for f in ('results/sweep.jsonl','results/probes.jsonl'):
    n=absent=0
    for line in open(f):
        if line.strip():
            n+=1; absent += 'engine_version' not in json.loads(line)
    print(f, 'n=%d absent=%d' % (n, absent))
EOF
```

That the two versions no longer collide:

```bash
python -c "
from specfp8.cells import expand, cell_id
a={c.mechanism: cell_id(c) for c in expand('sweeps/boot_stability.yaml')}
b={c.mechanism: cell_id(c) for c in expand('sweeps/boot_stability_v030.yaml')}
assert all(a[m]!=b[m] for m in a), 'COLLIDE'
print('distinct')"
```
