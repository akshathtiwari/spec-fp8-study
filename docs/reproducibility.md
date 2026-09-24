# Reproducibility: what is pinned, and what is not

**rev 1 · 2026-09-24**

Written after external review observed that a content-addressed
configuration id proves identical *declared configuration*, not identical
model bytes or runtime environment. That is correct, and this document
states the gap rather than leaving the `cell_id` to imply more than it
delivers.

---

## What `cell_id` actually guarantees

A `cell_id` is `sha256` over the canonical JSON of a `ServerCell`. It
guarantees that two records with the same id were produced from the **same
declared configuration**: same model name, same revision *string*, same
precision, same backend, same speculative settings.

It does **not** guarantee:

- the same model weights, if the revision string names a mutable ref
- the same engine build, beyond the pinned version number
- the same Python dependency versions
- the same GPU instance, clocks, or power state

`argv_fingerprint` closes part of this by hashing the actual launch command,
and `env` records GPU model, compute capability, VRAM, driver, CUDA and
torch versions per record. Neither reaches the model bytes.

## Model revisions

The sweeps declare `revision: main`, which is a **mutable branch**. Resolved
to commits on 2026-09-24:

| Model | Commit | Last modified upstream |
|---|---|---|
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | 2025-07-26 |
| `z-lab/Qwen3-4B-DFlash-b16` | `b74e3a329c4d963783143b1e970d95b002be72bd` | 2026-04-07 |

**These were resolved after the measurements, not during them.** Both
upstream modification dates precede every run in `results/` (earliest run
2026-09-20), so `main` almost certainly did not move mid-study — but that is
an inference from upstream metadata, not a record the harness made. A
repository that wanted this airtight would resolve and store the commit at
launch. Ours does not, and the sweeps should carry the commit rather than
`main` going forward.

## Environment actually used

Recorded per measurement in `env`:

```
GPU            NVIDIA L4, compute capability 8.9, 22.5 GB
driver         580.95.05
CUDA           13.0
torch          2.13.0+cu130
engine         vLLM 0.29.0, read from the running server over HTTP
base image     nvidia/cuda:13.0.3-devel-ubuntu24.04
```

Harness dependencies were declared as open ranges (`pandas>=2.0` and
similar). Versions present in the environment that produced the analysis:

```
httpx 0.28.1   pydantic 2.13.5   pyyaml 6.0.3
pandas 3.0.6   matplotlib 3.11.2   numpy 2.5.3
```

These affect analysis and orchestration, not the measurements themselves —
the harness speaks HTTP and never imports the engine — but they are recorded
because "analysis is reproducible" is a claim about them too.

## What would make this airtight

Not done, and listed so the gap is explicit rather than discovered:

1. Resolve and store the model commit at launch, in the record.
2. Include the model commit in `cell_id`, which would make the id a claim
   about bytes rather than about names. Note this trades away id stability
   across upstream updates.
3. Pin exact dependency versions and ship a lockfile.
4. Record the container image digest, not just its tag.
5. Record GPU serial or instance identity, so host effects can be separated
   from boot effects — which §4 of the paper explicitly cannot do.

Items 1, 3 and 4 are cheap. Item 5 matters most for the paper's open
question, and item 2 is a real design trade rather than an oversight.
