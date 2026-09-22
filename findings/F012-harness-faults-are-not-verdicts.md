---
id: F012
title: A leaked GPU process was recording host faults as compatibility verdicts
kind: method
status: established
confidence: high
date: 2026-09-21
evidence:
  runs: [2026-09-21T04-32Z_h1]
  analysis: [analysis/out/tables/runs.md]
  code_sha: b294050
---

## Claim

In run `2026-09-21T04-32Z_h1`, all eight cells failed and seven were recorded as
`launch_failed` — a status that reads as "this configuration is unsupported on
SM89", which is precisely hypothesis H1's claim. The real cause was a harness
defect. Had it not been caught, the study would have produced fabricated
supporting evidence for its own hypothesis.

## Evidence

```
ValueError: Free memory on device cuda:0 (2.03/22.03 GiB) on startup is less
than desired GPU memory utilization (0.92, 20.27 GiB).
```

Cell 1 hit the health-check timeout while downloading weights. `stop_process`
then signalled only the direct child, but vLLM 0.29 runs `EngineCore` and
`APIServer` as separate processes; those survived, holding ~20 GB, and every
subsequent cell failed to allocate.

## Reasoning

`cell_id` hashes the configuration, so a record keyed by it looks equally
authoritative whether the failure came from the configuration or from the host.
Nothing in the schema distinguished them.

## Fixes

1. Servers start with `start_new_session=True`; `stop_process` signals the
   whole process group and waits for it to disappear.
2. `probe_one_cell` checks free GPU memory before launching and records
   `status: harness_error` if the device is still occupied — a status that is
   always re-run and never counted as evidence about a configuration.
3. `argv_fingerprint` and `sampling_params` are recorded per cell, so a result
   produced by a different launch command or different sampling settings is
   detected as stale rather than silently reused.

## What this does NOT establish

- **That the seven affected cells are compatible.** The leaked process made
  their `launch_failed` records meaningless in both directions; they were
  re-run, and it is the re-run that carries evidence, not this finding.
- **That no other host fault was recorded as a compatibility signal.** A
  missing toolkit, a driver fault or an OOM fails a cell without changing
  the launch command, so `argv_fingerprint` cannot detect it. The fix adds a
  free-memory precondition and a distinct status; it does not make the
  harness able to recognise every environment fault.
- **That process-group kill is sufficient in general.** It covers workers
  forked by the engine. A process that re-parents itself out of the group,
  or a GPU held by a different container, would still be invisible.

## Lesson

A compatibility matrix must be able to say "the harness was broken" as a
distinct outcome from "the configuration is unsupported". Any status vocabulary
lacking that distinction will, under a harness bug, manufacture evidence for
whatever the matrix was built to test.
