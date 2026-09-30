---
id: F036
title: The fix for F031's declared-versus-measured defect was itself a declared value; budget.log recorded gpu_type "L4" for a run that took place on an A10, because SPECFP8_GPU is read locally and the remote container never sees it
kind: method
status: established
confidence: high
date: 2026-09-30
evidence:
  analysis: [cloud/modal_probe.py, analysis/budget_report.py, results/budget.log]
  runs: [2026-09-30T03-16Z_boot_variance]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

F031 recorded that the engine version was declared rather than measured, and
that a declared field feeding a decision is an assumption with a hash attached.
The same day, `analysis/budget_report.py` was found to price every run at a
hardcoded L4 rate with no record of which card ran, and `gpu_type` was added to
`budget.log` to fix it.

**That fix was itself a declared value, and it was wrong on its first live
use.** The budget entry for the A10 run reads:

```json
{"phase": "boot_variance", "session_gpu_s": 3390.8,
 "gpu_type": "L4", "detail": "boot_variance (container wall)"}
```

The run took place on an `NVIDIA A10`. Every measurement record from the same
run carries `env.gpu: "NVIDIA A10"`, captured from `nvidia-smi` inside the
container.

## Why it happened

`GPU_TYPE = os.environ.get("SPECFP8_GPU", "L4")` is evaluated at module import.
That happens twice, in two different places, and only one of them has the
variable set:

| where | `SPECFP8_GPU` | `GPU_TYPE` | effect |
|---|---|---|---|
| local shell running `modal run` | `A10G` | `A10G` | decorator requests `gpu="A10G"` — correct |
| remote container | unset | `"L4"` | budget entry records `"L4"` — wrong |

The variable did its job selecting the card and then silently failed to
describe it, because the value that reached the accounting code was computed
in a process that had never seen the card.

The deeper point is that the defect and its fix have the same shape. F031's
lesson was *do not let a declared constant stand in for a measured fact*. The
fix for it introduced a declared constant standing in for a measured fact, in
code written that hour, by someone holding the lesson. Naming a failure mode
is not the same as being immune to it.

The cost was small and quantifiable: 3390.8 GPU-s priced at $0.80/h instead of
$1.10/h, so the ledger undercounts that run by about $0.28, roughly 27%.

## What this does NOT establish

- **That any measurement is affected.** `gpu_type` is used only for costing.
  Every `env` block on every measurement record was captured from the card and
  is correct; `results/probes.jsonl` says `NVIDIA A10` throughout.
- **That the earlier entries are mispriced.** Runs before 2026-09-30 carry no
  `gpu_type` and ran on an L4, which is what they are priced at.
- **That the A10 entry is now correct.** It is not. `budget.log` is
  append-only and the wrong value stays on the record; only the ratio in this
  finding and the corrected code prevent it misleading a reader.

## Consequence

The billing decorator now asks the card rather than the environment:

```python
from specfp8.env import capture as _capture_env
_gpu = _capture_env().gpu          # "NVIDIA A10", from nvidia-smi
append_budget_log(..., gpu_type=_gpu)
```

This introduces a second problem that had to be handled rather than ignored:
the driver's name (`NVIDIA A10`) is not Modal's flag name (`A10G`), so the rate
table needed a mapping. An unmapped card now **reports itself loudly** instead
of defaulting quietly, because the silent default is a 9x undercount on a B300:

```
!! UNPRICED GPU(S) -- counted at the L4 rate, so this total is WRONG:
   'NVIDIA B300' has no entry in GPU_USD_PER_HOUR or DRIVER_TO_RATE_KEY
   Add it before trusting any figure above.
```

The rule worth carrying:

> **A value that describes the machine must be read on the machine.** If it
> can be computed in a process that never touched the hardware, it will
> eventually be computed there, and it will be wrong in a way that looks fine.

Three defects of this shape are now on record: F031 (engine version declared,
never stored), F030 (submission tarball assumed fresh, never verified), and
this one. All three were caught by something recomputing the value rather than
by anyone reading the code.

## How to reproduce

The wrong entry is on file:

```bash
grep '"gpu_type": "L4"' results/budget.log | tail -1
python -c "
import json
e = json.loads(open('results/budget.log').read().strip().split(chr(10))[-1])
print('budget says :', e.get('gpu_type'))
recs = [json.loads(l) for l in open('results/probes.jsonl') if l.strip()]
run = [r for r in recs if r.get('probe_run','').startswith('2026-09-30T03-16')]
print('records say :', {(r.get('env') or {}).get('gpu') for r in run})"
```
