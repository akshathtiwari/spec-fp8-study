---
id: F046
title: vLLM 0.30.0 pins two host buffers that FlashInfer's per-step decode planner copies to the GPU with non_blocking=True -- an internal arange buffer and kv_lens -- with comments saying the pinning keeps "the copy async"; in 0.29.0 neither is pinned, so every decode step makes a non-blocking copy from pageable memory, which CUDA stages synchronously; a code-grounded candidate for why 0.30.0 removed the deficit, not tested
kind: result
status: superseded
confidence: low
date: 2026-11-02
evidence:
  runs: []
  analysis: [analysis/vllm_pinning.py, analysis/out/tables/vllm_pinning.md]
  logs: [results/external/vllm/v0.29.0/flashinfer.py,
         results/external/vllm/v0.30.0/flashinfer.py]
  code_sha: pending
  external: [vllm v0.29.0 vllm/v1/attention/backends/flashinfer.py,
             vllm v0.30.0 vllm/v1/attention/backends/flashinfer.py,
             https://github.com/vllm-project/vllm/pull/54660,
             https://github.com/vllm-project/vllm/pull/54856,
             https://github.com/vllm-project/vllm/pull/54374]
supersedes: []
superseded_by: [F049]
---

> **Retracted by F049 (2026-11-03).** Tested under gVisor, where vLLM 0.29.0
> reproduces the deficit (F048): backporting 0.30.0's arange-buffer pin leaves
> it at 0.726. The kv_lens copy below is not made on the L4 (fa2 decode path),
> and both arms, not only the graph arm, make the remaining copy every step.
> The code reading stands as a description of the two releases. The candidate
> does not.

## Claim

On Modal, on driver 580.95.05, vLLM 0.29.0 was deficient and 0.30.0 was not
(F042). F044 and F045 showed the deficit needs the platform: 0.29.0 on a GCP L4
with the same driver is at parity. So whatever 0.30.0 changed cannot be a
plain GPU-side bug, which would reproduce on GCP. It has to remove something
the platform makes expensive. This finding reads the 0.29.0 -> 0.30.0 change
with that constraint.

The 0.30.0 release notes were searched for changes to speculative decoding,
DFlash, CUDA graphs, async scheduling, the V2 runner, FlashInfer, host-device
copies and synchronisation. Three candidates were examined:

| PR | change | verdict |
|---|---|---|
| [#54856](https://github.com/vllm-project/vllm/pull/54856) | skip the speculators' data-parallel sync (incl. DFlash), V2 runner | **down-ranked.** A DP collective, benchmarked at DP=2 for +1--2%; our runs are one GPU, where it is unlikely to cost a factor of two |
| [#54374](https://github.com/vllm-project/vllm/pull/54374) | DFlash drafters drop FlashAttention's AOT schedule | **down-ranked.** A GPU-side correctness bug whose symptom is acceptance collapsing to 1.0; our τ never collapsed (4.08--4.16), and a GPU-side bug would not depend on the platform |
| [#54660](https://github.com/vllm-project/vllm/pull/54660) | pin host tensors behind async host-to-device copies, incl. `vllm/v1/attention/backends/flashinfer.py` | **leading candidate**, below |

## What the code shows

`vllm/v1/attention/backends/flashinfer.py`, compared at the `v0.29.0` and
`v0.30.0` tags:

| per-step decode buffer | 0.29.0 | 0.30.0 |
|---|---|---|
| `paged_kv_indptr`, `paged_kv_last_page_len` | pinned (`_compute_flashinfer_kv_metadata`, l.1252--1277) | pinned (l.1730--1739) |
| `kv_lens` passed to the decode plan | not pinned in this file (`seq_lens_cpu` from common metadata, l.1405) | **pinned** (l.1745--1749): "plan() also copies kv_lens to the GPU with non_blocking=True" |
| FlashInfer's internal `arange` buffer (`_get_range_buf`) | **no reference** | **pinned** (l.2659--2661): "pin the cache entry so the copy stays async" |

The comments are the authors' own statement of the problem. A
`non_blocking=True` copy from pageable host memory is not asynchronous: CUDA
cannot DMA from pageable memory, so it stages the data through a pinned
buffer and the host waits. In 0.29.0 that happens inside FlashInfer's decode
planning, which runs every decode step.

An earlier reading in this session named `paged_kv_indptr` and
`paged_kv_last_page_len` as the pageable copies. **That was wrong**: in 0.29.0
both are pinned on the main decode path. The unpinned copies are `kv_lens` and
the `arange` buffer.

## Why this fits the record

| observation | how a per-step pageable copy accounts for it |
|---|---|
| deficit in the CUDA-graph arm only (F040) | graph replay makes per-step GPU work shortest, so a host-side stall idles the GPU most; eager is host-bound throughout |
| higher SM clock, less power, less work (F040) | a GPU waiting on the host between steps |
| fixed by 0.30.0 on the same driver (F042) | 0.30.0 pins exactly these buffers |
| absent on GCP with the same driver (F044, F045) | pageable staging is cheap on a plain host; the expectation, unmeasured, is that it costs more under a sandbox that intercepts the driver's memory operations |
| graph-arm dispersion 3.73--13.92% in deficit sessions (F040) | a stall whose cost depends on host state varies boot to boot |

## What this does NOT establish

- **That this is the cause.** It is the change in 0.30.0 that best fits the
  constraint, not one shown to matter. The decisive test is a cherry-pick:
  0.29.0 with these two buffers pinned, on a platform that produces the
  deficit, should be at parity. No such platform is available now (F042:
  Modal's fleet moved to a driver at which the deficit is gone; F044: GCP does
  not produce it).
- **That pageable staging is costlier under gVisor.** Expected from how the
  sandbox proxies the driver, not measured. The same cherry-pick on a
  sandboxed host would test it.
- **That `seq_lens_cpu` was pageable in 0.29.0.** It comes from the model
  runner's common metadata, whose allocation was not read. The `arange` buffer
  is unpinned in 0.29.0 regardless: the file never references it.
- **That #54660 is the only relevant change.** The 0.30.0 notes list several
  more pinned-copy changes (#55202, #56138) whose files were not read.
- **Why the driver rollout fixed 0.29.0.** On Modal, 0.29.0 on driver
  610.57.04 is at parity. If this mechanism is right, something in that
  rollout -- the driver, or the sandbox version that came with it -- made the
  staging cheap. Nothing here tests that.

## How to reproduce

The two upstream files are committed, unmodified, under
`results/external/vllm/` with their sha256 (Apache 2.0, vLLM authors), and the
table regenerates offline:

```bash
python analysis/vllm_pinning.py
```

Or from upstream directly:

```bash
for v in v0.29.0 v0.30.0; do
  curl -fsSLo flashinfer_$v.py \
    https://raw.githubusercontent.com/vllm-project/vllm/$v/vllm/v1/attention/backends/flashinfer.py
done
grep -n "_get_range_buf\|pin_host_range_buf\|kv_lens_decode_cpu\|copies kv_lens" flashinfer_v0.3*.py
```
