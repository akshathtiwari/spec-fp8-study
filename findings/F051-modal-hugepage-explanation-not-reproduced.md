---
id: F051
title: Modal attributes its late-October parity to a gVisor change that lets GPU-shared memory use huge pages (google/gvisor#16008 and #16009, upstream; #16215, unmerged); off Modal, upstream release-20261102.0, which contains the two merged changes, leaves vLLM 0.29.0 deficient on a second card (0.734, and 0.729 with host shmem huge pages set to advise), but the host never allocated shared-memory huge pages, so the explanation is neither confirmed nor refuted
kind: result
status: established
confidence: medium
date: 2026-11-04
evidence:
  runs: [2026-11-04T11-10Z_boot_variance, 2026-11-04T12-48Z_boot_variance,
         2026-11-04T14-24Z_boot_variance]
  logs: [results/logs/, results/gcp/2026-11-04_hugepage_decision_rule.txt,
         results/gcp/2026-11-04_hugepage_legA_run.log,
         results/gcp/2026-11-04_hugepage_legB_run.log,
         results/gcp/2026-11-04_hugepage_legBp_run.log]
  analysis: [analysis/sandbox_legs.py, analysis/out/tables/sandbox_legs.md]
  code_sha: pending
  external: [https://github.com/google/gvisor/pull/16008,
             https://github.com/google/gvisor/pull/16009,
             https://github.com/google/gvisor/pull/16215,
             Modal support reply 2026-11-04 (private correspondence)]
supersedes: []
superseded_by: []
---

## Claim

F048 left one thing unexplained: four Modal sessions on driver 610.57.04 with
vLLM 0.29.0 were at parity (1.003-1.015), while every upstream gVisor build
tested off Modal was deficient. Modal's answer (support, 2026-11-04): in late
October it rolled out a gVisor build that lets memory shared with the GPU use
huge pages, all of those sessions ran on it, and the changes are upstream as
google/gvisor#16008 and #16009. Modal says release-20261026.0 has only part of
the change and release-20261102.0 all of it.

Checked against upstream: 20261026.0 (commit e8a2a288) contains #16008 and not
#16009; 20261102.0 (commit 7a675ac6) contains both. #16009's own description
also needs #16215 ("mm: hugepage-align shared anonymous mappings"), which is
**not merged**, and its benchmark ran with the host's shared-memory THP set to
`advise`.

One new card (GPU-94fbf18f-fbf0-1f81-808a-042a54a26d05, asia-northeast1-c),
610.57.04, the F048 image, threads at one, 12 + 12 boots:

| leg | gVisor | host shmem THP | ratio [95%] | graph-arm util |
|---|---|---|---|---|
| A | 20261026.0 | never | **0.734** [0.719, 0.748] | 62.5% |
| B | 20261102.0 | never | **0.734** [0.719, 0.750] | 62.1% |
| B' | 20261102.0 | advise | **0.729** [0.717, 0.742] | 61.9% |

1. **Upstream 20261102.0 does not reproduce Modal's parity** on this host. The
   pre-registered reading of A and B says so, and B' agrees.
2. **The test did not reach the mechanism.** With THP for shared memory at
   `never` (Ubuntu's default, found only after B), gVisor cannot get huge
   pages from the host at all. With `advise`, `ShmemHugePages` was still
   **0 kB** ten minutes into B'. So in none of the three legs was any GPU-shared
   memory hugepage-backed. #16215, unmerged, is the likely missing piece:
   #16009's description says it is what makes the application's buffers
   hugepage-aligned in the first place.
3. **The deficit replicates on a second card.** A reproduces F048 leg 6
   (0.744) on a different card in a different zone (0.734), with the same
   idle-GPU signature.

## Evidence

The rule for A and B was committed before the VM existed
(`2026-11-04_hugepage_decision_rule.txt`). Addendum 1, committed after B and
before B', records the shmem finding and fixes B''s reading. **B' is
exploratory.** Leg C (20261026.0 with `advise`) was conditional on B' reaching
parity, and was not run. All three legs passed the F048 fidelity checks: gVisor
dmesg, no unsupported-driver warning on 610.57.04, no torch thread reduction.

The `shmem_enabled` values and the mid-run `ShmemHugePages` reading are in the
leg run logs. Records do not carry the setting, so
`analysis/sandbox_legs.py` holds it in a small table that cites those logs.

## Reasoning

Modal's explanation fits what F048 measured. Each step the GPU is idle about
a third of the time, waiting on the host. nvproxy mirrors pinned application
memory into ranges of its own (#16009), and if those mirrors are backed by
small pages, every transfer through them pays for it. Why that would cost the
graph arm far more than the eager arm is not explained by this, and a
mechanism that fits is not one that was shown. Off Modal, the precondition, hugepage-backed
GPU-shared memory, was never met, so the run says nothing either way about
whether meeting it removes the deficit.

## What this does NOT establish

- **That Modal's explanation is wrong.** The precondition was not met.
- **That it is right.** Only Modal's sessions show the parity, on a build that
  is not public.
- **What would complete the test:** a gVisor built with #16215 (or Modal's
  build) on a host whose shared-memory THP is enabled, with `ShmemHugePages`
  confirmed non-zero during the run. That needs a source build of gVisor and
  was not done.

## How to reproduce

```bash
bash cloud/gcp/host.sh driver 610.57.04   # reboot
bash cloud/gcp/host.sh docker && bash cloud/gcp/host.sh build
bash cloud/gcp/host.sh gvisor 20261026.0
RUNTIME=runsc THREADS1=1 bash cloud/gcp/host.sh run --mechanism dflash --boots 12   # A
bash cloud/gcp/host.sh gvisor 20261102.0
RUNTIME=runsc THREADS1=1 bash cloud/gcp/host.sh run --mechanism dflash --boots 12   # B
echo advise | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled
RUNTIME=runsc THREADS1=1 bash cloud/gcp/host.sh run --mechanism dflash --boots 12   # B'
grep ShmemHugePages /proc/meminfo                                                    # during B'
python analysis/sandbox_legs.py
```
