---
id: F045
title: Pinned to one vCPU -- torch at one thread, as on Modal -- driver 580.95.05 on a GCP L4 is still at parity (ratio 1.028, 95% CI [1.012, 1.044]; graph-arm CV 0.95%), so one starved CPU does not produce the deficit; of F044's two platform candidates the sandbox is left, though a fractional CPU quota, unlike a pinned core, was not tested
kind: result
status: established
confidence: medium
date: 2026-11-02
evidence:
  runs: [2026-11-02T03-54Z_boot_variance, 2026-11-01T19-13Z_boot_variance]
  logs: [results/logs/, results/gcp/2026-11-02_cpu_pinned_run.log,
         results/gcp/2026-11-02_cpu_decision_rule.txt]
  analysis: [analysis/session_regimes.py, analysis/engine_logs.py,
             analysis/out/tables/session_regimes.md]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

F044 found the F042 configuration at parity on a GCP L4 and named two platform
differences as candidates: gVisor's driver proxy, and CPU. Every GCP engine log
reported torch starting at two threads; no Modal log did, so Modal's
containers saw one CPU or less. This finding tests the CPU candidate.

Same VM type, same driver (580.95.05, installed by hand), same image and
harness as F044, with the container pinned to **one vCPU** (`CPUSET=0`):

| DFlash, c=1, 12 boots each, one vCPU | tok/s | CV |
|---|---|---|
| CUDA graphs + async (default) | 81.7 | 0.95% |
| eager | 79.5 | 1.46% |
| **ratio** | **1.028**, conservative 95% CI [1.012, 1.044] | |

**Parity**, on both signatures: the ratio is inside Modal's parity band
(1.003--1.039) and the graph-arm CV of 0.95% inside its parity CV band
(0.86--1.95%), far from the deficit band (3.73--10.38%). One card throughout
(`GPU-8cf63913-107d-d607-1bf2-bd44da62bbaa`), driver 580.95.05 on every record,
τ at 4.15--4.17.

## The pin reproduced what Modal looked like

Checked before measuring, and recorded on every record:

| container | CPUs visible | torch threads | "Reducing Torch threads" in engine log |
|---|---|---|---|
| unpinned (F044) | 4 | 2 | yes, every log |
| **pinned (this run)** | **1** | **1** | **no -- 0 occurrences, as in every Modal log** |

Every record carries `cpus_visible = 1`, a field added for this run because the
study had never recorded CPU at all, which is how F044's candidate stayed
hidden for six weeks.

## Pre-registration

The decision rule was committed before the VM existed (`cb45273`), and an
addendum before any measurement (`a35a499`), after the VM landed on a
different card from F044's:

- ratio below 0.95 with a deficit-band CV: continue to the async-off arm,
  and first run an unpinned control on the same card, so a deficit could be
  told apart from a deficient card;
- ratio at or above 0.95: stop; one starved CPU is not sufficient.

The ratio came back 1.028 and the run stopped there. The same-card control was
conditional on a deficit and was not needed. The stated prior was parity, and
the result agrees with it; that agreement is not evidence of anything beyond
the result itself.

## What this changes

**CPU starvation, as tested, is ruled out as sufficient.** Two GCP sessions on
two cards, one unpinned and one pinned to a single vCPU, both at parity under
the configuration that was deficient on Modal 3 times in 3.

**If anything, starving the CPU pushed the other way.** With one vCPU the graph
arm ran 2.8% *ahead* of eager, and the interval excludes 1. That is the
textbook reason CUDA graphs exist: they remove CPU-side launch overhead, which
costs more when CPU is scarce, so a starved CPU should favour the graph arm --
the opposite of a deficit. The comparison with F044's 1.001 crosses two
cards, so the 2.7-point shift is not attributed to the pin; what stands is
that the pinned ratio is above 1, not below it.

**Of F044's candidates, the sandbox is left** -- gVisor's `nvproxy`, which on
Modal changes together with the host driver by construction. It is now the
leading hypothesis by elimination, which is weaker than being shown: the gap
F044 recorded remains, since current gVisor does not support 580.95.05 and no
source shows it ever did.

The platform list, updated:

| difference | status |
|---|---|
| CPU, one pinned vCPU | **ruled out as sufficient (this finding)** |
| CPU, a fractional quota | not tested |
| gVisor `nvproxy` | leading by elimination; cannot be tested faithfully without Modal's configuration |
| kernel-module flavour | open here; Modal's unknown |
| host kernel, container runtime | differ; not isolated |

## What this does NOT establish

- **That CPU is irrelevant.** A cpuset gives the process one whole vCPU. A CFS
  quota (`--cpus=0.25`, say) gives it a slice of a core and throttles it when
  the slice runs out -- a much harsher starvation, and one that torch's
  thread count cannot tell apart from a pinned core. If Modal allocates by
  quota, this test did not model it. Which one Modal uses is question 4 in the
  message drafted for Modal (`private/`, not sent).
- **That the sandbox causes the deficit.** It is what remains, not what was
  shown.
- **That 1 vCPU on a GCP host equals Modal's CPU.** Clock speed, cache and host
  contention differ even at one CPU.
- **Anything about async scheduling.** The async-off arm was skipped by the
  rule: in a session with no deficit there is nothing for it to remove.

## Cost

VM lifetime 1.83 h, about **$1.33 (Rs 117)**. GCP this month, both runs: about
**$2.79 (Rs 245)**, 12.3% of the Rs 2,000 budget. Logged in
`results/budget.log`; the VM and its disk were deleted after the copy was
verified, and nothing in the project is left billing.

## How to reproduce

```bash
bash cloud/gcp/host.sh driver 580.95.05   # then reboot; docker; build; verify
CPUSET=0 bash cloud/gcp/host.sh run --mechanism dflash --boots 12 --noasync
python analysis/session_regimes.py        # platform rows, with the cpus column
```
