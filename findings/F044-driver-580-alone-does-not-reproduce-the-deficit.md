---
id: F044
title: Driver 580.95.05 installed by hand on a GCP L4, with vLLM, torch, CUDA, model and harness matched to Modal, is at parity (ratio 1.001, 95% CI [0.989, 1.013]; graph-arm CV 0.91%) -- F042's conjunction is incomplete; the platform is a necessary factor, and two platform differences are candidates, gVisor's driver proxy and the CPU allocation
kind: result
status: established
confidence: medium
date: 2026-11-01
evidence:
  runs: [2026-11-01T19-13Z_boot_variance]
  logs: [results/logs/, results/gcp/2026-11-01_leg1_run.log,
         results/gcp/2026-11-01_decision_rule.txt]
  analysis: [analysis/session_regimes.py, analysis/engine_logs.py,
             analysis/out/tables/session_regimes.md,
             analysis/out/tables/engine_logs.md]
  code_sha: pending
supersedes: []
superseded_by: []
---

> **Followed by F045 (2026-11-02).** The CPU candidate below was tested: pinned
> to one vCPU (torch at one thread, as on Modal), the same configuration is still
> at parity (1.028). One starved CPU is not sufficient. A fractional CPU quota,
> as opposed to a pinned core, was not tested. The sandbox is left as the leading
> candidate, by elimination.

## Claim

F042 found every deficit session shares one configuration -- L4, driver
580.95.05, vLLM 0.29.0, DFlash -- and called for an intervention: install the
driver by hand on a host where it can be chosen, and measure. This is that
measurement.

A GCP `g2-standard-4` (one L4) was given driver **580.95.05**, installed from
NVIDIA's own installer, and ran the unmodified `boot_variance` function body
inside a rebuild of the Modal image (`cloud/gcp/`):

| | Modal records | GCP | |
|---|---|---|---|
| driver | 580.95.05 | 580.95.05 | matched (the variable) |
| card | L4, cc 8.9, 22.5 GiB | L4, cc 8.9 | matched |
| torch | 2.13.0+cu130 | 2.13.0+cu130 | exact |
| vLLM | 0.29.0 | 0.29.0 | exact |
| CUDA base | 13.0.3-devel-ubuntu24.04 | same | exact |
| Python | 3.11 | 3.11.18 | matched |
| runner / async | V2 / on | V2 / on | matched (engine logs) |

**Result: parity.**

| DFlash, c=1, 12 boots each | tok/s | CV |
|---|---|---|
| CUDA graphs + async (default) | 81.0 | 0.91% |
| eager | 80.9 | 1.03% |
| **ratio** | **1.001**, conservative 95% CI [0.989, 1.013] | |

Both signatures that define the regimes say parity. The ratio sits inside
Modal's parity band (1.003--1.039) and nowhere near its deficit band
(0.487--0.768). The graph-arm CV of 0.91% sits inside the parity band
(0.86--1.95%), not the deficit band (3.73--10.38%). One card throughout:
every record carries `GPU-82670c91-0059-2f1f-1637-d52139f32e12`.
Acceptance τ is 4.0825 at CV 0.12%, matching Modal.

**The configuration F042 named as sufficient is not sufficient.** Matched in
every recorded factor, it reproduces the deficit on Modal and not here.

## The decision rule, and why the 610 leg was not run

Two boots in, the graph arm read 78.7 and 81.2 tok/s -- parity-level. Before
any eager record existed, a rule was written down
(`results/gcp/2026-11-01_decision_rule.txt`):

> ratio < 0.95 -> deficit reproduced: run the full plan (async-off arm, then
> swap to 610.57.04). ratio >= 0.95 -> parity: stop after the eager arm and
> skip the 610 leg, since parity against parity is uninformative.

The ratio came back 1.001 and the run was stopped as the rule said. The
async-off arm had started and recorded nothing usable; it would have measured
nothing in a session with no deficit to remove. The 610 leg was not run.

## What this changes

### The "driver" in F042 may be the sandbox

On Modal, the host driver cannot change on its own. Modal runs containers
under gVisor ([Modal's own guide](https://modal.com/resources/best-serverless-sandboxes-ai-code-execution)),
whose GPU access goes through `nvproxy`, a proxy that translates every CUDA
driver ioctl and must know each driver version's ABI exactly
([gVisor GPU docs](https://gvisor.dev/docs/user_guide/gpu)). A driver rollout
therefore requires an `nvproxy` that supports the new driver --
[gVisor release-20260817.0](https://github.com/google/gvisor/releases/tag/release-20260817.0)
added support for exactly 610.57.04, the driver Modal moved to. Driver and
sandbox version are collinear on Modal by construction, the same way driver
and date were collinear in F041.

| | sandbox | result |
|---|---|---|
| Modal, driver 580.95.05 | gVisor `nvproxy` (earlier) | deficit |
| Modal, driver 610.57.04 | gVisor `nvproxy` (supports 610) | parity |
| **GCP, driver 580.95.05** | **none** | **parity** |

### A second platform difference: CPU

Every GCP engine log contains

> Reducing Torch threads from 2 to 1 for serving.

No Modal engine log does, in any session. vLLM prints that line only when
torch starts with more than one thread, so on Modal torch was already at one:
the containers saw one CPU, or less. `boot_variance` requested no CPU on Modal,
so it ran on the platform default, which this study never recorded. The GCP VM
had 4 vCPUs.

That matters under F043's hypothesis. Async scheduling overlaps the CPU-side
scheduler with GPU execution; an API server, an engine core and a scheduler
sharing one CPU can starve the GPU, which is what F040's telemetry describes --
a card at a higher clock, drawing less power, doing less work.

### What neither candidate explains on its own

- **CPU alone does not explain the Modal split.** The function definition, and
  so the CPU request, was identical in every Modal session, yet driver 580 was
  deficient and driver 610 at parity. CPU would need help from something that
  changed with the rollout -- burst capacity on less contended new hosts, or
  the sandbox.
- **gVisor alone has an unexplained gap.** Current gVisor (`release-20261026.0`)
  supports ten 580.x drivers but **not 580.95.05**, and no source found
  shows it ever did. Either Modal's Functions do not run on stock gVisor, or
  Modal forced `nvproxy` onto a neighbouring ABI
  (`--nvproxy-driver-version`), which gVisor documents as unvalidated.
- **Neither yet explains 0.30.0.** On Modal, vLLM 0.30.0 on driver 580.95.05
  was at parity. Under either candidate, 0.30.0 would have to make fewer
  driver calls, or less CPU work, on that path. Untested.

The two are not exclusive. A proxy adding CPU work to every driver call,
inside a container with one CPU, is a specific joint mechanism.

## What could not be tested, and why

A faithful gVisor arm needs the `nvproxy` Modal ran in September. Current
gVisor rejects 580.95.05 outright (strict version match), the apt repository
serves only the latest release, and the documented release-bucket URL now
returns 404. Forcing a neighbouring ABI would test a configuration nobody has
shown Modal used. It was not run.

## Next experiments, cheapest first

1. **Ask Modal.** Do Functions run under gVisor; which `runsc` release served
   L4 hosts from 23 September to 4 October 2026; was `--nvproxy-driver-version`
   set; what CPU does a function get with no `cpu=` request. Free, and it
   settles which candidate is worth a GPU-hour.
2. **CPU starvation on plain hardware.** Rerun leg 1 on driver 580.95.05 with
   the container pinned to one CPU (`--cpuset-cpus`). Fully controllable,
   about Rs 150. A deficit there makes CPU starvation sufficient on its own;
   parity rules it out as sufficient and leaves the sandbox.
3. **Explicit CPU on Modal.** One Modal session with `cpu=4`, on whatever
   driver the fleet now runs. Parity is already expected there, so this is
   weak alone, but it is the one platform knob Modal exposes.

## What this does NOT establish

- **That the sandbox or the CPU causes the deficit.** Both are candidates the
  data makes specific; neither was varied.
- **That the driver is irrelevant.** On Modal the driver and the sandbox moved
  together; this shows only that the driver is not sufficient without
  whatever Modal adds.
- **That GCP is a complete stand-in for "plain hardware".** One VM, one card,
  one session. The kernel module here is NVIDIA's open flavour (the
  installer default); Modal's flavour is unknown. Host kernel, container
  runtime and CPU differ too.
- **That the deficit cannot occur off Modal.** One session at parity bounds
  nothing about rare events. It shows the F042 configuration does not
  produce the deficit reliably, which on Modal it did, 3 sessions of 3.

## Cost

GCP VM lifetime 2.01 h, including the driver install, a 12-minute image build
and leg 1: about **$1.46, roughly Rs 128**, against a hard cap of 14 h
(about Rs 896) set at VM creation. Logged in `results/budget.log`. The VM and
its disk were deleted after the results were copied off; nothing in the
project is left billing.

## How to reproduce

```bash
# on a GCP VM with one L4 (see cloud/gcp/host.sh for each step)
bash cloud/gcp/host.sh driver 580.95.05   # then reboot
bash cloud/gcp/host.sh docker
bash cloud/gcp/host.sh build
bash cloud/gcp/host.sh verify
bash cloud/gcp/host.sh run --mechanism dflash,none --boots 12 --noasync

python analysis/session_regimes.py   # the platform row in the neighbour table
python analysis/engine_logs.py       # nb:platform: V2 runner, async on
```
