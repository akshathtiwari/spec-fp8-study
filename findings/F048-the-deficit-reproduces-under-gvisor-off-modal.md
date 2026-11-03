---
id: F048
title: The CUDA-graph deficit reproduces off Modal under upstream gVisor -- vLLM 0.29.0 on a GCP L4 gives a graph/eager ratio of 0.726-0.755 under two gVisor releases and two host drivers, against 1.018 under runc on the same card -- and vLLM 0.30.0 under the same gVisor is at parity (1.040); the sandbox is necessary on this host and the unsupported driver ABI is not, but the deficit sessions' boot-to-boot dispersion did not reproduce, and Modal's own parity on driver 610.57.04 is not explained by any upstream gVisor build tested
kind: result
status: established
confidence: medium
date: 2026-11-03
evidence:
  runs: [2026-11-03T03-29Z_boot_variance, 2026-11-03T05-08Z_boot_variance,
         2026-11-03T06-55Z_boot_variance, 2026-11-03T08-29Z_boot_variance,
         2026-11-03T10-14Z_boot_variance, 2026-11-03T11-51Z_boot_variance]
  logs: [results/logs/, results/gcp/2026-11-03_gvisor_decision_rule.txt,
         results/gcp/2026-11-03_gvisor_leg1_run.log, results/gcp/2026-11-03_gvisor_leg2_run.log,
         results/gcp/2026-11-03_gvisor_leg3_run.log, results/gcp/2026-11-03_gvisor_leg4_run.log,
         results/gcp/2026-11-03_gvisor_leg5_run.log, results/gcp/2026-11-03_gvisor_leg6_run.log]
  analysis: [analysis/sandbox_legs.py, analysis/out/tables/sandbox_legs.md,
             analysis/session_regimes.py, analysis/out/tables/session_regimes.md,
             analysis/engine_logs.py, analysis/out/tables/engine_logs.md]
  code_sha: pending
  external: [gVisor release-20260914.0 pkg/sentry/devices/nvproxy/version.go,
             https://storage.googleapis.com/gvisor/releases/release/20260914.0/x86_64/gvisor.tar.zstd,
             https://storage.googleapis.com/gvisor/releases/release/20261026.0/x86_64/gvisor.tar.zstd,
             Modal support reply 2026-11-03 (private correspondence)]
supersedes: []
superseded_by: []
---

## Claim

Six containers on one GCP L4 (GPU-8e3f74b9-3a97-0a87-df25-3459cebe7833,
asia-northeast1-a), the image F044 used (CUDA 13.0.3, Python 3.11, vLLM from
PyPI), DFlash, concurrency 1, 12 graph and 12 eager boots each, library
thread defaults of one as on Modal:

| leg | runtime | host driver | engine | graph tok/s (CV) | eager tok/s (CV) | ratio [95%] |
|---|---|---|---|---|---|---|
| 1 | gVisor 20260914.0 | 580.95.05 (unsupported ABI) | 0.29.0 | 57.2 (1.73%) | 77.7 (1.26%) | **0.736** [0.722, 0.750] |
| 2 | runc | 580.95.05 | 0.29.0 | 82.6 (1.10%) | 81.1 (1.02%) | **1.018** [1.004, 1.032] |
| 3 | gVisor 20260914.0 | 580.95.05 (unsupported ABI) | 0.30.0 | 81.4 (1.31%) | 78.3 (1.43%) | **1.040** [1.022, 1.058] |
| 4 | gVisor 20260914.0 | 580.95.05 (unsupported ABI) | 0.29.0 + F046 pin | 56.7 (0.91%) | 78.1 (1.57%) | **0.726** [0.715, 0.738] |
| 5 | gVisor 20260914.0 | 610.57.04 (supported) | 0.29.0 | 58.5 (1.33%) | 77.5 (1.26%) | **0.755** [0.742, 0.767] |
| 6 | gVisor 20261026.0 | 610.57.04 (supported) | 0.29.0 | 57.8 (1.34%) | 77.6 (1.31%) | **0.744** [0.732, 0.757] |

What this shows, in the order the legs were run:

1. **The deficit's size reproduces off Modal.** Under gVisor with nvproxy,
   vLLM 0.29.0's CUDA-graph arm runs at 0.736 of its eager control, inside
   Modal's deficit band (0.487-0.768). F044 and F045 found parity on the same
   driver and image under runc, on two other cards.
2. **The sandbox, not the card, does it.** runc on the same card, minutes
   later, is at 1.018. gVisor costs vLLM 0.29.0 about 30% of graph-arm
   throughput and about 4% of eager-arm throughput.
3. **The engine release matters under the sandbox, as it did on Modal.**
   vLLM 0.30.0 under the same gVisor and driver is at parity, matching
   Modal's two 0.30.0 sessions on 580.95.05 (1.014, 1.017).
4. **The driver does not matter here.** On 610.57.04, which nvproxy supports
   (no "not officially supported" warning), 0.29.0 under gVisor is still at
   0.755, and under the later gVisor 20261026.0 at 0.744.

F042 found every Modal deficit session on driver 580.95.05 and every 610.57.04
session at parity, and F044 showed the driver alone was not sufficient. F048
says what the driver was standing in for on this host: gVisor is necessary,
the driver is not. Modal told us a driver change on its platform requires a
sandbox change, which is why the two could not be separated there.

## Evidence

The legs were pre-registered (`results/gcp/2026-11-03_gvisor_decision_rule.txt`,
committed before the VM existed). Leg 1 landed in the rule's third case:
deficit ratio, graph-arm CV below the 3.73% threshold. Addendum 1 records
that case as pre-registered ("neither regime as defined") and fixes the
reading of legs 2-5 before they ran. Addendum 2 does the same for leg 6.
**Legs 2-6 are exploratory.** Their readings were committed before each ran,
but they were not part of the original rule.

Fidelity checks on leg 1, all passed:
- the container's dmesg reports gVisor ("Starting gVisor...");
- nvproxy's own log says "NVIDIA driver version 580.95.05 is not officially
  supported", the path Modal described (`-nvproxy-allow-unsupported-driver`);
- no engine log reports torch reducing its threads, as on Modal.

Every record carries `sandbox`, `omp_num_threads`, `image`, `cpus_visible`
and the card's UUID. Every graph-arm log on GCP, gVisor or runc, shows the V2
model runner with async scheduling on (`engine_logs.md`), so no leg changed
those factors as well. `sandbox_legs.py` uses the conservative interval F044
used and reproduces F044's published 1.001 [0.989, 1.013].

The two gVisor releases are the upstream tarballs, sha512-checked by
`cloud/gcp/host.sh gvisor`. release-20260914.0 was published 2026-09-16 and
registers 580.95.05 as `addUnsupportedDriverABI(580, 95, 05, v580_65_06)`.
Modal says its 2026-10-04 containers ran its build "Release20260916"
(cfce95d57233, not an upstream commit). The upstream build is the closest
public one, not a copy of Modal's.

**The GPU is idle, not throttled.** Each record samples GPU telemetry
across its measurement window. In every gVisor deficit leg the graph arm's
mean GPU utilisation is **62.2-63.2%**, against 90.0-92.3% in every parity
graph arm (runc, and 0.30.0 under gVisor). It clocks higher (2020-2024 MHz
against 1610-1893), draws less (67.5-68.0 W against 69.8-70.2) and is
power-capped less often (0.66-0.79 of samples against 0.96). That is the
signature section 4 of the paper found in Modal's 10-04 deficit session and
could not complete, because utilisation sampling started after Modal's last
deficit session. A GPU that boosts higher, draws less and does less work is
waiting for the host about a third of the time.

## Reasoning

A difference between runc and gVisor on one card, with the same image,
driver, threads and engine, can only come from what gVisor puts between the
process and the kernel and the device: the sentry's syscall handling, its
memory management, and nvproxy's ioctl forwarding. The graph arm loses far
more than the eager arm (30% against 4%). An ioctl-per-launch cost would hit
the eager arm hardest, since it issues every kernel launch, so the cost is
probably not per-ioctl. That points at something the graph path does and the
eager path does not, under 0.29.0 and not 0.30.0.

F046 proposed one such thing, a pageable host-to-device copy per decode step.
Leg 4 backported 0.30.0's pin into 0.29.0, and the deficit stayed (0.726).
F049 records that.

## What this does NOT establish

- **That this is Modal's mechanism.** It is the same symptom under a public
  gVisor build on one GCP host. Modal's build, host kernel, platform mode
  (systrap or KVM) and CPU allocation are not known to match.
- **Why Modal's 610.57.04 sessions were at parity.** Four Modal sessions on
  610.57.04 with vLLM 0.29.0 were at 1.003-1.015. Here 0.29.0 under gVisor
  is deficient on 610.57.04 under both a September and a late-October
  upstream release. Something about Modal's later sandbox, or about its hosts,
  differs and is not identified. `session_regimes.md` lists these four as the
  sessions the shared factors fail to explain.
- **The dispersion.** Modal's deficit sessions scattered from boot to boot
  (graph-arm CV 3.73-13.92%). Here the gVisor graph arm is as steady as
  runc's (0.91-1.73%). The deficit and the dispersion coincided on Modal and
  separate here. Shared-host contention on Modal is one reading of that, and
  it is untested.
- **Which change between vLLM 0.29.0 and 0.30.0 removes it.** Not the range
  buffer pin (F049). No bisection was run.
- **Generality.** One card, one region, one session per leg, DFlash only,
  concurrency 1. Each ratio's interval is narrow, but the replication unit is
  the container, and there is one per cell.

## How to reproduce

```bash
# GCP VM with one L4; each step in cloud/gcp/host.sh
bash cloud/gcp/host.sh driver 580.95.05   # reboot
bash cloud/gcp/host.sh docker
bash cloud/gcp/host.sh build                       # vLLM 0.29.0
VLLM=0.30.0 bash cloud/gcp/host.sh build
F046=1 bash cloud/gcp/host.sh build                # 0.29.0 + F046 pin
bash cloud/gcp/host.sh gvisor 20260914.0
RUNTIME=runsc THREADS1=1 bash cloud/gcp/host.sh run --mechanism dflash --boots 12              # leg 1
THREADS1=1 bash cloud/gcp/host.sh run --mechanism dflash --boots 12                            # leg 2
VLLM=0.30.0 RUNTIME=runsc THREADS1=1 bash cloud/gcp/host.sh run --mechanism dflash --boots 12 \
    --sweep-file sweeps/boot_stability_v030.yaml                                               # leg 3
F046=1 RUNTIME=runsc THREADS1=1 bash cloud/gcp/host.sh run --mechanism dflash --boots 12       # leg 4
bash cloud/gcp/host.sh driver 610.57.04   # reboot
RUNTIME=runsc THREADS1=1 bash cloud/gcp/host.sh run --mechanism dflash --boots 12              # leg 5
bash cloud/gcp/host.sh gvisor 20261026.0
RUNTIME=runsc THREADS1=1 bash cloud/gcp/host.sh run --mechanism dflash --boots 12              # leg 6

python analysis/sandbox_legs.py
python analysis/session_regimes.py
python analysis/engine_logs.py
```
