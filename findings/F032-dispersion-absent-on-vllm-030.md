---
id: F032
title: On vLLM 0.30.0 the speculative boot dispersion is absent and default-path throughput is double the 0.29.0 figure, matching the eager arm; but the comparison is not host-controlled, because the 0.29.0 baseline pools three container sessions and this run is one
kind: result
status: established
confidence: medium
date: 2026-09-29
evidence:
  runs: [2026-09-29T04-48Z_boot_variance]
  logs: [results/logs/2026-09-29T04-48Z_boot_variance_default_0.log]
  analysis: [analysis/boot_statistics.py]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

Twelve boots of an identical DFlash speculative configuration on vLLM 0.30.0,
NVIDIA L4, bf16 weights, auto KV, FLASHINFER, concurrency 1, fixed prompt set:

| series | n | median tok/s | CV | IQR |
|---|---|---|---|---|
| 0.29.0 default | 12 | 40.8 | **13.92%** | 10.8 |
| 0.29.0 eager | 8 | 80.9 | 1.44% | 1.6 |
| **0.30.0 default** | 12 | **82.9** | **0.99%** | 0.9 |
| 0.30.0 eager | 12 | 81.9 | 1.28% | 0.2 |

Two things moved, and the second is the more informative:

1. **The dispersion is absent.** CV 13.92% to 0.99%, a factor of fourteen.
2. **Default-path throughput doubled**, 40.8 to 82.9 tok/s, arriving at the
   value the eager arm already had.

The eager arm barely moved: 80.9 to 81.9 tok/s, 1.44% to 1.28%. So 0.30.0 did
not make this configuration faster in general. It brought the default
CUDA-graph path **up to** the eager path, which on 0.29.0 it was running at
roughly half of. Whatever the graph path was intermittently doing on 0.29.0,
it stopped doing on 0.30.0.

Acceptance is unchanged and stable throughout: tau 4.15--4.20 across every
boot and concurrency in both arms, consistent with \S5 -- the quantity that
moves is throughput, not acceptance.

## Evidence

Measured, not declared (F031 now enforces this):

```
engine_version : {'0.30.0': 72}       all 72 records
engine_check   : {'honoured': 72}     declared engine_ref matched the server
gpu_uuid       : GPU-e08becdf-f674-3a4a-0cfa-3609da3036ec   (1 distinct)
clocks/power   : 2040 MHz max SM, 72 W limit
```

All 24 boots -- 12 default and 12 eager -- ran on **one physical card**. This
is the first measurement in this study where that is known rather than assumed,
because `gpu_uuid` did not exist before 2026-09-29.

## Two candidate mechanisms, both eliminated by the engine log

Before speculating about what changed, the persisted engine log says what did
*not*.

**The CUDA-graph fallback still happens on 0.30.0.** The warning is
byte-identical to the one \S4 quotes from 0.29.0:

```
WARNING [compilation.py:1473] CUDAGraphMode.FULL_AND_PIECEWISE is not
supported with spec-decode for attention backend FlashInferBackend
(support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE);
setting cudagraph_mode=PIECEWISE
```

Capture then proceeds under `PIECEWISE`, exactly as on 0.29.0. So "0.30.0
gave the speculative path full CUDA graphs" -- the reading suggested by
release notes #53867 and #51700, and by issue #33341 -- **is not what
happened here**. The fallback is present in both versions and the dispersion
is present in only one, so the fallback is not sufficient to produce it.

**Backend selection is also unchanged.** The 0.30.0 boot logs two distinct
selections, the target forced to FlashInfer and a second selection landing on
FLASH\_ATTN:

```
INFO [cuda.py:478] Using AttentionBackendEnum.FLASHINFER backend.
INFO [cuda.py:538] Using FLASH_ATTN attention backend out of potential
                   backends: ['FLASH_ATTN','FLASHINFER','TRITON_ATTN','FLEX_ATTENTION'].
```

That is confound 1 (F019) still live on 0.30.0. But it is *the same split as
0.29.0*: `results/logs/c6519fb025e64635.log`, the 0.29.0 DFlash cell, shows
the identical FLASHINFER-plus-FLASH\_ATTN pair. The draft side did not move
to a faster backend between versions.

So the two most plausible explanations are both refuted by stored evidence.
The mechanism is unidentified, and now unidentified in a stronger sense: it is
not merely uninvestigated, it is not the graph mode and not the backend.

## The confound, stated before the conclusion

**This comparison is not host-controlled, and the direction of the bias is
against the 0.29.0 arm.**

The 13.92% baseline pools 12 boots drawn from *three separate container
sessions* (`2026-09-23T07-20Z`, `11-08Z`, `19-28Z`) on rented instances whose
physical identity was never recorded. It therefore contains an unknown amount
of host-to-host variation.

This 0.30.0 measurement is 12 boots in *one session on one known card*. It
structurally cannot contain cross-host variation.

So part of the fourteenfold difference may be experimental structure rather
than engine version. The CV comparison is suggestive, not clean.

Two things are not explained by that confound:

- **The median doubled.** Cross-host scatter moves dispersion; it does not
  move a central tendency from 40.8 to 82.9. A host that is uniformly half
  speed is not a plausible reading of the 0.29.0 grid, where the eager arm on
  the same sessions reached 80.9.
- **The eager arm is unchanged across versions.** If 0.30.0 or a faster class
  of host were responsible, eager should have moved too. It did not.

The clean experiment is 12 boots of **0.29.0 in one session with `gpu_uuid`
recorded**, structurally matched to this one. That run is in flight. If
single-session 0.29.0 still shows ~14%, this is a boot effect and the version
difference is real. If it shows ~1%, then the published 13.92% was largely
cross-session variation and \S4's headline needs restating.

## What this does NOT establish

- **That vLLM 0.30.0 "fixed" anything.** No mechanism is identified, no
  upstream change is traced to this effect, and the comparison is not
  host-controlled. The defensible statement is that *on this card, in this
  configuration, the dispersion present on 0.29.0 was not present on 0.30.0*.
- **Which upstream change is responsible**, if any. 0.30.0's notes include
  decode-only FULL CUDA graphs (#53867), FULL graphs for microbatched steps
  (#51700), gc frozen during graph capture (#54646), and an online acceptance
  estimator driving adaptive verification (#52228). The graph-mode candidates
  are ruled out above: capture still runs under PIECEWISE. The acceptance
  estimator is not ruled out, but tau is flat at 4.15--4.20 across every boot,
  so if adaptive verification is active it is not changing the accepted-token
  rate. No change is traced to this effect, and the two candidates that were
  checkable were checked and failed.
- **That the 0.29.0 result was wrong.** It was measured, replicated across
  three sessions, and reproduces from the stored records today. What is open
  is whether it was boot dispersion or host dispersion.
- **Anything about other hardware, models, or mechanisms.** One L4, one
  target, one drafter, one workload, concurrency 1 for the dispersion figure.
- **That 0.30.0 is dispersion-free in general.** Twelve boots in one session
  on one card bounds dispersion under those conditions and nothing wider.

## Consequence

If the control run confirms a boot effect, the paper changes shape: it stops
being a dispersion measured on one engine release and becomes a quantified
regression caught by a measurement method and absent in the following
release. That is a stronger contribution, and it pre-empts the reviewer
question the current draft invites -- whether any of this still holds.

If the control run instead shows low dispersion on single-session 0.29.0, then
\S4's 13.92% is substantially a cross-session quantity, the paper must say so,
and the honest headline becomes *boot-to-boot dispersion is small; session-to-
session dispersion is not, and no published benchmark separates them*. That is
also publishable, and closer to the paper's actual thesis.

Both outcomes are reportable. Neither is assumed here.

The practice this vindicates:

> **Record physical device identity at measurement time.** Twelve boots could
> not settle boot-versus-host, and no number of additional boots would have.
> One field would have.

## How to reproduce

```bash
SPECFP8_VLLM_VERSION=0.30.0 modal run cloud/modal_probe.py \
    --engine bootvar --repeats 12 \
    --sweep-file sweeps/boot_stability_v030.yaml
```

```bash
python -c "
from analysis.boot_statistics import boots, describe
for er, hu in (('vllm-0.29.0', False), ('vllm-0.30.0', True)):
    v = boots('dflash', False, probes_only=True, engine_ref=er,
              gpu='NVIDIA L4', has_uuid=hu)
    print(er, describe(v))"
```

`has_uuid` is required: the 0.29.0 and 0.30.0 series share a GPU model and
differ only in `engine_ref`, and a 0.29.0 re-run would otherwise pool into the
published baseline and move it.
