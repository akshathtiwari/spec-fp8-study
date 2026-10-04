---
id: F035
title: On an NVIDIA A10 (SM86, Ampere) the same vLLM 0.29.0 configuration gives a default/eager ratio of 0.959 and CV 2.08%, against the L4's 0.487/0.768 and 13.92%/3.73%; the large deficit and the dispersion are largely properties of the L4, not of vLLM 0.29.0 speculative decoding
kind: result
status: established
confidence: medium
date: 2026-09-30
evidence:
  runs: [2026-09-30T03-16Z_boot_variance, 2026-09-29T06-29Z_boot_variance,
         2026-09-23T19-28Z_boot_variance]
  analysis: [analysis/boot_statistics.py]
  code_sha: pending
supersedes: []
superseded_by: []
# reading (3) refuted by F037
---

## Claim

The identical sweep --- vLLM 0.29.0, Qwen3-4B with the DFlash drafter, bf16
weights, auto KV, FLASHINFER, concurrency 1, one fixed prompt set, 12 boots in
a single container --- run on an NVIDIA A10 instead of an L4:

| card | arch | engine | session | arm | n | mean tok/s | CV |
|---|---|---|---|---|---|---|---|
| **A10** | SM86 | 0.29.0 | 09-30 | default | 12 | **132.6** | **2.08%** |
| **A10** | SM86 | 0.29.0 | 09-30 | eager | 12 | **138.3** | 1.28% |
| L4 | SM89 | 0.29.0 | 09-23 | default | 12 | 39.2 | 13.92% |
| L4 | SM89 | 0.29.0 | 09-23 | eager | 8 | 80.5 | 1.44% |
| L4 | SM89 | 0.29.0 | 09-29 | default | 12 | 61.2 | 3.73% |
| L4 | SM89 | 0.29.0 | 09-29 | eager | 12 | 79.7 | 1.69% |

Within-session default/eager ratio, which is the only quantity in this study
that has survived replication:

| card | engine | session | default / eager |
|---|---|---|---|
| **A10** | 0.29.0 | 09-30 | **0.959** |
| L4 | 0.29.0 | 09-23 | 0.487 |
| L4 | 0.29.0 | 09-29 | 0.768 |
| L4 | 0.30.0 | 09-29 b | 1.014 |
| L4 | 0.30.0 | 09-29 c | 1.017 |

**Same engine release, different card, and the effect largely disappears.**
On the L4, vLLM 0.29.0's CUDA-graph path costs 51% and 23% of eager
throughput. On the A10 it costs 4%. The boot dispersion follows: 13.92% and
3.73% on the L4, 2.08% on the A10.

The 4% is probably real rather than noise --- the two means differ by more
than either arm's CV --- so a small deficit does exist on Ampere. But it is an
order of magnitude smaller than the L4's, and it is the same size as the
0.30.0 residual on the L4.

## Why this matters to the paper

The title scopes the study to "vLLM 0.29.0 and NVIDIA L4". Until now that was
a statement of what was measured. It is now a statement about where the effect
lives: **the L4 half of that scope is load-bearing.** A reader who took
\S\ref{sec:c2} as a general property of speculative decoding under CUDA graphs
would be wrong, and the paper can now say so from evidence rather than from
caution.

It also changes what the finding is *about*. Two readings survive and they are
not the same:

1. The effect is architecture-conditioned (Ada versus Ampere).
2. ~~The effect is memory-capacity-conditioned.~~ **Retracted 2026-10-05.**
   This finding originally claimed the A10 has 24 GiB against the L4's 22.03
   GiB usable, and called capacity the more likely reading. **Both cards
   report 22.5 GiB.** The claim was never checked against `env.vram_gb`, which
   was sitting in every record. Capacity is not the difference.

3. The effect is power-conditioned. The L4 is a **72 W** part at 2040 MHz; the
   A10 is a **150 W** part at 1695 MHz. Less than half the power budget, and
   the faster execution path draws more. A card that throttles under graphs
   and not under eager would produce exactly what we see: the default arm
   slower *and* more variable, the eager arm neither, and the gap closing on
   the card with headroom.

Reading (3) replaces (2) as the plausible alternative to architecture.
**It was then measured and refuted: see F037.** Both arms are power-capped
84--95% of the time and the faster arm is capped more, at a lower clock,
drawing more power. Architecture is left standing, alongside a new reading
(stalls rather than a ceiling) that F037 states and does not claim. **Our records cannot test it:**
`env` stores the configured clock and power *limits*, not achieved clocks or
draw under load. A throttling hypothesis is precisely what this
instrumentation cannot see, which is the same gap `gpu_uuid` was added to
close one finding ago.

## What this does NOT establish

- **That the A10's numbers are reproducible.** One session. F033 is precisely
  about this: a dispersion from a single session is a draw, and by that
  standard the 2.08% is not yet quotable as the A10's dispersion. The ratio is
  a within-session quantity and so on firmer ground, but it is still n=1
  session.
- **A mechanism.** F032 rules out the two checkable candidates on 0.30.0, and
  nothing here identifies what differs between cards.
- **That Ampere is immune.** A 4% deficit was measured, not zero.
- **Anything about the intermediate cards.** Two GPU models, at opposite ends
  of a 3.4x throughput range (39--61 against 133 tok/s on the default arm).
- **That the L4 result was wrong.** It reproduces from the stored records and
  was measured twice. Its magnitude varies between sessions (F033); its
  existence does not.

## Provenance

```
engine_version : 0.29.0 on every record          (measured, F031 guard)
engine_check   : honoured on every record
env.gpu        : "NVIDIA A10"                    (driver name)
gpu_uuid       : GPU-9b62506e-596b-b3aa-402d-e6fe9f138d88   (1 distinct)
```

The card reports itself as `NVIDIA A10` although Modal's flag is `A10G`. The
paper uses the driver's name, since that is what the harness measured.

`analysis/boot_statistics.py` filters on `env.gpu` with a default of
`NVIDIA L4`, so these records cannot enter the published L4 series --- their
`cell_id` is identical to the L4 cells, because GPU model is deliberately not
part of a cell's identity.

## Consequence

\S\ref{sec:c2} should report the A10 row and state that the deficit is largely
L4-specific, with the surviving readings (architecture, power) named and
neither claimed. The honest headline for the section becomes narrower and more useful:

> On this card and this engine release the CUDA-graph path costs 23--51% of
> eager throughput, unpredictably. One release later it costs nothing. On a
> different card of the same generation-adjacent class it costs 4%. The effect
> is real, and it is conditioned on more than the engine.

**The experiment this now calls for is different from the one it originally
called for.** Capacity cannot be swept, because the cards already match at
22.5 GiB. What can be done, and cheaply, is to stop recording only the
configured limits and start sampling **achieved SM clock and power draw during
the measurement window**. If the L4's default arm throttles where its eager
arm does not, reading (3) is supported directly; if clocks hold at 2040 MHz
throughout, it is dead and architecture is left standing.

That is a harness change plus one L4 session, and it tests a hypothesis rather
than adding a third card to a comparison that already has three.

## How to reproduce

```bash
SPECFP8_GPU=A10G modal run cloud/modal_probe.py \
    --engine bootvar --repeats 12 \
    --sweep-file sweeps/boot_stability.yaml
```

```bash
python -c "
from analysis.boot_statistics import boots, describe
for gpu in ('NVIDIA L4', 'NVIDIA A10'):
    v = boots('dflash', False, probes_only=True,
              engine_ref='vllm-0.29.0', gpu=gpu, has_uuid=True)
    print(gpu, describe(v) if len(v) > 1 else '(none)')"
```
