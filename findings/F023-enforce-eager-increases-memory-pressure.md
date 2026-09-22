---
id: F023
title: enforce_eager increases KV allocation and can make a speculative cell unbootable, so it is not a free fix for the bimodality
kind: result
status: established
confidence: high
date: 2026-09-23
evidence:
  runs: [2026-09-22T19-4xZ_perf_v2]
  cells: [4ae0a860802f247c]
  analysis: []
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

`--enforce-eager` is the fix F021 recommends for the bimodal speculative
throughput. It is **not free**. Disabling CUDA graphs makes vLLM allocate
*more* KV cache, not less, and on a memory-tight card that is enough to make
a speculative configuration fail to boot.

On a 22.03 GiB L4 the failing set is precisely **speculative decoding +
`fp8_e4m3` KV cache + `enforce_eager`**: both such cells in the grid fail
to boot, three times across two containers. The same cells with CUDA graphs
on boot normally, and eager cells with bf16 KV boot normally — including
the very cell F021's recommendation was measured on.

## Evidence

From `results/boot_failures.jsonl`, with an entirely clean GPU:

```
free_before_boot = 22561.0 MiB      (22.03 GiB card, nothing resident)
status           = oom

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 892.00 MiB.
GPU 0 has a total capacity of 22.03 GiB of which 745.06 MiB is free
  logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
```

The mechanism is stated by the engine itself, in the log of the *successful*
graphs-on boot of the paired cell:

```
CUDA graph memory profiling is enabled (default since v0.21.0). The current
--gpu-memory-utilization=0.9200 is equivalent to
--gpu-memory-utilization=0.8843 without CUDA graph memory profiling.
```

With graphs on, the memory profiler reserves headroom for graph capture, so
the *effective* utilization is 0.8843 and the successful boot settles at
9.61 GiB weights + 9.07 GiB KV + 1.59 GiB peak activation = 20.27 GiB, with
room to spare.

With `--enforce-eager` that reservation is skipped, so the engine allocates
KV against the full 0.92. The extra KV is taken from exactly the headroom
the float32 logits buffer needs, and 892 MiB no longer fits in the 745 MiB
left.

The logits buffer is large because this is speculative decoding: the
verification step materialises logits for the drafted tokens, so the buffer
scales with speculative width times a ~151k vocabulary in float32.

### Reproduced

The failure occurred twice, in two separate Modal containers on different
L4 instances (the first run was preempted mid-grid and retried). Both had a
clean card and failed identically at the same allocation. It is a property
of the configuration on this hardware, not a one-off allocation race.

The paired graphs-on cell `5824814b68901229` booted successfully in the
same session, so the two are not separated by anything but the flag.

## Reasoning

The intuition is that turning CUDA graphs off should *save* memory, since
nothing is captured. That is wrong here, and backwards for a specific
reason: the allocation being skipped is a **reservation**, not a
consumption. vLLM sizes the KV cache to fill whatever the profiler says is
available, so removing a reservation hands that memory to the KV cache
rather than leaving it free.

### Correction: weight precision is not the discriminator

This finding first reasoned that the failing cell was "the heaviest
combination in the grid" because it pairs bf16 weights with FP8 KV. **That
was wrong**, and the grid falsified it within the hour:
`dflash | fp8 | fp8_e4m3 | eager` failed too, with *lighter* weights.

All three recorded failures are identical to the byte:

| time | cell | free before boot | allocation attempted |
|---|---|---|---|
| 20:00:57 | dflash\|bf16\|fp8_e4m3\|eager | 22561 MiB | 892.00 MiB |
| 20:15:58 | dflash\|bf16\|fp8_e4m3\|eager | 22561 MiB | 892.00 MiB |
| 20:35:30 | dflash\|fp8\|fp8_e4m3\|eager | 22561 MiB | 892.00 MiB |

The identical 892 MiB across both weight precisions is the tell: the logits
buffer is sized by speculative width times vocabulary in float32, and has
nothing to do with how the weights are stored.

The correct mechanism is more interesting than the one it replaces.
**vLLM sizes the KV cache to fill whatever the profiler leaves**, so the
residual headroom is *invariant* to weight precision. FP8 weights free up
roughly 4.8 GiB compared to bf16 — and the engine spends all of it on KV.
Lighter weights do not buy free memory; they buy more KV cache at the same
free memory. Which is why the same allocation fails by the same margin in
both.

That also means the trade cannot be escaped by quantizing more. Anything
that frees memory is absorbed by the KV cache, so the only lever that
changes the outcome is the utilization target itself — deliberately not
touched here, since it defines the quantity F004 measures.

### The boundary: KV dtype, not weights and not speculation alone

The grid settles it. `dflash | bf16 | auto KV | eager` (cell
`a15aaeb7ae8ece66`) **boots** — 72s to healthy, KV 69808 tokens, and runs
at 74.6 / 84.3 tok/s at c=1, matching F021's eager measurements of the same
configuration. Non-speculative eager cells boot too (`6bf93e55`, `9d09fcab`).

So the failing set is precisely **speculative + `fp8_e4m3` KV + eager**:

| mechanism | KV dtype | eager | boots? |
|---|---|---|---|
| none | auto | yes | yes |
| none | fp8_e4m3 | yes | yes |
| dflash | auto | yes | **yes** |
| dflash | fp8_e4m3 | yes | **no** (x3) |
| dflash | fp8_e4m3 | no | yes |

### What the 892 MiB actually is

The allocation is identical in all three failures because it does not
depend on anything that varied:

```
max_num_seqs (256) x (5 speculative + 1 bonus) x vocab (152064) x 4 bytes
  = 891.00 MiB        observed: 892.00 MiB
```

A float32 logits buffer over the full speculative verification batch. It is
fixed by scheduler width and speculative width, which is why weight
precision moved it not at all.

## What this does NOT establish

- **Why `fp8_e4m3` KV specifically tips it over.** This is the honest gap.
  The logits buffer is constant, the utilization target is constant, and
  non-speculative FP8 KV cells boot fine under eager. Something about the
  FP8 KV path leaves less residual headroom than the bf16 path at the same
  utilization — plausibly FlashInfer FP8 workspace, or block-allocation
  granularity at doubled token capacity (F004) — but this study has not
  measured which, and the difference is not visible in the recorded
  capacities. Stated as unexplained rather than guessed at.

### Further limits

- **That it fails on other hardware.** This is a 22.03 GiB L4. A 40 or 80
  GiB card has slack to absorb the difference and would very likely boot
  both arms.
- **A general rule about which cells fail.** One cell is confirmed
  unbootable. The rest of the grid's eager arm is still running, and the
  boundary between bootable and not is not yet mapped.
- **That `--gpu-memory-utilization` could not work around it.** Lowering it
  would very likely let the cell boot. That was deliberately *not* done:
  utilization determines KV capacity, KV capacity is the quantity F004
  measures, and changing it per-arm would make the two arms incomparable on
  exactly the axis the study cares about.
- **Anything about acceptance.** This is a boot-time allocation failure; no
  tokens were generated.

## Consequence

F021's recommendation needs qualifying, and the paper's §4 with it. The
honest form is:

> `--enforce-eager` removes the bimodality and is faster at every
> concurrency tested **on configurations that can boot with it**. It
> increases KV allocation, so on memory-tight hardware it can convert a
> working speculative configuration into one that does not start.

That is a more useful result than an unqualified "use eager", because it
names the trade and the hardware condition under which it bites.

It also vindicates carrying `enforce_eager` as a grid **axis** rather than
pinning it. Had it been pinned, this cell would simply have disappeared
from the results with a one-word `oom` in a log, and the most likely
reading would have been that the configuration itself is unsupported on
SM89 — which is what F012 is about, and which would have been wrong.

## How to reproduce

```bash
modal run cloud/modal_probe.py --engine sweep --sweep perf_v2 --force
```

The failure is recorded in `results/boot_failures.jsonl` with the verbatim
error and the free memory observed before the boot.
