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

`dflash | bf16 weights | fp8_e4m3 KV | FLASHINFER | enforce_eager=True`
(cell `4ae0a860802f247c`) **cannot boot on a 22.03 GiB L4**. The same
configuration with CUDA graphs on (cell `5824814b68901229`) boots and has
12 recorded measurements.

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

## Reasoning

The intuition is that turning CUDA graphs off should *save* memory, since
nothing is captured. That is wrong here, and backwards for a specific
reason: the allocation being skipped is a **reservation**, not a
consumption. vLLM sizes the KV cache to fill whatever the profiler says is
available, so removing a reservation hands that memory to the KV cache
rather than leaving it free.

This makes the flag's effect configuration-dependent in a way the
throughput result alone does not reveal. F021 measured it on
`dflash | bf16 | auto KV`, which boots in both arms. The cell that fails
here pairs bf16 weights with FP8 KV, which is the heaviest combination in
the grid: FP8 KV raises the token capacity per byte (F004), so the engine
packs more KV into the same budget, and there is less slack to lose.

## What this does NOT establish

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
