# What You Are Actually Measuring When You Benchmark Speculative Decoding Under FP8

**Draft rev 1 · 2026-09-23.** Sections 1–5, 7 and 8 are written against
established findings. Section 6 is blocked on `sweeps/perf_v2.yaml`, in
flight. Every number carries a finding id; `analysis/check_provenance.py`
verifies each one resolves to raw records.

---

## Abstract

Speculative decoding and FP8 quantization are both standard in production
LLM serving, and their interaction is almost always reported from single
runs. We set out to measure that interaction on Ada-class hardware and
could not, three times, until the measurement apparatus itself was fixed.

We identify and quantify three confounds on vLLM 0.29.0 / NVIDIA L4 (SM89).
First, attention-backend selection is **conditioned on KV-cache dtype**, so
an FP8-vs-BF16 comparison under default settings silently swaps the
attention kernel as well; target and draft models select backends
independently, so pinning one does not pin the other. Second, speculative
throughput is **bimodal across boots** of an identical configuration,
varying up to 2.16× while acceptance does not; the mode is selected by the
CUDA-graph execution path and is fixed at boot. Third, the acceptance rate
τ has a **boot-to-boot noise floor of 1.32%** that no prior work reports,
below which "precision does not affect acceptance" claims are unfalsifiable
and above which small effects have been reported as real.

Each confound is individually large enough to produce the headline result
this study set out to report: the second alone spans a 1.16×–2.71× range in
measured speculative speedup. We then report FP8 × speculative goodput under
concurrency with all three controlled, and release the harness, raw records,
and the full history of our own corrections — six retractions, each
traceable to the same error shape.

---

## 1. Introduction

Speculative decoding trades a small draft model's cheap guesses against the
target model's expensive verification. Its benefit is governed almost
entirely by one quantity: **τ**, the mean number of tokens accepted per
verification step. FP8 quantization independently reduces weight and
KV-cache footprint, which matters because KV capacity is what binds under
concurrency. Serving stacks ship both. The obvious question — does
quantization degrade acceptance, and what does the combination buy under
load — is asked frequently and answered from single-run measurements.

We could not answer it. Three separate times, a result we were about to
report turned out to be produced by something other than the variable under
study. This paper reports those three things, because they are larger than
the effect they obscured.

**Contributions.**

1. **Backend substitution under dtype change** (§3). vLLM's `auto` backend
   selection is conditioned on KV-cache dtype, and target and draft select
   independently. A precision A/B under defaults is a kernel A/B as well.
2. **Bimodal speculative throughput** (§4). Identical configurations boot
   into a fast or slow mode and stay there. Spread reaches 1.64× between
   two boots at concurrency 16. `--enforce-eager` removes the bimodality and
   is faster at every concurrency tested.
3. **The acceptance noise floor** (§5). τ reproduces to 1.32% across boots.
   We report FP8's effect on τ as bounded *below* that floor rather than as
   a null result, and withdraw one of our own published claims that sat at
   1.7× the floor.
4. **Goodput with all three controlled** (§6), plus a compatibility matrix
   for FP8 KV across SM89 attention backends.
5. **A fully auditable artifact** (§2, §7): append-only raw records,
   script-regenerated derived tables, and every retraction preserved rather
   than removed.

We emphasise (5) because it is what made (1)–(3) findable. Eight of our
twelve documented corrections were caught by re-reading stored records
against a hypothesis formed later — impossible if results are summarised at
write time.

---

## 2. Setup

| | |
|---|---|
| Engine | vLLM 0.29.0, pinned; version read from the running server over HTTP |
| GPU | NVIDIA L4 (SM89, Ada, 24 GB), rented per-second |
| Target | Qwen3-4B |
| Draft | `z-lab/Qwen3-4B-DFlash-b16`, block size 16, 5 speculative tokens |
| Workload | GSM8K, sampled without replacement |
| Base image | `nvidia/cuda:13.0.3-devel-ubuntu24.04` |

L4 was chosen deliberately: it is the compute capability named in the
upstream reports that motivated the study, and at 24 GB the KV-capacity
constraint actually binds, which it does not on an 80 GB card.

**The harness never imports the engine.** It launches vLLM as a subprocess
and speaks HTTP. This costs some convenience and buys the guarantee that
what we measure is what a serving deployment runs, not an in-process
configuration that no server would produce.

**Every measurement is content-addressed.** A configuration hashes to a
`cell_id`; identical configuration yields an identical id on any machine.
Records are append-only, so a re-run supersedes rather than overwrites and
the superseded record remains readable. Three of our corrections depend on
that history being intact.

τ is defined as

> τ = 1 + accepted_draft_tokens / verification_steps

The `+1` is the bonus token, which vLLM's counter excludes (F001). Omitting
it understates τ by exactly 1 and silently changes every speedup ratio
derived from it.

---

## 3. Confound 1 — the attention backend moves when the dtype moves

**Finding.** vLLM's `auto` attention-backend selection is a function of
KV-cache dtype. At BF16 KV it selects FlashAttention-2; at `fp8_e4m3` it
selects FlashInfer (F003). An experimenter varying only precision therefore
varies the attention kernel too, and any resulting difference is jointly
attributable.

**It is worse than one substitution.** Target and draft models select
backends **independently** (F019). The selection happens at two distinct
call sites, and pinning the backend affects the target while the draft
continues to select on its own. A pinned-backend precision comparison is
therefore still confounded on the draft side for speculative cells — which
we record per cell rather than assert away.

**Compatibility is not uniform.** FP8 KV works on exactly two of four SM89
backends we tested, and `auto` selects one of the working two (F017). The
original hypothesis motivating this study — that FlashInfer's FP8 attention
kernels are SM90-only, so FP8 KV plus non-causal drafting would crash on Ada
— is **refuted** (F002). The combination runs.

**Our own error here.** We initially concluded that `--attention-backend`
was accepted, echoed back, validated against, and then ignored, and reported
this upstream. That was wrong: the flag controls the target, and what we
observed was the draft selecting independently. We corrected the report
(F018 → F019). We include this because the mistake is instructive: we
compared requested-vs-selected backend for *a server* when there is no such
thing as a server's backend — there are two.

---

## 4. Confound 2 — speculative throughput is bimodal across boots

**Finding.** Booting an identical speculative configuration repeatedly
produces throughput in two clusters, not a spread. Across six boots we
measured 28.4 and 33.2 tok/s against 50.7, 60.8 and 61.2 (F020, F021).
Acceptance was unchanged throughout, so this is the cost of producing
accepted tokens rather than how many are accepted.

**The selector is the CUDA-graph path.** vLLM emits, at every speculative
boot:

```
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for
attention backend FlashInfer
```

and falls back. Pinning `cudagraph_mode: PIECEWISE` reproduces the slow mode
exactly (30.04 vs 30.57 tok/s), indicating the fallback *is* the slow mode
rather than a third state. Disabling CUDA graphs with `--enforce-eager`
selects the fast mode reliably.

**It holds under load, and eager is also the stable arm.**

| concurrency | default | boot spread | eager | boot spread |
|---|---|---|---|---|
| 1 | 31.1 | 1.02× | 74.2 | 1.18× |
| 16 | 464.3 | **1.64×** | 776.7 | 1.07× |
| 64 | 1203.8 | 1.18× | 1648.2 | 1.04× |

**We deliberately report no speedup ratio from this table**, and the reason
is the finding itself. Measuring both arms of this configuration inside a
single run, with the default boot landing in the *fast* mode, gives
**1.13× / 1.06× / 1.05×**. Against a slow-mode boot the same eager numbers
give 2.19×. The ratio is a statement about which mode the comparison arm
drew, not about the flag.

What the table establishes is the spread: two boots of an identical
configuration differ by 1.64× at concurrency 16 under defaults, and by
1.07× with graphs off. Turning CUDA graphs off does not reliably make
speculative decoding *faster* — it makes it *measurable*, and whatever
throughput it gains is the slow-mode boots thereby avoided.

Our first draft of this section led with 2.38×, obtained by comparing
against an arm we assumed representative, which was one draw from the
bimodal distribution this same section describes. We flag it because it is
the paper's own thesis applied to the paper.

**Independent corroboration with a built-in control.** Our earlier
performance grid, collected before this hypothesis existed, contains the
effect with a control attached. Four configurations were measured across
two boots hours apart:

| cell | conc | spread | gap |
|---|---|---|---|
| non-speculative | 1 | 1.016× | 4.6 h |
| non-speculative | 16 | 1.105× | 2.7 h |
| speculative | 1 | 1.663× | 4.2 h |
| speculative | 16 | **1.624×** | 2.3 h |

Same host, same container lifecycle, same time gaps, same prompts; only
speculation differs. The non-speculative rows move ~1.0–1.1×, the
speculative rows 1.62–1.66×. The concurrency-16 figure, 1.624×, matches the
1.64× from the deliberate test above to within 1% — and was recorded before
anyone was looking for it.

**The fix is not free.** `--enforce-eager` increases KV allocation rather
than reducing it: vLLM reserves headroom for CUDA-graph capture, so
graphs-on runs at an effective `gpu_memory_utilization` of 0.8843 rather
than the requested 0.92, and removing graphs hands that reservation to the
KV cache. On our 22 GiB L4 this is enough to make one speculative
configuration fail to boot entirely, with a 892 MiB float32 logits
allocation failing against 745 MiB free on an otherwise clean card. The
recommendation is therefore conditional: eager removes the bimodality and
is faster at every concurrency we tested, on configurations that can boot
with it, and it costs headroom that memory-tight hardware may not have.

This is also why we carry the flag as a grid *axis* rather than pinning it.
Pinned, that cell would have vanished from the results behind a one-word
`oom`, and the natural reading would have been that the configuration is
unsupported on SM89 — a compatibility claim, and a false one.

**Why this matters beyond one engine.** A single-boot speculative benchmark
on this stack can report any speedup in a 1.16×–2.71× range depending on
which mode it happened to boot into. Single-run reporting is, from our
prior-art survey (§8), the norm.

---

## 5. Confound 3 — acceptance has a noise floor nobody had measured

**Finding.** τ reproduces to **1.32%** across boots of an identical
configuration with identical prompts (4.0913–4.1452 over four boots).

This number did not exist before we measured it, and its absence had
consequences in both directions.

**In the permissive direction.** We had reported that τ is invariant to FP8
precision, measuring a 0.7% difference. With the floor known, the correct
statement is stronger and more precise: *no effect larger than 1.3% is
detectable*, and the observed 0.7% sits below it. A null result becomes a
bounded one.

**In the restrictive direction.** We had also reported that backend choice
perturbs τ more than precision does, citing a 2.25% spread across four
backends. That is 1.7× the noise floor, from single boots. It is **not
separable from noise**, and we withdraw it (F006, amended).

That claim moved three times before anyone measured the floor: asserted,
withdrawn for a wrong reason (F018), restored (F019), and now withdrawn for
a sound one. The reason it kept moving is that every argument about it was
about the effect, and none was about the floor. One four-boot run settled it
permanently and cost about an hour of GPU time.

**The general prescription is cheap.** Measure the noise floor of the
quantity being compared *before* comparing it. In our experience this is the
single highest-yield methodological step, and it is routinely skipped
because the floor feels like overhead rather than like a result.

---

## 6. Results with all three controlled

The grid carries `enforce_eager` as an **axis** rather than a pinned
setting: 16 boots, both arms of every configuration measured in one run.
Pinning it would have been our fourth instance of the same error, and the
measurement shows the error would have been large.

### The execution-path flag is not a free control

| config | conc | graphs on | graphs off | off/on |
|---|---|---|---|---|
| none \| bf16 \| auto | 1 | 26.9 | 25.6 | 0.95× |
| none \| bf16 \| auto | 64 | 835.0 | 797.5 | 0.96× |
| none \| **fp8** \| auto | 1 | 41.3 | 20.3 | **0.49×** |
| none \| **fp8** \| auto | 64 | 1252.2 | 825.4 | **0.66×** |
| none \| **fp8** \| fp8_e4m3 | 1 | 42.3 | 21.2 | **0.50×** |
| dflash \| bf16 \| auto | 1 | 70.0 | 79.5 | 1.13× |
| dflash \| **fp8** \| auto | 1 | 118.1 | 72.0 | **0.61×** |

`--enforce-eager` costs roughly **half the throughput** of any FP8-weight
configuration and costs BF16 essentially nothing (F024). Individual boots
do not overlap on any FP8 row and overlap on every BF16 row. τ is unmoved
throughout, so this is the cost of producing tokens, not acceptance.

Two further constraints on the same flag: it cannot boot speculative cells
with FP8 KV at all on this card, failing on a 892 MiB float32 logits buffer
that is fixed by scheduler and speculative width (F023); and its apparent
throughput benefit on the cells where it *does* help is a statement about
the comparison arm's mode rather than about the flag (§4).

Had we pinned it — the obvious response to §4 — every FP8-weight baseline
would have been halved, every speculative FP8-KV cell would have failed to
boot, and the speculation-vs-baseline speedups would have been inflated by
roughly 2× on exactly the cells this paper is about, with nothing in the
output to indicate it.

### The pre-registered prediction, and how it did

We recorded a prediction before the grid ran: that the apparent ~3×
FP8-weight speculative advantage was mostly mode assignment, that it would
survive but shrink to roughly 1.2–1.4×, and that FP8 rows would move least
between arms.

- **(1) holds.** BF16 speculative reaches 79.5 tok/s with graphs off, in the
  predicted 70–80 band.
- **(3) holds**, but for the wrong reason: FP8 rows move *most* in absolute
  terms, downward, because eager penalises them (F024).
- **(2) fails.** With graphs on the FP8-weight speculative advantage is
  **1.69×**, not the ~3× the single-boot data suggested and not the
  1.2–1.4× predicted; with graphs off it **inverts to 0.91×**. The mode
  story explains part of the original 3× and F024 explains the inversion.

We report this as measured. The prediction was written down so it could
fail in public, and one of three did.

### Remaining results

Compatibility matrix (F002, F017), KV capacity under FP8 (F004), and task
accuracy at n=256 with McNemar exact tests (F016) are unchanged by the
re-measurement and are carried from the earlier phases. Goodput under a
declared SLO across the concurrency sweep is computed offline from the
stored per-request rows (`analysis/goodput.py`), so the SLO threshold can
be swept without further GPU time.

---

## 7. Threats to validity

- **One GPU, one model, one engine version, one draft mechanism.** Every
  result is SM89 / Qwen3-4B / vLLM 0.29.0 / DFlash. We make no claim about
  SM90, about other draft mechanisms, or about other engines.
- **The mode selector is unidentified.** We show *that* the CUDA-graph path
  selects the mode and that it is fixed at boot; we do not show *why* a
  given boot lands where it does.
- **The draft-side backend confound is acknowledged, not removed** (§3).
- **Our retraction list covers errors we caught.** The same blindness that
  produced them applies to finding them. We assume undiscovered instances
  exist; one was found the same day we wrote the list.
- **No base rate.** Comparable studies do not report their retractions, so
  our count is evidence of legibility, not of unusual error-proneness.

Every finding in `findings/` carries a mandatory *"What this does NOT
establish"* section, and this paper inherits those limits verbatim rather
than softening them.

---

## 8. Related work

| Work | What it did | Gap this addresses |
|---|---|---|
| *Spec Decoding Meets Quantization* (2505.22179) | EAGLE-2 × W8A8/W4A16/W4A8, single-request | No FP8, no KV-cache quantization, not batched |
| SpecKV (2605.02888) | Adaptive γ under compression | Algorithmic, step-level, not serving goodput |
| QSpec (EMNLP 2025) | W4A4 draft + W4A16 verify, shared weights | Shared-weight special case, not FP8 |
| QuantSpec (2502.10424) | Self-speculation, 4-bit hierarchical KV | A method, not a cross-mechanism measurement |
| ML-SpecQD, Quasar | Quantized drafts, W8A8 verification | Neither reports availability or variance |
| vllm#41559 | Bug report and fix | Issue tracker, not a study; no Ada data, no τ, no goodput |

None reports boot-level variance, and every single-request result in this
table is vulnerable to the confound in §4.

---

## 9. Artifact

Harness, raw records, derived tables and findings:
`github.com/akshathtiwari/spec-fp8-study`.

```bash
python analysis/check_provenance.py
```

Verifies that every finding's citations resolve to stored records, that
stored ids still recompute from their stored configs, and that no result
rests on terminal output. It exits with the defect count and currently
reports the gaps it finds rather than hiding them.

```bash
python analysis/budget_report.py
```

Reports GPU-seconds and cost by phase. The figure is stated as a **floor**:
container start and image pull fall outside the timed region, and four GPU
entrypoints went unbilled until 2026-09-23, including the runs behind §4.
The final number is to be filled in from the log before submission rather
than from recollection.

The scale is the point. This study was run on rented per-second L4 time at a
cost in the tens of dollars. The error analysis in §3–§5 is what that budget
can actually establish, and on the evidence of §8 it is what the field is
missing — not another single-run speedup number measured on hardware most
readers cannot rent.
