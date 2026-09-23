# Paper outline — rev 1 · 2026-09-23

## The framing has moved, and this records why

`requirements.md` §2 planned two headlines. H1 was refuted (F002), which puts
us on **headline B**: "τ under FP8 across draft mechanisms, weight-quant vs
KV-quant separated, goodput under concurrency."

**Headline B is now the weaker paper.** Not because the measurements failed,
but because of what they turned up on the way. Three separate times, a result
this study was about to report turned out to be produced by something other
than the variable under study:

| # | Confound | Size | Found by | Would have produced |
|---|---|---|---|---|
| 1 | Backend selection is conditioned on KV dtype; target and draft select independently | changes the kernel, not a percentage | F003, F019 | "FP8 KV is slower" — actually a different attention backend |
| 2 | Speculative throughput is bimodal across boots via the CUDA-graph path | 2.16x range; 1.64x between two boots | F020, F021 | Any speculative speedup between 1.16x and 2.71x, at will |
| 3 | τ had no measured noise floor | 1.32% | F020 → F006 | A 2.25% "backend affects acceptance" effect at 1.7x noise |

Each one is individually large enough to generate the headline result. Number
2 alone spans the entire effect size the paper was going to report.

A "FP8 × speculative decoding is N% faster" paper measured on one rented L4
is a weak paper: small hardware, one model, one engine, and a number nobody
can check. A paper that says **"here is why most such numbers are
unreliable, and here is each error term measured"** is a strong paper from
exactly this budget, because the contribution is the error analysis and the
error analysis is what one careful GPU-month can actually establish.

This also converts the study's six retractions from a liability into the
evidence base. They are not confessions; they are the three confounds being
caught, which is the paper.

---

## Proposed thesis

> Benchmarks of speculative decoding under quantized inference are dominated
> by measurement artifacts rather than by the quantization. We identify and
> quantify three — dtype-conditioned attention-backend selection, bimodal
> CUDA-graph execution modes across boots, and an unmeasured acceptance-rate
> noise floor — on vLLM 0.29 / Ada SM89, and show that each is individually
> large enough to account for a reported speedup. We then give the FP8 ×
> speculative measurements with all three controlled.

Working title candidates:

1. *What You Are Actually Measuring When You Benchmark Speculative Decoding
   Under FP8*
2. *Three Confounds in Speculative Decoding Benchmarks*
3. *Boot Variance, Backend Substitution, and the Acceptance Noise Floor*

Prefer (1): it names both halves, and the second half is what makes it
findable by people searching FP8 + speculative decoding.

---

## Section plan

**1. Introduction.** Speculative decoding and FP8 are both standard in
serving. Their interaction is reported from single runs. We set out to
measure it and could not, three times, until the measurement itself was
fixed. State the three confounds and their sizes up front.

**2. Setup.** vLLM 0.29.0, L4 (SM89, 24GB), Qwen3-4B, DFlash b16 draft,
GSM8K. HTTP-only harness (never imports the engine). Content-addressed cells,
append-only records, everything in the repo. §Reproducibility points at
`analysis/check_provenance.py`.

**3. Confound 1 — the backend moves when the dtype moves.** F003, F017, F019.
`auto` serves BF16 KV on FlashAttention-2 and FP8 KV on FlashInfer, so a
precision A/B silently swaps the kernel. Target and draft select
*independently* (F019, `cuda.py:432` vs `:492`), so pinning the target does
not pin the draft. Includes the retraction (F018): we claimed the flag was
ignored, posted that upstream, and were wrong.

**4. Confound 2 — speculative throughput is bimodal across boots.** F020,
F021. 2.16x across six boots at constant acceptance; bimodal, not continuous;
selected by the CUDA-graph path; `--enforce-eager` removes it (boot spread
1.64x → 1.04–1.07x) *and* is faster at every concurrency (2.38x / 1.67x /
1.37x at c = 1 / 16 / 64). **Strongest single result in the study.**

**5. Confound 3 — acceptance has a noise floor nobody measured.** F006 as
amended. τ boot-to-boot is 1.32% for an identical configuration. The FP8
precision effect (0.7%) sits *below* it — so "τ is invariant to FP8" is a
bounded statement, not a null result. A previously-reported 2.25% backend
effect sits at 1.7x noise and is withdrawn. Note that this claim moved three
times before anyone measured the floor.

**6. Results with all three controlled.** Compatibility matrix (F002, F017),
KV capacity (F004), accuracy (F016, with the power gap F008 stated), goodput
under concurrency (the re-measured grid).

**7. Threats to validity.** One GPU, one model, one engine version, one draft
mechanism for the speculative results. §What this does NOT establish is a
mandatory section in every finding and the paper inherits those limits
verbatim rather than softening them.

**8. Related work.** Position against requirements §9. The specific gap: none
of them report boot-level variance, and all of the single-request results are
vulnerable to confound 2.

---

## What must still be true before this is submittable

- [ ] The re-measured goodput grid, with the execution path controlled
      (blocked on the `mechanism=none` eager fairness check).
- [ ] Confound 1 needs a quantified cost, not just a mechanism. We show the
      backend *changes*; we should show what that change is worth in tok/s,
      or state plainly that we did not measure it.
- [ ] Confound 2 is one cell (dflash / FLASHINFER / bf16). A second cell —
      different KV dtype, or ngram instead of dflash — decides whether it is
      a property of the config or of speculative decoding on this engine.
      This is the single highest-value remaining experiment.
- [ ] F008 (accuracy underpowered at n=16) is superseded by the n=256 run;
      confirm the paper cites the powered numbers.
- [x] ~~Check whether F020's non-speculative stability claim is
      concurrency-dependent.~~ **Withdrawn 2026-09-23: the comparison was
      invalid.** It read tonight's cell `28971d07` against the prior grid's
      `none|bf16|fp8_e4m3` rows and called the 1.15x gap at c=64 a
      reproducibility problem. But `28971d07` is `enforce_eager=True` and
      the prior rows are `enforce_eager=False`. That is an arm comparison,
      not a repeat measurement, so it says nothing about stability.

      What it may actually say is the opposite, and it bears on the
      fairness question the axis exists to answer: graphs-off was 0.97x at
      c=1, 1.03x at c=16 and 1.15x at c=64 against graphs-on, i.e.
      `--enforce-eager` does **not** appear to cost the non-speculative
      baseline and may help it under load. Still cross-session and one boot
      per arm, so it settles nothing; `analysis/cudagraph_arms.py` does the
      within-grid paired version.

      Recorded as F022 instance 14. Third time in one day that a comparison
      was made across the `enforce_eager` axis while assuming it fixed.

---

## Pre-registered prediction for perf_v2 (recorded 2026-09-23, before the run)

Written down in advance so it cannot be retrofitted to whatever the grid
returns. If the measurement contradicts this, the contradiction gets reported
as-is.

The existing Phase 2 speculative rows at concurrency 1 split cleanly on
weight precision:

| mech | weights | kv | tok/s |
|---|---|---|---|
| dflash | bf16 | auto | 36.3 |
| dflash | bf16 | fp8_e4m3 | 30.6 |
| dflash | fp8 | auto | 109.6 |
| dflash | fp8 | fp8_e4m3 | 97.4 |

Read naively this is "FP8 weights make speculative decoding ~3x faster",
which is close to the withdrawn 1.16x-vs-2.71x interaction (F020).

**It is probably mostly mode assignment.** F021 measured *the same bf16 cell*
at **74.2 tok/s** with CUDA graphs off. So the bf16 rows sit in the slow mode
and the fp8 rows do not. Each of these cells is a single boot, and F020
established that a single boot cannot tell you which mode you got.

Concretely, the prediction is:

1. With graphs off, **all four** speculative cells land in fast mode, and the
   bf16 rows rise to roughly 70-80 tok/s rather than staying near 30.
2. The FP8-weight advantage **survives but shrinks sharply** — from ~3x to
   roughly 1.2-1.4x. It should not vanish: fewer weight bytes is a real
   bandwidth saving, and 97-110 already exceeds the 74.2 that bf16 reaches
   in fast mode, which is hard to explain by mode alone.
3. The **fp8-weight rows move least** between arms, since they appear to be
   in fast mode already.

If (1) holds and (2) does not — if the gap stays near 3x with graphs off —
then FP8 weights genuinely do carry a large speculative advantage and the
mode story does not explain this split. That would be a real result, and it
would mean the withdrawal in F020 was too aggressive.

This is also the sharpest available test of whether making `enforce_eager` an
axis was worth the extra eight boots.

## Outcome of the pre-registered prediction (recorded 2026-09-23, after two grids)

Written here beside the prediction so the pair can be read together, and
because `analysis/out/tables/cudagraph_arms.md` only ever holds the most
recent grid.

| | grid 1 | grid 2 |
|---|---|---|
| bf16 speculative c=1, graphs off | 79.5 tok/s | 83.6 tok/s |
| FP8-weight speculative advantage, graphs **on** | **1.69x** | **1.90x** |
| FP8-weight speculative advantage, graphs **off** | **0.91x** | **1.09x** |
| bf16 arm movement between arms | 1.13x | 1.32x |
| fp8 arm movement between arms | 0.61x | 0.76x |

- **(1) HOLDS** in both grids: 79.5 and 83.6, inside the predicted 70-80
  band or just above it.
- **(2) FAILS** in both grids, instructively. The ~3x apparent advantage
  does shrink, but to 1.7-1.9x rather than the predicted 1.2-1.4x, and the
  further collapse to parity under graphs-off is a *second* effect (F024,
  the eager penalty on FP8 weights) that did not exist in the model when
  the prediction was written. Two effects, one of them unknown at the time,
  landing near a band guessed for the other.
- **(3) HOLDS** in both grids but for the wrong reason: fp8 rows move
  *most* in absolute terms, downward, because eager penalises them.

One of three held cleanly. The prediction was recorded so it could fail in
public, and the way it failed — by being right about a direction for a
mechanism that turned out not to be the operative one — is worth more than
a clean hit would have been.

## Deliberately not claimed

- Nothing about SM90, since the study ran on SM89 only.
- No claim that `--enforce-eager` should be the default. We measured one
  configuration; the flag disables CUDA graphs engine-wide and the
  non-speculative cost is measured separately, not assumed.
- No causal account of *why* the CUDA-graph path picks a mode. Unidentified,
  and stated as open (F021).
