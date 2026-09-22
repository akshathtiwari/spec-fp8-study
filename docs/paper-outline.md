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

## Deliberately not claimed

- Nothing about SM90, since the study ran on SM89 only.
- No claim that `--enforce-eager` should be the default. We measured one
  configuration; the flag disables CUDA graphs engine-wide and the
  non-speculative cost is measured separately, not assumed.
- No causal account of *why* the CUDA-graph path picks a mode. Unidentified,
  and stated as open (F021).
