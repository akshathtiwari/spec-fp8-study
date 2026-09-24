# Related work — reading notes

**rev 1 · 2026-09-24**

Notes from reading the closest prior work, recorded because the paper cites
figures from these papers and `analysis/check_paper.py` requires every figure
in the paper to resolve to a record. An externally-sourced number needs a
record too; otherwise "traceable" means only "traceable to our own runs".

It also exists because the reading itself was a finding. Three of these
papers were missing from the related-work section until 2026-09-24, and one
of them is an MLSys oral on an adjacent question. A literature review that
happens only in conversation is not part of the artifact.

---

## Speculative Decoding: Performance or Illusion? (arXiv:2601.11580)

Liu, Yu, Park, Stoica, Cheung. MLSys 2026 oral. Primary cs.CL.

**Setup.** vLLM v0.10.1.1, NVIDIA H100 80GB (4x for 70B/106B). FP16 KV cache.
All default vLLM optimizations enabled, explicitly including CUDA graphs.
FlashAttention-3 for chain configurations, FlashInfer for tree.

**Findings.** Verification dominates runtime, 42–95% across variants.
Increasing batch size improves absolute throughput but systematically reduces
SD's *relative* speedup, amplified for larger models. Acceptance varies at
three levels: within a request, across requests, across datasets. Oracle
experiments suggest up to 2.2x further headroom from adaptive method
selection.

**Checked against our four confounds.**

| Our confound | Present in their work? |
|---|---|
| Backend selection conditioned on KV dtype | No — FP16 KV throughout; dtype never varied |
| Boot-to-boot throughput variance | No — no repeated-run variance reported |
| $\tau$ measurement noise floor | No — they measure *variation of acceptance*, not noise of the measurement |
| ITL as inter-chunk latency | No — they use token throughput, not per-token latency |

**Relationship, stated carefully.** Their acceptance-variability result is
about the quantity varying with context; ours is about a repeated identical
measurement varying. Complementary, and easy to conflate. Their batch-size
result is genuine prior art for load-dependence and we claim no novelty near
it. They enable CUDA graphs on H100, which is the setting our own confound 2
concerns — though we measure L4 and localise the effect to draft-model
speculation, so this is a question rather than a criticism.

---

## An Interpretable Latency Model for Speculative Decoding in LLM Serving (arXiv:2605.15051)

Kong, Flynn, Peng, Shavit, Kurtz, Marques. Primary cs.LG, secondary cs.PF.

**Setup.** vLLM 0.13.0 driven by GuideLLM 0.5.2. Single A100 SXM (4x/8x for
larger models), also validated on H100. Llama-3.1-8B/70B, gpt-oss-20b, Qwen3
family including MoE.

**Contribution.** A latency model inferring effective batch size from request
rate via Little's Law, decomposing per-request demand into load-independent
and load-dependent components: $L = C_1 / (1 - \text{RPS} \times C_2)$.
Goodness of fit reported as $R^2$, improving from **0.902 to 0.997** for
gpt-oss-20b.

**Methodology.** A synchronous baseline, a throughput benchmark to find the
highest stable request rate, then eight evenly spaced constant-RPS
benchmarks. **Single sweep per configuration.** No repeated runs, no error
bars, no confidence intervals, no variance reported. No mention of CUDA
graphs or eager mode anywhere. FP8 and quantized KV are not experimental
variables. Focuses on the pre-saturation regime, discarding the throughput
ceiling.

**Relationship.** This is the paper our confound 2 bears on most directly: a
model fitted to single-sweep vLLM SD measurements, to three-decimal $R^2$,
with no term for boot-to-boot variation. We do not claim the fit is wrong —
different hardware, and our effect is localised to draft-model speculation —
only that the term is unmeasured.

---

## A Calibrated Instrument for Measuring How Inference Optimizations Affect Output Quality (arXiv:2609.18005)

Jerry Kaplan. Submitted 2026-09-16. Primary cs.CL, secondary cs.LG.

**Method.** Scores outputs with an LLM judge, then *calibrates the judge*:
compares its scores on "two ordinary runs of a model given the same prompts",
verifies no systematic preference between statistically equivalent outputs,
and measures its per-sample noise. Each design includes a null condition
"provably identical in distribution to the unmodified model, whose measured
difference must be zero".

**Findings.** 4-bit indistinguishable from 16-bit at ±0.3 resolution. At
3-bit: 0.5 points lost in English prose, 0.9 in Chinese, 1.1 on multi-step
math. Domain-dependent costs for early-exit. Results vary by model provider.

**Relationship, and why it matters to us.** The null condition is the same
epistemic move as our noise floor: run the identical thing twice, and treat
the observed difference as the resolution limit. Published eight days before
we wrote our §5. **Our prescription is therefore not novel as a principle**,
and the paper now says so. His instrument is an LLM judge and his quantity is
output quality; ours is a serving harness and our quantities are acceptance
and throughput. Concurrent independent arrival at the same principle in
adjacent subfields is evidence it is right.

---

## SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding (arXiv:2604.09557)

Abramovich, Ashkenazi, Putterman, Chislett, Mitra, Darvish Rouhani,
Zilberstein, Geifman. Submitted 2026-02-10, revised 2026-05-28. Primary
cs.DC, cross-list cs.AI.

**What it is.** A benchmark suite for standardising speculative-decoding
evaluation, integrating with production engines including vLLM and
TensorRT-LLM. Two data splits: a curated qualitative split emphasising
semantic diversity, and a throughput split supporting evaluation across
varying concurrencies.

**Findings of note.** Synthetic inputs overestimate real-world throughput.
Optimal draft length depends on batch size.

**Checked against our four confounds.**

| Our confound | Present? |
|---|---|
| Backend selection conditioned on KV dtype | No |
| Run-to-run dispersion for identical configurations | No |
| Acceptance dispersion under repeated identical boots | No |
| ITL as inter-chunk latency | No |
| SLO-attainment goodput | No |

**Why it matters to us.** This is the closest thing to a community standard
for the question we are measuring, and it was missing from our related work
until 2026-09-24. Its absence was the same shape as the TurboSpec omission:
a paper doing adjacent work in the same engine, not found because the
literature search was not systematic. That a standard benchmark does not
report run-to-run dispersion is itself support for §4 -- it is one more
place the error term would go unnoticed.

---

## Method note

These notes were taken from the papers' full text where available, not from
abstracts. An earlier pass in this study asserted "none of our confounds is
scooped" on the basis of a summary of an abstract page — a quick check, in a
study whose thesis is that quick checks produce wrong answers. That assertion
happened to survive scrutiny; it was not entitled to.
