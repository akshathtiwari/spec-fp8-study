# Related work — reading notes

**rev 2 · 2026-09-26**

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

## The Silent Hyperparameter (arXiv:2605.19537)

Pape, Evertz, Schönherr. CISPA Helmholtz Center. Submitted 2026-05-19,
revised 2026-05-20. **Primary cs.LG.**

**Setup.** Five engines — vLLM, SGLang, llama.cpp, LMDeploy, Ollama — plus
HuggingFace transformers as reference. Models include **Qwen3-4B**, our own
target, plus Llama-3.1-8B, Qwen3-30B, DeepSeek-R1-Distill-Qwen-7B. **H100
primary, also NVIDIA L40 — Ada Lovelace, SM89, our architecture.** FP16
throughout; no quantization, no speculative decoding. **Twelve seeds per
configuration.**

**Findings.** Backend choice alone shifts GSM8K accuracy by 16.60 percentage
points between best and worst engine. Divergence attributed to custom kernels
and engine defaults in logit processing — LMDeploy's multi-threaded Top-K
kernel has a race condition that breaks ties arbitrarily — and to FP32
accumulation differences in llama.cpp/Ollama.

**The number that matters to us.** *"disabling CUDA graphs shifted
performance across engines by up to +0.15%."* On accuracy. They measure no
throughput and no latency anywhere in the paper.

**Relationship.** This is the strongest external support for §4 in the
literature we have found, and it arrives by reporting a null. A careful
multi-seed cross-engine reproducibility study, on our model and our GPU
architecture, toggled the exact switch our §4 concerns and found 0.15% —
because the quantity it measured was output agreement and not throughput. We
measure 13.92% CV across the same switch. Two quantities, one switch, and
only one of them had been looked at.

Their recommendation — avoid single evaluation runs, average across seeds —
is **the same prescription as our §5**, reached from a third direction. With
Kaplan that makes three independent concurrent arrivals, and §5 now says so.

---

## The Integer Alibi (arXiv:2608.13756)

Teng-Ruei Chen, Krixvon (Taipei). Submitted 2026-08-13, revised 2026-08-18.
Primary cs.LG.

**Setup.** **NVIDIA RTX 4090, SM89 — same architecture as our L4.** vLLM
0.27.1 pinned by container digest, driver 580.173.02. Compares vLLM's
`CutlassInt8ScaledMMLinearKernel` against `TritonInt8ScaledMMLinearKernel`.
GEMM only; attention backends are not examined.

**Findings.** The two nominally interchangeable INT8 kernels agree on no
sequence in 64 end-to-end comparisons. Divergence localised to scale
application and output rounding after the INT32 accumulator, proved exact;
validated by bit-identical results under power-of-two scales. INT8
differences stay at 1.9–7.6 ppm across K=512–32768; **FP8 differences rise
from 8.2% to 52.7% across the same range.** No throughput, accuracy or
calibration consequences measured.

**Their limitations, in their words.** Single GPU generation, model family and
engine version. 2.9% residual variance unexplained outside the INT8 epilogue.
Prefill regime M=512 only — **decode (M=1) untested**, which is where all of
our work lives. No positive controls.

**Relationship.** Two things. First, their FP8 result is *element-level
disagreement between implementations*, and our accuracy table is an
*end-to-end task null at n=256*. These do not contradict; they do not even
address each other, and the accuracy section now says so explicitly rather
than letting a reader assume our null covers numerics. Second, and more
useful: each of their arms reproduces **bit-for-bit across two cold
restarts**. Byte-identical output across boots, on SM89, in vLLM. Our
throughput does not reproduce across boots on the same architecture. Both
facts are true of this stack and neither implies the other, which is a
cleaner framing of §4 than we had.

---

## Identifying and Mitigating Systemic Measurement Bias in Production LLM Inference Benchmarks (arXiv:2605.24217)

Chandrasekar, Kramberger. Google. v1 2026-05-22, v2 2026-05-26. Primary
cs.AI, cross-list cs.DC.

**Setup.** Eight client tools including vLLM Bench, GuideLLM, Inference Perf,
NVIDIA AI Perf, k6, Locust, MLPerf. Driven against `llm-d-inference-sim`, a
**zero-latency simulator**, so any degradation is provably a client artifact.
Client hosts c4-standard-144 and e2-medium.

**Findings.** Modelling the client as an M/G/1 queue, wait time diverges as
utilisation approaches 1. **Onset around 1000 QPS**, where single-process
clients saturate at 146–443 QPS; TTFT overhead from 8 ms to 58 s; vLLM Bench
processed 75,574 tokens where a multi-process client processed 545,733, a
7.2× discrepancy. Define NTPOT = end-to-end latency / output tokens, to
amortise prefill and queueing. **Concurrency 64 is never discussed as a
distortion threshold.** No speculative decoding, no chunked streaming.

**Relationship, and a correction.** On an abstract-level read this looked
like a threat to §6: our goodput result is computed at concurrency 64 from a
Python async client, and if the client were queuing, the ITL distribution
would carry client delay. The full text does not support that. Their onset is
1000 QPS; at concurrency 64 with generations of hundreds of milliseconds we
are one to two orders of magnitude below it, and SPEED-Bench independently
places GIL effects at BS>256.

So this is related work rather than a defect — but it is the same *class* of
error as §6, where the client's view of the stream rather than the server
produces the misleading number, and it is now cited as such. We have added a
threats bullet conceding that we argued this from load regime and never ran a
client-saturation control against a mock server, which is the measurement
that would settle it.

This entry also records a process failure: the abstract-level read produced a
confident "this threatens §6" that the full text refuted. That is the third
time in this study that reading an abstract produced a wrong answer, in a
study about quick checks producing wrong answers.

---

## Method note

These notes were taken from the papers' full text where available, not from
abstracts. An earlier pass in this study asserted "none of our confounds is
scooped" on the basis of a summary of an abstract page — a quick check, in a
study whose thesis is that quick checks produce wrong answers. That assertion
happened to survive scrutiny; it was not entitled to.
