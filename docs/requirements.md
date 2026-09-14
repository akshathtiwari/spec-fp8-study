# Requirements — Speculative Decoding × FP8 Compatibility Study

**Akshath Tiwari · independent · spec 2026-09-14 · rev 2 (headline rebased after reading
[vllm#41559](https://github.com/vllm-project/vllm/issues/41559) in full)**

---

## 0. The idea in plain words

*(This section exists so the doc is readable months later, and by anyone who isn't
already inside this subfield. The precise version starts at §1.)*

**Speculative decoding.** Normally a language model writes one token at a time, and each
token costs a full pass over a huge model. Speculative decoding adds a small, fast "draft"
model that guesses the next few tokens. The big model then checks all the guesses in a
single pass. Correct guesses are kept for free. So the whole speedup depends on one
number: **how often the guesses get accepted.** Call it τ (tau), the average number of
tokens accepted per check.

**Quantization.** Storing numbers in fewer bits. FP8 means 8 bits instead of the usual 16.
You can do this in two separate places, and they are genuinely different:

- **Weights** — the model's learned parameters.
- **KV cache** — the running memory of the conversation so far. This is what actually
  fills up your GPU when you serve many users at once.

**Why the two interact.** Quantizing the big model slightly changes its answers. The draft
model was tuned to predict the *unquantized* big model, so its guesses now get rejected
more often and τ drops. But quantizing also frees memory, which lets you serve more
requests simultaneously, which raises throughput. **Two effects pulling in opposite
directions, and nobody has published where they cross.**

**The twist we found.** One family of draft methods (DFlash and similar) proposes a whole
*block* of tokens at once. To do that, its attention has to look across the entire block
in both directions. That is called **non-causal** attention. Ordinary text models only
look backwards, which is **causal**. It turns out the code that lets a KV cache be stored
in FP8 was written only for the causal case.

For a while this meant DFlash simply could not use an FP8 KV cache in vLLM at all. That
got fixed in mid-2026. But the fix appears to be **gated to one GPU family**: the routine
that converts FP8 back into normal numbers only switches on for SM100 (datacenter
Blackwell, e.g. B200). Meanwhile the backend happily *accepts* the configuration on
anything from SM80 up to SM121 — which includes SM89, the Ada cards (RTX 4090, L4, L40S)
that independent people and small teams actually rent.

Someone reported exactly this failure on SM121 in July 2026, after the fix. **Nobody has
checked SM89.**

**So the study asks:** does "FP8 KV cache is supported with speculative decoding" hold on
the hardware people actually deploy on, and if not, what does the forced fallback to
16-bit KV cost you once you are serving more than one user at a time?

---

## 1. Context

Production serving stacks speculative decoding and FP8 quantization together. The
published literature evaluates them separately, or at INT4/INT8. A prior-art survey (§8)
shows the general question "does quantization reduce acceptance" is already answered.
What is **not** answered:

1. **FP8 specifically.** Every prior study uses INT4/INT8/GPTQ/SmoothQuant. The nearest
   work, *Speculative Decoding Meets Quantization* ([2505.22179](https://arxiv.org/html/2505.22179v1)),
   tests no FP8 and **explicitly excludes KV-cache quantization**.
2. **Whether FP8 KV quantization actually works per mechanism, per GPU family.** Support
   is claimed upstream but appears capability-gated (see §2).
3. **Batched serving.** 2505.22179 measures single-request latency; SpecKV
   ([2605.02888](https://arxiv.org/abs/2605.02888)) works at step level. Nobody reports
   goodput under an SLO across a concurrency sweep, which is where lost KV capacity binds.

### Provenance rule (hard)

Nothing from the author's employer enters this artifact — not numbers, not workload
descriptions, not configuration, not "in prior production experience we observed…".
Employer knowledge may direct *where to look*; only publicly reproducible evidence is
cited. Every result must be re-derivable from this repo by a third party on rented
hardware.

---

## 2. The claim is PROVISIONAL — week 1 decides it

**Do not write the paper's framing until R1/R2 return.** Rev 1 of this spec asserted that
DFlash is *structurally* locked out of KV quantization. Reading the full issue thread
falsified that. The corrected state of knowledge:

| Fact | Evidence | Status |
|---|---|---|
| DFlash + FP8 KV failed in vLLM v0.20.0 on all backends | issue body, May 2026 | true **then**, on a **non-stock** image |
| It was never architecturally impossible | llama.cpp/ggml shipped DFlash + quantized KV on the same 3090, 5.46×, 256K ctx in 24GB | **settled** — drop any "structural" language |
| vLLM added support | PR #39995, PR #43081 (`supports_non_causal()`, `_noncausal_prefill_wrapper`), closed `completed` 2026-08-11 | **fixed upstream** |
| The fix is capability-gated | `trtllm_prefill_attn_kvfp8_dequant` gated to `is_device_capability_family(100)`; `supports_compute_capability` accepts `>= (8,0) and <= (12,1)` | reported via code reading |
| It still crashes on SM121 after the fix | illegal memory access, GB10, July 2026 | reported, one user |
| **Behaviour on SM89 (Ada / RTX 4090 / L4 / L40S)** | — | **UNKNOWN. This is the study's pivot.** |

### Hypothesis H1 (to test in week 1)

> SM89 falls inside the accepted compute-capability range but outside the SM100 dequant
> gate, so FP8 KV cache combined with a non-causal speculative drafter is accepted at
> config time and then fails or silently misbehaves on Ada hardware.

### Branch on the result

- **H1 confirmed → headline A.** *"FP8 KV cache with non-causal speculative drafting is
  supported on datacenter Blackwell and broken on the consumer hardware people actually
  rent, because the dequant path is capability-gated."* Live, unreported, and the
  maintainer's closing note explicitly invites it: *"If specific feature support is
  desired, please create fine-grained issues for each."*
- **H1 refuted (it works fine on Ada) → headline B.** Fall back to the measurement study:
  τ under FP8 across four draft mechanisms, weight-quant vs KV-quant separated, goodput
  under concurrency. Still novel on the FP8 and KV axes. Compatibility matrix demotes to
  a supporting table.
- **Either way the experiments in §4 are the same.** Only the framing moves. This is why
  R1/R2 gate everything else.

### Evidence-quality caveats (do not over-cite this thread)

- The original report used a **custom image** (`...vllm-openai:v0.20.0-tq-hybrid-v2`);
  `turboquant` is not a mainline backend. Not stock vLLM.
- Original hardware was a 3090 (Ampere SM86), target a 27B int4 model. Neither matches
  our setup.
- The third comment is an **unsent draft** (`Status: DRAFT — awaiting user OK`) from a
  build carrying third-party `genesis-vllm-patches`. The author offered to re-verify on a
  stripped image and did not. Treat as anecdote, not evidence.
- **Cite this thread for the code-path claim and the timeline. Cite our own measurements
  for everything empirical.**

---

## 3. Functional requirements

- **R1 — Compatibility matrix, and it decides the headline.** Enumerate every
  `{mechanism} × {weight precision} × {KV precision} × {attention backend}` cell on our
  own hardware. Record: launches / serves correctly / fails, plus the **verbatim error
  string**, engine, commit, and compute capability. Complete before any performance sweep.
- **R1a — Silent-failure check.** "It launched" is not "it worked." Every cell that starts
  must pass an output-correctness check before being marked supported. A capability-gated
  dequant path can produce garbage rather than crashing, and a matrix that only records
  crashes would miss exactly the failure mode we are hunting.
- **R2 — Cross-engine confirmation.** Every verdict checked on **both SGLang and vLLM**.
  Agreement is evidence the gap is real rather than one team's bug; disagreement is itself
  reportable. SGLang has its own FP8-KV and spec-decode paths and its own sm_120 defects.
- **R3 — Acceptance length (τ) for every runnable cell.** Mean accepted tokens per
  verification step, logged per request, aggregated with dispersion.
- **R4 — Weight and KV quantization separated.** Measure the 2×2
  (`BF16/BF16`, `FP8-W/BF16-KV`, `BF16-W/FP8-KV`, `FP8-W/FP8-KV`) wherever it runs. The
  grid may come out **asymmetric**; empty cells are a result, not a gap.
- **R5 — Dual baselines.** Compare each cell against **both** a BF16 non-speculative
  baseline **and** an FP8 non-speculative baseline at matched precision. Speculative
  decoding is lossless w.r.t. *the target it verifies against*, so measuring only against
  BF16 tangles quantization error with speculation error. Both must be separable in the
  released data.
- **R6 — Batched serving, not single-request.** Every performance cell swept across
  concurrency (1 → whatever 24 GB allows). Single-request appears only as the
  concurrency=1 point, never as the headline.
- **R7 — Goodput under a declared SLO.** Successful requests/sec meeting an explicit
  latency SLO, not raw tokens/sec. The SLO is a **swept parameter** over a small declared
  set, so the conclusion's sensitivity to it is visible rather than hidden in one choice.
- **R8 — KV capacity per configuration.** Measure (not compute from a formula) KV-token
  capacity for each precision config. This is the mechanism linking compatibility to
  throughput.
- **R9 — Workload coverage.** GSM8K, MT-Bench, and a **structured-JSON** task from a
  public dataset. Constrained generation is a distinct acceptance regime and is
  under-represented in prior work. HumanEval/MBPP optional if budget allows.
- **R10 — Variance reported.** Minimum 3 repeats per cell; mean and spread. No bare point
  estimates anywhere.
- **R11 — Output equivalence check.** Generated text must match the matched-precision
  non-speculative baseline. A mismatch is a correctness bug and blocks that cell's
  performance results (and, per R1a, may itself be the finding).

---

## 4. Publication deliverables

- **R12 — Public repo.** Harness, configs, raw logs, plotting scripts, reproducibility
  appendix with exact per-configuration commands.
- **R13 — Zenodo DOI.** Repo snapshotted, cited by DOI. A bare GitHub link is not an
  acceptable artifact reference.
- **R14 — arXiv preprint**, cs.LG primary, cross-list cs.DC / cs.PF. Framed unmistakably
  as original empirical work (new experiments, hardware stated early), because arXiv CS
  rejects survey/position papers lacking prior peer review.
- **R15 — Endorsement secured before the paper is finished.** arXiv's 21 Jan 2026 policy
  requires institutional email *and* prior authorship, or a personal endorsement. We have
  neither credential. Blocking dependency, long lead time, start early October.
- **R16 — Upstream issue filed.** If H1 confirms, file the fine-grained issue the
  maintainer asked for, referencing our reproduction. Costs nothing, establishes the
  finding's date, and is the kind of contribution that reads well to faculty.

---

## 5. Non-functional requirements

- **N1 — Budget ceiling.** Free Ada-class tiers first (Lightning AI L40S, Modal $30/mo,
  Colab L4 for development). Paid RTX 4090 hours only for final clean sweeps. Hard cap
  ≈80 GPU-hours / ₹5,000. Spend logged per sweep.
- **N2 — Single 24 GB Ada GPU, and that is now load-bearing.** Compute capability **8.9**
  (RTX 4090 / L4 / L40S). No Blackwell consumer (sm_120 has open FP8 and NGRAM
  spec-decode defects), no multi-node, no TP > 1, no custom kernels. Under H1 the choice
  of SM89 stops being a budget compromise and becomes the experimental variable.
- **N2a — Record compute capability in every result record.** The whole hypothesis is
  about capability gating, so `compute_cap` is a first-class field, not metadata trivia.
- **N3 — Public models and data only.** Qwen3-4B / Qwen3.5-4B family, public draft
  checkpoints, public datasets. No gated weights.
- **N4 — Interruption-tolerant sweeps.** Free tiers are interruptible and Colab sessions
  time out. The harness writes results per-cell to durable storage and **resumes** a
  partial sweep without re-running finished cells. A disconnect mid-sweep must not
  invalidate comparability between cells.
- **N5 — Determinism and pinning.** Fixed seeds; sampling params, SGLang/vLLM commit,
  CUDA, driver, and model revision hashes in every result record. Metadata discipline is a
  stated contribution, so it must hold in the artifact itself.
- **N6 — One-command reproduction.** Clone, set one env var for the model cache, run a
  single entrypoint to reproduce any named figure.

---

## 6. Scope for the FIRST increment (week 1 — go/no-go)

Deliver the **compatibility matrix only**, on one rented Ada GPU:

- Harness skeleton: launch server, run a fixed probe workload, parse τ, write one result
  record with full version + compute-capability metadata.
- Enumerate `{NGRAM, EAGLE-3, DFlash, MTP} × {BF16, FP8-W, FP8-KV, both}` on SGLang and
  vLLM. Record launch / serves-correctly / fails with verbatim errors.
- **Test H1 directly:** DFlash (or any non-causal drafter) + `--kv-cache-dtype fp8` on
  SM89, current mainline. Distinguish three outcomes: clean success, crash, or
  **launches-but-wrong-output**.
- Baseline sanity: reproduce one published acceptance number within tolerance.

This increment is the paper's **floor**. If everything downstream fails, R1+R1a+R2 alone
is a short arXiv note with practical value to anyone renting consumer GPUs.

---

## 7. Unresolved

- ~~**U1** — Is vllm#41559 still reproducible?~~ **RESOLVED 2026-09-14.** Closed
  `completed` 2026-08-11 via FlashInfer/CUTLASS. Superseded by H1 (§2), which asks the
  narrower and better question: does the fix reach SM89?
- **U2** — Is DFlash block size a runtime flag (`--speculative-dflash-block-size`, per the
  Qwen3.5-4B card) or baked into the `-b16` checkpoint? Decides whether block size is a
  free axis. If baked in, target Qwen3.5-4B.
- **U3** — Does a usable EAGLE-3 checkpoint exist for the chosen target? The only Qwen3-4B
  candidate found is a Chinese-focused retrain — a confound to declare or avoid.
- **U4** — Does the target expose MTP heads at this size?
- **U5** — Actual concurrency ceiling at 24 GB with BF16 KV. Sets the top of the sweep;
  the ceiling is itself reportable.
- **U6** — Does SGLang have the same capability gate, or a different one? Its FP8-KV path
  is independent of vLLM's.

---

## 8. Out of scope

- Multiple target model families. One family; generality is a stated limitation.
- INT4 / AWQ / GPTQ precisions — covered by prior work, not re-litigated.
- Algorithmic fixes (adaptive γ, hierarchical frameworks). That is SpecKV's and HierSpec's
  lane; this is a measurement study.
- Training or fine-tuning draft models.
- **Patching the gate.** We measure and report; we do not ship a kernel fix. Filing the
  upstream issue (R16) is in scope, writing the PR is not — it converts a 9-week
  measurement paper into an open-ended systems project.
- Any claim that non-causal drafting is *architecturally* incompatible with KV
  quantization. llama.cpp disproves it. Language must stay at the implementation level.
- Block size as a headline axis. Cheap sub-result only; the MLX `block_size <= 5` note is
  an Apple-Silicon kernel-efficiency observation, not an acceptance-rate claim.
- Any employer data, workload, hardware, or observation (see §1 Provenance rule).

---

## 9. Prior art to position against

| Work | What it did | Why we are still novel |
|---|---|---|
| [2505.22179](https://arxiv.org/html/2505.22179v1) Spec Decoding Meets Quantization | EAGLE-2 × W8A8/W4A16/W4A8, A100+3090, single-request | No FP8, no KV-cache quant, not batched |
| [SpecKV 2605.02888](https://arxiv.org/abs/2605.02888) | Adaptive γ under compression | Algorithmic; step-level, not serving goodput |
| [QSpec (EMNLP 2025)](https://arxiv.org/abs/2410.11305) | W4A4 draft + W4A16 verify, shared weights | Shared-weight special case; not independent drafts, not FP8 |
| [QuantSpec 2502.10424](https://arxiv.org/html/2502.10424v1) | Self-spec, 4-bit hierarchical KV | A method, not a cross-mechanism measurement |
| [ML-SpecQD](https://arxiv.org/pdf/2503.13565), [Quasar](https://arxiv.org/html/2603.01399v1) | Quantized drafts; W8A8 verification | Partial overlap; neither reports availability |
| [vllm#41559](https://github.com/vllm-project/vllm/issues/41559) | Bug report + fix, May–Aug 2026 | Issue tracker, not a study. No Ada data, no τ, no goodput |

---

## 10. Timeline constraint

arXiv target **15 November 2026**. GRE **1 November** — no new runs that week.
Endorsement outreach starts **early October**, non-blocking on experiments.
