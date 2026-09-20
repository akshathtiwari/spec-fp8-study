# Requirements — Speculative Decoding × FP8 Compatibility Study

**Akshath Tiwari · independent · spec 2026-09-14 · rev 3 (evidence table updated
2026-09-20 after checking [#54690](https://github.com/vllm-project/vllm/issues/54690),
[#44879](https://github.com/vllm-project/vllm/issues/44879), FlashInfer kernel scope,
and third-party FA2 plugin evidence)**

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
got fixed in mid-2026 (PR #39995, merged May 18; PR #43081, merged June 22). But the fix
routes through FlashInfer's FP8 attention kernels, which are **compiled only for SM90
(Hopper)**. There are no SM89 kernel variants. The backend happily *accepts* the
configuration on SM89 (the compute-capability check allows SM80 through SM121), but the
actual attention kernel dispatch has no SM89 codepath.

Two independent users have reported exactly this failure on SM89 hardware:
[#54690](https://github.com/vllm-project/vllm/issues/54690) (4× RTX 4090, vLLM 0.28.0,
Sep 2026 — native crash during draft attention init) and
[#44879](https://github.com/vllm-project/vllm/issues/44879) (L4, vLLM 0.22.1 — CUDA
illegal memory access). Both issues remain open. A third-party FA2 plugin
([fa2-fp8kv-sm86](https://github.com/AntonProkopyev/fa2-fp8kv-sm86)) proves the
combination is architecturally possible on SM86/SM89, but stock vLLM does not use it.

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
| DFlash + FP8 KV failed in vLLM v0.20.0 on all backends | [#41559](https://github.com/vllm-project/vllm/issues/41559) issue body, May 2026 | true **then**, on a **non-stock** image; issue now **closed** |
| It was never architecturally impossible | llama.cpp fork ([Cybertiron/llama.cpp-rtx3090-...](https://github.com/Cybertiron/llama.cpp-rtx3090-prefill-kvarn-dflash2_moved_to_vllm)) shipped DFlash2 + quantized KV on 3090, ~2.4× context expansion, 240K tokens in 24GB | **settled** — drop any "structural" language (note: 5.46× speedup not confirmed; actual gain is context capacity) |
| vLLM added support | [PR #39995](https://github.com/vllm-project/vllm/pull/39995) (FlashInfer + DFlash + FP8 KV, merged May 18), [PR #43081](https://github.com/vllm-project/vllm/pull/43081) (DFlash with FlashInfer, merged June 22) | **merged**, but SM89 **untested by authors** |
| FlashInfer FP8 kernels are SM90-only | `batch_prefill_fp8_sm90_kernel_inst.jinja`, `single_prefill_fp8_sm90.cu`; confirmed in [#44879](https://github.com/vllm-project/vllm/issues/44879) root cause; FA3 FP8 KV PR ([flashinfer#4977](https://github.com/flashinfer-ai/flashinfer/pull/4977)) also SM90-only | **confirmed** — no SM89 FP8 attention kernel exists in FlashInfer |
| Config is accepted on SM89 | `supports_compute_capability` accepts `>= (8,0)` and `<= (12,1)` — no startup rejection | **confirmed** via code reading |
| DFlash + FP8 KV **crashes on SM89** | [#54690](https://github.com/vllm-project/vllm/issues/54690) (4× RTX 4090, vLLM 0.28.0, Sep 2026): FlashInfer `ValueError: Unrecognized dtype: auto`, then native crash with explicit dtype. [#44879](https://github.com/vllm-project/vllm/issues/44879) (L4, vLLM 0.22.1): `CUDA error: illegal memory access`. Both **OPEN**. | **two independent crash reports on SM89** |
| Third-party plugin proves SM8x is possible | [fa2-fp8kv-sm86](https://github.com/AntonProkopyev/fa2-fp8kv-sm86) plugin enables DFlash2 + FP8 KV on 2× RTX 3090 (SM86) via FA2 path, vLLM 0.29.0, verified at 262K context ([HyperQwen#153](https://github.com/syv-ai/HyperQwen/issues/153)) | **working** — but not stock vLLM |
| FlashInfer inline FP8 KV scale for FA2 | [flashinfer#5164](https://github.com/flashinfer-ai/flashinfer/pull/5164) → redirected to [#5272](https://github.com/flashinfer-ai/flashinfer/pull/5272), per-(token,head) FP8 scale on SM80+ including SM89 | **in development** (Sep 2026), not yet in stock vLLM |
| It still crashes on SM121 after the fix | illegal memory access, GB10, July 2026 | reported, one user |
| vLLM v0.29.0 did not fix SM89 FP8 KV | release notes checked — no SM89 or FP8 KV fixes | **unfixed as of latest release** |
| **Behaviour on SM89 with stock vLLM** | crash reports exist, systematic measurement does not | **BROKEN per reports. Our study provides first systematic confirmation.** |

### Hypothesis H1 (to test in week 1)

> SM89 falls inside the accepted compute-capability range (SM80–SM121) but FlashInfer's
> FP8 attention kernels only exist for SM90 (Hopper). DFlash's non-causal attention with
> FP8 KV cache is therefore accepted at config time and then crashes or silently
> misbehaves on Ada hardware. Two open issues ([#54690](https://github.com/vllm-project/vllm/issues/54690),
> [#44879](https://github.com/vllm-project/vllm/issues/44879)) report crashes consistent
> with this mechanism. Our study provides the first *systematic* confirmation with
> controlled measurements.

**Refined mechanism (rev 3):** The original rev 2 hypothesis attributed the gate to
`trtllm_prefill_attn_kvfp8_dequant` being gated to SM100. Further investigation shows
that gate applies to **NVFP4** KV, not FP8. The actual bottleneck for FP8 KV + non-causal
is that FlashInfer's FP8 attention kernels (`batch_prefill_fp8_sm90_kernel_inst`,
`single_prefill_fp8_sm90`) are compiled for **SM90 only**. SM89 has no variant. The
compute-capability check at config time accepts SM89, but the kernel dispatch at runtime
has no codepath for it.

### Branch on the result

- **H1 confirmed → headline A.** *"FP8 KV cache with non-causal speculative drafting
  passes configuration validation on Ada (SM89) but crashes at runtime because FlashInfer
  ships no FP8 attention kernel for that architecture. A third-party FA2 plugin proves the
  combination is possible; stock vLLM does not deliver it."* Two open issues provide
  anecdotal evidence; our study provides systematic measurement across all five draft
  mechanisms on SM89 hardware.
- **H1 refuted (it works fine on Ada) → headline B.** Fall back to the measurement study:
  τ under FP8 across four draft mechanisms, weight-quant vs KV-quant separated, goodput
  under concurrency. Still novel on the FP8 and KV axes. Compatibility matrix demotes to
  a supporting table. (This would also mean #54690/#44879 are model-specific, not
  architecture-specific — which is itself a finding.)
- **Either way the experiments in §4 are the same.** Only the framing moves. This is why
  R1/R2 gate everything else.

### Evidence-quality caveats

**Issue #41559 (the original thread):**
- The original report used a **custom image** (`...vllm-openai:v0.20.0-tq-hybrid-v2`);
  `turboquant` is not a mainline backend. Not stock vLLM.
- Original hardware was a 3090 (Ampere SM86), target a 27B int4 model. Neither matches
  our setup.
- The third comment is an **unsent draft** (`Status: DRAFT — awaiting user OK`) from a
  build carrying third-party `genesis-vllm-patches`. Treat as anecdote, not evidence.
- **Cite this thread for the code-path claim and the timeline only.**

**Issue #54690 (the strongest prior evidence):**
- Filed Sep 2026 on vLLM 0.28.0, **4× RTX 4090** (SM89). This is our target hardware.
- Uses hybrid GDN models (Qwen3.5-family), which adds complexity beyond our Qwen3-4B
  setup. Our study uses a simpler model, so if we reproduce the crash, it is not
  model-specific.
- Author states "appears unchanged in current main."
- **Cite for the SM89 crash report. Our measurements provide systematic confirmation.**

**Issue #44879 (MTP variant):**
- Filed on vLLM 0.22.1, **NVIDIA L4** (SM89). Different speculative method (MTP, not
  DFlash) but same root cause (FlashInfer FP8 SM90-only kernels).
- Uses compressed-tensors FP8 model, not manual `--kv-cache-dtype fp8`.
- **Cite for breadth — the problem extends beyond DFlash to any spec decode + FP8 KV.**

**llama.cpp fork (Cybertiron):**
- Confirms DFlash2 + quantized KV is architecturally possible on SM86 (3090).
- The 5.46× speedup figure in rev 2 of this spec was **not confirmed** in the source;
  actual published gain is ~2.4× context capacity. Correct or remove from any citations.

**HyperQwen #153 (third-party FA2 plugin):**
- Proves FP8 KV + DFlash2 works on SM86 via a custom FA2 plugin, not stock vLLM.
- **Cite only to establish that the fix is possible, not that stock vLLM delivers it.**

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
