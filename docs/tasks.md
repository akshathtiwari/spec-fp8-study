# Tasks — Speculative Decoding × FP8 Compatibility Study

**implements [design.md](design.md) rev 2 · 2026-09-14**

---

## Increment 1 — compatibility matrix probe (week 1, go/no-go)

Goal: run `./run.sh probe` on one Ada GPU and get the compatibility matrix + H1 answer.
Build order follows design §12. Each task is one reviewable unit.

### 1.1 Project skeleton

- [x] T1  `pyproject.toml` — deps: httpx, pydantic, pyyaml, pandas. Entry point
      `specfp8`. Python ≥3.10. No engine deps.
- [x] T2  Directory structure: `specfp8/`, `specfp8/launchers/`, `specfp8/metrics/`,
      `specfp8/workloads/`, `sweeps/`, `analysis/`, `results/` (gitignored).
- [x] T3  `run.sh` — dispatches `probe`, `sweep`, `figures` subcommands (N6).
      Reads `SPECFP8_MODEL_CACHE` env var. Exits non-zero on unknown subcommand.

### 1.2 Core modules

- [x] T4  `specfp8/env.py` — capture GPU name, compute capability, VRAM, driver,
      CUDA version, torch version via `nvidia-smi --query-gpu` and
      `torch.cuda.get_device_capability`. Returns an `EnvInfo` pydantic model.
      **Test:** runs on any CUDA machine and returns all fields non-null.
- [x] T5  `specfp8/cells.py` — `ServerCell`, `RunCell` pydantic models per design §2.
      `cell_id` = `sha256(json.dumps(model_dump(), sort_keys=True, separators=(',',':')))[:16]`.
      `expand(yaml_path) -> list[ServerCell]`: Cartesian product of axes, merge
      defaults + per_mechanism, drop excludes, deduplicate.
      **Test:** hand-written YAML with 3 axes → correct expansion count, exclusions
      applied, cell_id stable across two calls.
- [x] T6  `specfp8/store.py` — `completed_ids(results_dir) -> set[str]` (scan
      `cells.jsonl` once), `append(record, results_dir)` (write to temp file, fsync,
      `os.replace`). Budget log append to `results/budget.log`.
      **Test:** append two records, kill mid-third (simulate with no-fsync path),
      `completed_ids` returns exactly 2.

### 1.3 Engine launchers

- [x] T7  `specfp8/launchers/base.py` — `Launcher` protocol, `ServerHandle` model,
      `HealthResult` model per design §3. Abstract `argv()`, `start()`,
      `wait_healthy()`, `kv_capacity()`, `stop()`.
- [x] T8  `specfp8/launchers/vllm.py` — vLLM launcher. `argv()` builds the
      `vllm serve <model>` command with all ServerCell fields mapped to CLI flags
      (speculative settings go in the `--speculative-config` JSON blob, per the
      vLLM 0.29 surface). `start()` spawns subprocess, streams stderr to
      `logs/<cell_id>.log`. `wait_healthy()` polls `/health`, watches child, returns
      `HealthResult` with verbatim error on failure. `kv_capacity()` parses
      `num_gpu_blocks` from startup log. `stop()` sends SIGTERM, waits 10s, SIGKILL.
      **Test:** launch vLLM with a tiny model (Qwen3-0.6B), confirm health, stop.
      Requires GPU — mark as integration test.
- [x] T9  `specfp8/launchers/sglang.py` — SGLang launcher. Same interface, maps to
      `python -m sglang.launch_server`. Health endpoint: `/health_generate`.
      **Test:** same pattern as T8 with SGLang.

### 1.4 Client (minimal for probe)

- [x] T10 `specfp8/client.py` — async httpx client. For probe: concurrency 1,
      sends `n_warmup` requests (discarded), then `n_measure` requests. Records
      per-request: `ttft_ms`, `e2e_ms`, `output_tokens`, `itl_ms[]`, `ok`,
      `sampling_params`. Streaming SSE parsing for token-level timing.
      Hard per-request timeout (configurable, default 120s).
      **Test:** mock HTTP server returning SSE chunks → correct timing extraction.

### 1.5 Correctness

- [x] T11 `specfp8/correctness.py` — implements design §4's three tests.
      - **Test 1 (equivalence):** 32 fixed prompts (16 MT-Bench + 16 GSM8K),
        256 max tokens, temperature=0, fixed seed. Compare spec vs matched-precision
        non-spec. Threshold: exact_match ≥ 0.95. Always report raw rate.
      - **Test 2 (damage):** same prompts, FP8 non-spec vs BF16 non-spec. Report
        token-level disagreement fraction (tokens diverging / total tokens).
      - **Test 3 (garbage):** non-empty check, degenerate-repetition detector
        (any 4-gram repeating ≥5× in output), GSM8K accuracy against floor of 0.50
        (below this = broken; 0.50–0.70 = degraded; ≥0.70 = correct for 4B model).
      - Returns `CorrectnessResult` with verdict logic.
      **Test:** feed known-good and known-garbage outputs → correct verdicts.
- [x] T12 Commit the 32 fixed prompts as `specfp8/workloads/correctness_prompts.json`.
      16 from MT-Bench (diverse categories), 16 from GSM8K (with reference answers
      for Test 3). Pin by index so the set never drifts.

### 1.6 Metrics

- [x] T13 `specfp8/metrics/base.py` — `SpecStats` model (design §3). `scrape(url)`
      parses Prometheus text format from `/metrics`. `delta(before, after) -> SpecStats`.
      τ = `accepted_tokens / verification_steps`. Null if counters missing.
- [x] T14 `specfp8/metrics/vllm.py` — maps vLLM counter names to `SpecStats` fields.
      Document which counters and which vLLM versions expose them.
- [x] T15 `specfp8/metrics/sglang.py` — maps SGLang counter names to `SpecStats` fields.

### 1.7 Probe orchestrator

- [x] T16 `sweeps/compat.yaml` — full Phase 1 matrix per design §2a.
      `{NGRAM, EAGLE-3, DFlash, MTP, none} × {BF16, FP8} weights × {auto, fp8_e4m3} KV
      × {sglang, vllm}`. Include `guided_decoding: true` cells for structured-JSON
      compat check. Exclude nonsensical combos.
- [x] T17 `specfp8/probe.py` — Phase 1 orchestrator.
      1. `expand(compat.yaml)` → list of ServerCells.
      2. `completed_ids()` → skip finished.
      3. For each ServerCell: boot, warmup (3 throwaway requests), run correctness
         suite, scrape one short τ measurement (32 prompts, concurrency 1).
      4. Write result record with status, correctness, τ, env, wallclock_s.
      5. Print running GPU-time total after each server group.
      6. Append to `results/budget.log`.
      **Test:** mock launcher (always healthy) + mock client → correct orchestration
      flow, resume skips completed cells.

### 1.8 Integration test on GPU

- [x] T18 End-to-end: `./run.sh probe --sweep sweeps/test_mini.yaml` on a real GPU.
      `test_mini.yaml` has 2 cells: `{none, ngram} × bf16 × auto × vllm`.
      Confirm: results written, cell_ids stable, resume works (re-run skips both),
      budget.log populated.
      **Done** on Modal L4 (SM89, 22.5GB, CUDA 13.0, torch 2.13.0+cu130,
      vLLM 0.29.0). Both cells `ok`: KV capacity 168,240 / 159,664 tokens,
      τ = null for `none` (correct — no speculation) and τ = 1.47 for `ngram`,
      no degenerate or empty outputs. Resume, staleness and budget.log all
      exercised. Required fixing the vLLM 0.29 CLI surface, the spec counter
      names, the τ definition, KV-capacity parsing, and the Qwen3 thinking
      mode — see T18a.

- [ ] T18a **Test 1's premise does not hold on this stack — blocks T19's third
      outcome.** With `temperature: 0`, BF16, same model, same seed, speculative
      (ngram) and non-speculative outputs agree on only **11/32 = 34.4%** of
      prompts, and the divergences are semantic, not cosmetic ("heart of the
      Pacific" vs "island of the sun"). Design §4 assumes ≥95% exact match at
      matched precision and uses any shortfall as the sharp instrument for
      detecting a broken FP8 kernel. At a 34.4% floor that instrument cannot
      detect anything: FP8 damage would be indistinguishable from the baseline.
      Two further observations: task accuracy for the *same* config changed
      across boots (18.75% → 25.0%), so some of this is boot-to-boot
      nondeterminism rather than speculation per se.
      Likely cause: speculative decoding is lossless in distribution (rejection
      sampling preserves the target distribution) but greedy argmax can flip on
      near-ties when verification-batch numerics differ from single-token decode
      numerics, and one flipped token diverges the rest of the text.
      **Experiment run** (`--engine determinism`, L4, Qwen3-0.6B, BF16):
      - same config, **fresh** server vs fresh server: **32/32 = 100.0%**
      - same server, **second** pass: **29/32 = 90.6%**

      So the engine is exactly reproducible given identical server state, and
      what perturbs it is *accumulated* state — pass B hits KV left behind by
      pass A. Prefix caching is the prime suspect (enabled by default in
      vLLM 0.29); the mechanism is that a cached prefix and a freshly computed
      one give slightly different logits, which flips argmax on near-ties, and
      one flipped token diverges everything after it.

      Two consequences, one good and one bad:

      1. **The harness is sound.** Every probe cell boots a fresh server and
         makes one pass, which is the 100% condition. Cross-cell comparisons
         are therefore reproducible, and the 34.4% spec-vs-nonspec gap is
         *real signal attributable to speculation*, not measurement noise.
      2. **Test 1 still cannot be used as specified.** vLLM's speculative path
         is not token-identical to the non-speculative one at temperature 0 on
         this stack, so the ≥95% exact-match threshold fails at the BF16
         baseline, before FP8 enters. The threshold has to be recalibrated
         against the *measured BF16 spec baseline* rather than against 100% —
         the question becomes whether FP8 degrades agreement materially below
         that baseline, which `sweeps/h1.yaml` collects the matched arms for.

- [ ] T18b **Phase 2 hazard from the same finding.** `group_by_server` boots
      once and runs many RunCells against that server, which is exactly the
      90.6% condition. Harmless for throughput and latency, but any
      correctness or token-identity claim drawn from a reused server is
      measuring accumulated cache state as much as the configuration. Either
      re-boot per correctness arm or restrict correctness claims to
      fresh-server cells.
- [x] T19 **The H1 test — H1 is REFUTED. Headline B.**
      `sweeps/h1.yaml`, Modal L4 (SM89, 22.03 GiB), vLLM 0.29.0 (version read
      from `/version`, not assumed), Qwen3-4B, DFlash draft
      `z-lab/Qwen3-4B-DFlash-b16`, temperature 0, thinking disabled.
      All 8 cells booted and served; **no cell crashed**.

      | mechanism | weight | kv_cache | KV tokens | τ | accept |
      |---|---|---|---|---|---|
      | none   | bf16 | auto     |  77,168 | –     | –      |
      | none   | bf16 | fp8_e4m3 | 147,776 | –     | –      |
      | none   | fp8  | auto     | 104,768 | –     | –      |
      | none   | fp8  | fp8_e4m3 | 204,384 | –     | –      |
      | dflash | bf16 | auto     |  58,080 | 2.685 | 0.3371 |
      | dflash | bf16 | fp8_e4m3 | 115,920 | 2.699 | 0.3398 |
      | dflash | fp8  | auto     |  77,872 | 2.690 | 0.3380 |
      | dflash | fp8  | fp8_e4m3 | 155,232 | 2.680 | 0.3361 |

      Three results, in order of how well they are supported:

      1. **DFlash + fp8_e4m3 KV dispatches and serves on SM89.** The predicted
         runtime kernel failure did not occur. This is the pivot: the paper is
         headline B, not A.
      2. **FP8 KV buys ~1.96× KV capacity** (1.915/1.951/1.996/1.993×), against
         a theoretical 2.0× for 16→8 bit. Stacking FP8 weights and FP8 KV gives
         2.65× over the BF16 baseline (77,168 → 204,384), because smaller
         weights also free memory for cache.
      3. **τ is invariant to precision.** Across all four DFlash cells τ spans
         2.680–2.699 (range 0.019, 0.7%) and acceptance 0.3361–0.3398 (1.1%).
         Neither FP8 weights nor FP8 KV measurably changes speculative
         acceptance.

      Also visible: DFlash costs 25% of KV capacity (77,168 → 58,080 at
      bf16/auto) because the draft model occupies memory the cache would
      otherwise use. That is the operator tradeoff the study set out to
      quantify, and FP8 KV more than pays for it.

      **Not established, and must not be claimed:**
      - *No quality damage.* Every cell reads `broken`, but only because the
        GSM8K floors inherited for a 4B target are wrong here — accuracy runs
        25–43.75% on **16** prompts, where one question is 6.25%, so the whole
        spread is 4/16 to 7/16 and the arm cannot resolve FP8 damage at all.
        See T19a.
      - *Why it works.* `attn_backend` was `auto` and the logs that name the
        chosen backend live only inside the container. Without that, the
        refutation has no mechanism. See T19b.
      - *Generality.* One GPU (L4), one engine version (0.29.0). Says nothing
        about RTX 4090, L40S, or whether issues #54690 / #44879 were fixed
        versus never applying to this path.

- [ ] T19a **The accuracy arm is underpowered.** 16 GSM8K prompts gives 6.25%
      resolution, so it cannot distinguish 31% from 44%, let alone detect the
      subtler damage FP8 would cause. Two fixes, both needed: raise the GSM8K
      prompt count until the interval is narrow enough to matter, and set the
      floors from the *measured* bf16/auto/none baseline rather than an assumed
      4B number. Until then no quality claim about FP8 is supportable.

- [x] T19b **DONE — backend axis resolved. H1 is refuted on its own terms.**
      `sweeps/backend.yaml`, 12 cells, Modal L4 (SM89), vLLM 0.29.0 +
      flashinfer 0.6.18, Qwen3-4B, bf16 weights. Backend attributed from
      persisted logs, not inferred. **All 12 cells served.**

      | mech | kv | requested | **actual** | tau | KV tokens |
      |---|---|---|---|---|---|
      | none | auto | auto | **FLASH_ATTN** | - | 77,168 |
      | none | auto | FLASHINFER | FLASHINFER | - | 73,888 |
      | none | auto | TRITON_ATTN | TRITON_ATTN | - | 77,168 |
      | none | fp8_e4m3 | auto | **FLASHINFER** | - | 147,776 |
      | none | fp8_e4m3 | FLASHINFER | FLASHINFER | - | 147,776 |
      | none | fp8_e4m3 | TRITON_ATTN | TRITON_ATTN | - | 153,872 |
      | dflash | auto | auto | **FLASH_ATTN** | 2.703 | 58,080 |
      | dflash | auto | FLASHINFER | FLASHINFER | 2.716 | 58,416 |
      | dflash | auto | TRITON_ATTN | TRITON_ATTN | 2.651 | 58,528 |
      | dflash | fp8_e4m3 | auto | **FLASHINFER** | 2.695 | 115,920 |
      | **dflash** | **fp8_e4m3** | **FLASHINFER** | **FLASHINFER** | **2.697** | 116,832 |
      | dflash | fp8_e4m3 | TRITON_ATTN | TRITON_ATTN | 2.745 | 116,832 |

      1. **Backend selection is conditioned on KV dtype.** `auto` picks
         FLASH_ATTN at BF16 KV and switches to FLASHINFER when FP8 KV is
         requested. Log line: `Using FLASH_ATTN attention backend out of
         potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN',
         'FLEX_ATTENTION']`.
      2. **The h1.yaml sweep therefore did exercise FlashInfer** — its
         `fp8_e4m3` cells ran on it. The worry that H1's mechanism was never
         tested was unfounded, but checking it is what produced this matrix.
      3. **H1 refuted on both failure modes.** FlashInfer + FP8 KV +
         non-causal drafting serves at tau = 2.697, against 2.716 for the same
         backend at BF16. No crash, and tau intact rules out silent
         corruption — a broken kernel would collapse acceptance.
      4. **Every backend works.** FP8 KV + dflash succeeds on FLASHINFER
         (2.697) and TRITON_ATTN (2.745). tau spans 2.651-2.745 across all
         six dflash conditions, so backend choice moves tau (3.6%) slightly
         more than FP8 does.
      5. **The two open issues are version-specific, not architectural.**
         #44879 is 0.22.1 on an L4 — the same GPU class we ran; #54690 is
         0.28.0. Both paths work on 0.29.0. Note flashinfer#5272 is still
         open, so whatever fixed this is something else and needs identifying
         before filing (R16).

- [ ] T19d **Two backends in the SM89 candidate set are still untested.**
      The engine lists `['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN',
      'FLEX_ATTENTION']`. FLASH_ATTN is the BF16 default but `auto` switches
      away from it the moment FP8 KV is requested, so **FLASH_ATTN x fp8_e4m3
      never ran** — a hole directly under the default configuration.
      FLEX_ATTENTION is untouched. `sweeps/backend.yaml` excluded FLASH_ATTN
      on the wrong premise that it was unavailable: the absent `flash_attn`
      pip package is not vLLM's FlashAttention, which is vendored. Add both
      to close the backend axis.

- [ ] T19c **Order sweep cells by information value, not by Cartesian expansion.**
      `cells.expand` emits the axis product in declaration order, so in
      `sweeps/backend.yaml` all six `none` cells run before any `dflash` cell —
      putting the six that actually test H1 last, with the most exposure to the
      function timeout. Resume makes this recoverable rather than fatal, but the
      ordering is still backwards: the cells that can end a sweep early should
      run first. Either allow a sweep to declare a priority ordering, or sort
      so that each axis's levels are interleaved rather than blocked.

- [x] T19b-orig **Identify the attention backend, and persist launch logs.**
      H1 predicted a FlashInfer SM90-only dispatch failure; it did not happen,
      and the paper needs to say what actually served — a different backend
      (FlashAttention/Triton), a newer FlashInfer with an SM89 variant, or a
      fix landed since the issues were filed. `log_path` is recorded in every
      result but the logs themselves are never copied to the results volume,
      so this evidence was lost when the container exited. Persist them, then
      re-read the backend selection line.

### 1.9 Cross-engine comparison

- [x] T20 `specfp8/analysis/cross_engine.py` — takes `results/cells.jsonl`, groups
      cells by config (minus engine field), pairs SGLang vs vLLM verdicts. Outputs a
      comparison table: config → {sglang_status, vllm_status, agreement}. Flags
      disagreements. This is R2's design answer (was gap G2).

### 1.10 Docs and cleanup

- [x] T21 `README.md` at project root — what the study is, how to run, hardware
      requirements, link to docs/.
- [x] T22 `.gitignore` — results/, logs/, *.pyc, .venv/, model caches.

---

## Increment 2 — performance sweep (weeks 2–5)

Only cells Phase 1 marked `ok` or `degraded` enter here.

- [ ] T23 `sweeps/perf.yaml` — adds `run_axes`: workload (gsm8k, mtbench),
      concurrency (1, 4, 8, 16, 32, 64), seed (3 values), repeat (0, 1, 2).
- [ ] T24 `specfp8/client.py` — full async load generator. Fixed concurrency (N in
      flight at all times). Warmup phase. Per-request JSONL output with
      `sampling_params` recorded.
- [ ] T25 `specfp8/workloads/base.py` — `Workload` protocol per design §9.
- [ ] T26 `specfp8/workloads/gsm8k.py` — prompt sampling, numeric-answer validator.
- [ ] T27 `specfp8/workloads/mtbench.py` — prompt sampling, no validator.
- [ ] T28 `specfp8/workloads/structured_json.py` — schema-constrained generation,
      JSON-parse + schema-conform validator. Public dataset TBD.
- [ ] T29 `specfp8/sweep.py` — Phase 2 orchestrator. Groups RunCells by ServerCell,
      boots once per group, iterates workloads × concurrency × repeats. Writes
      summary record + per-request JSONL. Budget tracking. Resume-safe.
- [ ] T30 Integration: `./run.sh sweep --sweep sweeps/perf.yaml` end-to-end on GPU.

---

## Increment 3 — analysis and figures (weeks 6–8)

- [ ] T31 `analysis/goodput.py` — reads `results/requests/*.jsonl`, computes goodput
      at each SLO in a declared set (p95 ITL ≤ {20, 35, 50, 75} ms,
      TTFT ≤ {200, 500, 1000} ms). Outputs a tidy CSV.
- [ ] T32 `analysis/figures.py` — generates all paper figures from `results/`.
      Minimum set:
      - Fig 1: compatibility matrix heatmap (status per cell)
      - Fig 2: τ by mechanism × precision (grouped bar)
      - Fig 3: weight-quant vs KV-quant τ impact (2×2 panel)
      - Fig 4: goodput vs concurrency, one line per precision config
      - Fig 5: goodput sensitivity to SLO threshold
      - Fig 6: KV capacity per config (bar)
- [ ] T33 `./run.sh figures` — regenerates every figure from `results/`. No GPU
      needed. Deterministic from the same data.
- [ ] T34 Cross-engine comparison table (from T20) formatted for the paper.

---

## Increment 4 — paper and release (weeks 8–9)

- [ ] T35 LaTeX draft: intro, background, method, results, discussion, conclusion.
      Figures inserted from `analysis/` output.
- [ ] T36 Reproducibility appendix: exact commands, env snapshot, per-figure
      reproduction instructions.
- [ ] T37 Zenodo DOI — snapshot repo, upload, get DOI (R13).
- [ ] T38 arXiv submission — cs.LG primary, cs.DC cross-list (R14).
- [ ] T39 If H1 confirmed: file upstream issue on vLLM with reproduction steps (R16).

---

## Ongoing (non-blocking, parallel track)

- [ ] T40 arXiv endorsement outreach — start early October. Target: authors of
      cited papers (§9), not faculty cold emails. Need one endorser for cs.LG.
- [ ] T41 Resolve U2 (DFlash block size: runtime flag vs baked into checkpoint).
- [ ] T42 Resolve U3 (EAGLE-3 checkpoint availability for chosen target model).
- [ ] T43 Resolve U4 (MTP head availability at 4B size).

---

## Verification checklist (applied to every increment)

- [ ] All result records contain `compute_cap`, `env`, `engine` fields (N2a, N5)
- [ ] `cell_id` is stable: same config → same ID across machines
- [ ] Resume works: re-running a completed sweep adds zero new records
- [ ] No engine code imported anywhere in `specfp8/` (design §1 boundary)
- [ ] Provenance rule: no employer data, configs, or observations in any file
