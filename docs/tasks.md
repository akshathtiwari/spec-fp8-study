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
      `python -m vllm.entrypoints.openai.api_server` command with all ServerCell
      fields mapped to CLI flags. `start()` spawns subprocess, streams stderr to
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

- [ ] T18 End-to-end: `./run.sh probe --sweep sweeps/test_mini.yaml` on a real GPU.
      `test_mini.yaml` has 2 cells: `{none, ngram} × bf16 × auto × vllm`.
      Confirm: results written, cell_ids stable, resume works (re-run skips both),
      budget.log populated.
- [ ] T19 **The H1 test.** Run DFlash + `kv_cache_dtype: fp8_e4m3` on SM89 (RTX 4090
      or L4 or L40S). Record the three possible outcomes:
      - Clean success → H1 refuted, headline B
      - Crash with verbatim error → H1 confirmed, headline A
      - Launches but Test 1 fails (silent corruption) → H1 confirmed, headline A+
      This single cell is the paper's pivot point.

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
