# Design — Speculative Decoding × FP8 Compatibility Study

**rev 2 · 2026-09-14 · implements [requirements.md](requirements.md)**
**rev 2 changes:** added sweep YAML schema (§2a), budget tracking (§6a), fixed Test 1 prompt spec (§4)

---

## 0. The design in plain words

We need to run the same benchmark many times over many combinations: four draft
mechanisms × four precision settings × two engines × several workloads × several
concurrency levels × three repeats. That is a few hundred runs, on a rented GPU that can
disappear mid-sweep.

So the harness is built around three ideas:

1. **One "cell" = one configuration.** Every cell gets a stable ID computed from its
   settings. Results are appended to a file keyed by that ID. Restarting a sweep skips
   cells that already finished. A disconnect costs you one cell, not the sweep.
2. **The harness never imports vLLM or SGLang.** It launches them as subprocesses and
   talks to them over HTTP. This matters more than it sounds: both engines have enormous,
   conflicting CUDA dependency trees, and we need to test *multiple versions of both*. A
   thin HTTP client survives all of that; a Python package that imports them does not.
3. **Store raw, derive later.** We save every individual request's timings, not just
   averages. Goodput at any SLO is then computed offline from the same data, so sweeping
   the SLO (R7) costs zero extra GPU time.

---

## 1. Delivery shape

```
spec-fp8-study/
├── docs/{requirements,design,tasks}.md
├── pyproject.toml            # harness deps only: httpx, pydantic, pyyaml, pandas
├── run.sh                    # N6: single entrypoint
├── specfp8/
│   ├── cells.py              # Cell model, stable id, sweep expansion
│   ├── env.py                # GPU / driver / CUDA / engine version capture
│   ├── store.py              # atomic JSONL append + resume set
│   ├── client.py             # async OpenAI-compatible load generator
│   ├── correctness.py        # equivalence + damage + garbage tests
│   ├── probe.py              # PHASE 1 — compatibility matrix (R1, R1a)
│   ├── sweep.py              # PHASE 2 — performance sweep (R3–R10)
│   ├── launchers/{base,sglang,vllm}.py
│   ├── metrics/{base,sglang,vllm}.py
│   └── workloads/{base,gsm8k,mtbench,structured_json}.py
├── sweeps/{compat.yaml,perf.yaml}
├── analysis/{goodput.py,figures.py}
└── results/                  # gitignored; published as release asset + Zenodo
```

**Dependency boundary (load-bearing).** `specfp8/` is pure Python over HTTP. Engines live
in their own venv, container, or image and are started via `subprocess`. The harness is
told *how* to launch via config, never by importing engine code. Consequence: testing
vLLM `v0.20.0` against `main` is a config change, not an environment rebuild.

---

## 2. The Cell abstraction

```python
# specfp8/cells.py
class ServerCell(BaseModel):        # one server boot
    engine: Literal["sglang", "vllm"]
    engine_ref: str                 # version tag or commit
    model: str
    model_revision: str
    mechanism: Literal["none", "ngram", "eagle3", "dflash", "mtp"]
    draft_model: str | None
    draft_revision: str | None
    weight_precision: Literal["bf16", "fp8"]
    kv_cache_dtype: Literal["auto", "fp8_e4m3", "fp8_e5m2"]
    attn_backend: str
    spec_tokens: int | None
    block_size: int | None
    guided_decoding: bool = False   # structured-JSON cells

class RunCell(BaseModel):           # one measurement against a booted server
    server: ServerCell
    workload: str
    concurrency: int
    seed: int
    repeat: int
```

`cell_id = sha256(canonical_json(cell))[:16]`. Canonical JSON means: keys sorted
recursively, no whitespace, no trailing commas — `json.dumps(cell.model_dump(),
sort_keys=True, separators=(',', ':'))`. Stable across machines, so results from a
Lightning session and a 4090 session merge cleanly.

**Grouping is the main cost optimisation.** Server boot is 1–3 minutes. `sweep.py` groups
all `RunCell`s sharing a `ServerCell`, boots once, runs every workload × concurrency ×
repeat against that server, then tears down. Naive per-cell boots would cost more GPU time
than the measurements.

---

## 2a. Sweep YAML schema

Sweep files declare axes and optional exclusions. `cells.py` expands the Cartesian product
and drops excluded combinations.

```yaml
# sweeps/compat.yaml — Phase 1 compatibility matrix
model: Qwen/Qwen3-4B
model_revision: main
spec_tokens: 5

axes:
  engine: [sglang, vllm]
  engine_ref: [latest]              # or pinned tags
  mechanism: [none, ngram, eagle3, dflash, mtp]
  weight_precision: [bf16, fp8]
  kv_cache_dtype: [auto, fp8_e4m3]
  attn_backend: [auto]              # let engine pick; override per-cell if needed

defaults:
  draft_model: null                 # overridden per-mechanism below
  draft_revision: null
  block_size: null
  guided_decoding: false

per_mechanism:
  eagle3:
    draft_model: some-org/Qwen3-4B-EAGLE3
    draft_revision: main
  dflash:
    draft_model: z-lab/Qwen3-4B-DFlash-b16
    draft_revision: main
    block_size: 16

exclude:
  - {mechanism: none, kv_cache_dtype: fp8_e4m3, weight_precision: bf16}
    # pure BF16 non-spec baseline doesn't need FP8-KV-only variant
```

`cells.py` exposes `expand(yaml_path) -> list[ServerCell]` which:
1. Takes the Cartesian product of all `axes` lists.
2. Merges `defaults`, then `per_mechanism` overrides for the matching mechanism.
3. Drops any cell matching an `exclude` entry.
4. Returns de-duplicated `ServerCell` objects.

For Phase 2 (`sweeps/perf.yaml`), the same schema adds `run_axes` for workload,
concurrency, seed, and repeat — those expand into `RunCell` lists per `ServerCell`.

---

## 3. Engine abstraction

The seam: **launch and metrics are engine-specific; client, workloads, and analysis are
shared.** Both engines expose an OpenAI-compatible API, so the load generator is written
once.

```python
# specfp8/launchers/base.py
class Launcher(Protocol):
    def argv(self, cell: ServerCell) -> list[str]: ...
    def start(self, cell) -> ServerHandle: ...   # spawn, stream logs to file
    def wait_healthy(self, timeout_s: int) -> HealthResult: ...
    def kv_capacity(self, handle) -> int | None: ...  # parse startup log
    def stop(self, handle) -> None: ...

class HealthResult(BaseModel):
    ok: bool
    status: Literal["ok","launch_failed","oom","timeout","unsupported_config"]
    error_verbatim: str | None     # R1 requires the exact string
    log_path: str
```

`wait_healthy` polls `/health` (vLLM) or `/health_generate` (SGLang) and simultaneously
watches the child process. If the process dies, the last ~200 stderr lines are captured
verbatim. **Never summarise an error into a category and discard the text** — R1's value
is the exact strings, and the capability-gate hypothesis will be argued from them.

### Acceptance length (τ)

Both engines expose Prometheus counters at `/metrics`. Scrape before and after each
`RunCell` and diff:

```python
# specfp8/metrics/base.py
class SpecStats(BaseModel):
    draft_tokens: int
    accepted_tokens: int
    verification_steps: int
    @property
    def tau(self) -> float:        # mean accepted tokens per verification step
        return self.accepted_tokens / self.verification_steps
```

`metrics/vllm.py` and `metrics/sglang.py` map each engine's counter names onto this.
Delta-based scraping means server reuse across cells is safe.

**If a mechanism exposes no counters, τ is recorded as `null`, never estimated.** A
fabricated τ would poison the central result.

---

## 4. Correctness: how R1a actually works

The failure mode we are hunting is *silent*: a missing FP8 dequant produces wrong numbers,
not a crash. "The server started" proves nothing. Three layered tests, run at
**concurrency 1** to minimise batching nondeterminism:

**Test 1 — speculation equivalence (the sharp instrument).**
Compare speculative vs non-speculative output **at the same precision**, greedy decoding,
fixed seed, 32 fixed prompts (16 MT-Bench + 16 GSM8K), **256 max tokens**. The prompt set
is fixed across all cells and committed to the repo. 256 tokens is the minimum needed
for GSM8K chain-of-thought; 64 would truncate reasoning chains before divergence surfaces.
These must match: speculative decoding is
lossless with respect to *the target it verifies against*, whatever precision that target
is in. Divergence here is a real bug, and it is precision-neutral, so it does not confuse
quantization error with a broken kernel.

This is exactly the right discriminator for H1. The missing dequant lives in the
**non-causal prefill wrapper**, which only the draft path touches. Non-speculative FP8-KV
uses the causal path and works. So spec-vs-nonspec at matched FP8-KV precision isolates it.

**Test 2 — quantization damage (reported, not asserted).**
Non-speculative FP8-KV vs non-speculative BF16-KV. Divergence here is *expected* and is
the thing we are measuring, not a failure. Recorded as a number.

**Test 3 — garbage backstop.**
Covers the case where both paths are equally broken and Test 1 passes on matching
nonsense: non-empty output, degenerate-repetition detector, and task accuracy on a GSM8K
subset against a declared floor. An accuracy collapse from ~80% to single digits is
unambiguous.

```python
class CorrectnessResult(BaseModel):
    exact_match_rate: float        # Test 1, vs matched-precision non-spec
    mean_prefix_agreement: float   # tokens agreeing before first divergence
    damage_vs_bf16: float          # Test 2
    task_accuracy: float | None    # Test 3
    verdict: Literal["correct", "degraded", "broken"]
```

**Tolerance, not a boolean.** Greedy decoding is not bit-exact across batch sizes because
batching changes floating-point reduction order. So Test 1 uses a threshold
(exact-match ≥ 0.95 over 32 prompts) and *always reports the raw rate*. A hard equality
assert would produce spurious failures we would waste days chasing.

---

## 5. Two phases

### Phase 1 — `probe.py` (increment 1, gates everything)

Cheap. For every `ServerCell` in `sweeps/compat.yaml`: attempt launch, record verdict,
and if it boots, run the correctness probe and one short τ measurement. Output is the
compatibility matrix.

```
status ∈ {ok, degraded, broken, launch_failed, oom, timeout, unsupported_config}
```

`ok` requires **both** a successful boot **and** `verdict == "correct"`. This is the R1a
rule encoded: launching is not passing.

### Phase 2 — `sweep.py` (increments 2+)

Only runs cells Phase 1 marked `ok`. Boots each server once, then iterates
workloads × concurrency × repeats, writing one summary record plus one per-request file
per `RunCell`.

---

## 6. Data model

Two files per run, both JSONL, both append-only.

**`results/cells.jsonl`** — one summary record per executed cell:

```jsonc
{
  "cell_id": "9f2c...", "ts": "2026-09-20T11:04:12Z", "phase": "sweep",
  "env": {"gpu": "NVIDIA GeForce RTX 4090", "compute_cap": "8.9", "vram_gb": 24,
          "driver": "560.35.03", "cuda": "12.4", "torch": "2.9.0"},
  "engine": {"name": "vllm", "version": "0.21.1", "commit": "a1b2c3d"},
  "config": { /* ServerCell verbatim */ },
  "load": {"workload": "gsm8k", "concurrency": 32, "seed": 1234, "repeat": 0},
  "outcome": {"status": "ok", "error_verbatim": null, "log_path": "logs/9f2c.log"},
  "spec": {"tau": 3.41, "accepted_tokens": 118204, "draft_tokens": 214016,
           "verification_steps": 34664},
  "capacity": {"kv_tokens_max": 131072, "num_gpu_blocks": 8192, "block_size": 16},
  "correctness": { /* CorrectnessResult */ },
  "requests_path": "results/requests/9f2c.jsonl"
}
```

**`results/requests/<cell_id>.jsonl`** — one record per request: `ttft_ms`, `e2e_ms`,
`output_tokens`, `itl_ms[]`, `ok`, `sampling_params` (temperature, top_p, max_tokens as
sent). This is what makes the SLO sweep free.

---

## 6a. Budget tracking (N1)

Every result record includes `wallclock_s` — wall-clock seconds from server boot to
teardown for probe cells, or per-RunCell measurement duration for sweep cells.

`probe.py` and `sweep.py` both:
1. Print a running GPU-time total after each `ServerCell` group completes.
2. On exit, print the cumulative total for the session.
3. Write a one-line append to `results/budget.log`: `{ts, phase, session_gpu_s, cumulative_gpu_s}`.

The harness does **not** enforce a hard stop — free tiers have their own kill switch, and
a mid-cell abort is worse than finishing 5 minutes over. But the running total makes the
spend visible so you can stop manually before starting the next server group.

---

`env` and `engine` are captured by `env.py` at runtime (`nvidia-smi --query-gpu=...`,
`torch.cuda.get_device_capability`, engine `--version`), never hand-written. **`compute_cap`
is first-class per N2a — it is the hypothesis variable.**

---

## 7. Durability and resume (N4)

```python
# specfp8/store.py
def completed_ids(results_dir) -> set[str]   # scan cells.jsonl once at startup
def append(record) -> None                   # write tmp file, fsync, os.replace
```

Atomic rename means a kill mid-write leaves the last good record intact and loses at most
the in-flight cell. `sweep.py` computes its full cell list up front, subtracts
`completed_ids`, and runs the remainder. Re-running the same command after an interruption
is always safe and always correct.

**Consequence for free tiers:** a Lightning session dying at hour 7 costs one cell. Resume
on Modal or a 4090 and the results merge, because `cell_id` is machine-independent and
`env` is recorded per record so cross-machine mixing stays visible in the data.

---

## 8. Load generation

`client.py` — async, `httpx`, fixed concurrency (N in flight at all times, not a fixed
arrival rate) so the concurrency axis means what it says. Per request it records TTFT
(first streamed chunk), inter-token latencies, end-to-end, and output token count.

Warmup requests are issued and discarded before measurement so compilation and CUDA graph
capture do not land in the numbers.

---

## 9. Workloads

```python
class Workload(Protocol):
    def prompts(self, n: int, seed: int) -> list[Prompt]: ...
    def sampling(self) -> dict: ...
    def validate(self, prompt, output) -> bool | None: ...   # Test 3 accuracy
```

- `gsm8k` — public, validator extracts the final numeric answer.
- `mtbench` — public chat prompts, no validator.
- `structured_json` — schema-constrained generation via the engine's guided decoding
  (xgrammar/outlines), validator = "parses and conforms to schema".

**Note:** guided decoding × speculative decoding is itself a compatibility question, so
`guided_decoding: true` cells are part of the Phase 1 matrix, not assumed to work.

---

## 10. Analysis

`analysis/goodput.py` reads `results/requests/*.jsonl` and computes, for each cell and
each SLO in a declared set:

```
goodput(slo) = |{requests meeting slo}| / wall_clock_seconds
```

SLOs swept over e.g. `p95 ITL ≤ {20, 35, 50, 75} ms` and `TTFT ≤ {200, 500, 1000} ms`.
Because this is offline, adding an SLO later costs nothing. Figures are regenerated from
raw logs, never hand-edited.

---

## 11. Reproduction (N6)

```bash
./run.sh probe   --sweep sweeps/compat.yaml     # Phase 1
./run.sh sweep   --sweep sweeps/perf.yaml       # Phase 2, resumable
./run.sh figures                                # regenerate every figure from results/
```

`run.sh` reads `SPECFP8_MODEL_CACHE` and an engine launch profile; nothing else is
environment-dependent.

---

## 12. Increment 1 build order

Only what Phase 1 needs:

1. `env.py`, `store.py`, `cells.py`
2. `launchers/base.py` + `launchers/vllm.py` + `launchers/sglang.py`
3. `client.py` (minimal — concurrency 1 is enough for the probe)
4. `correctness.py` Tests 1 and 3
5. `metrics/*` for τ
6. `probe.py` + `sweeps/compat.yaml`

Deferred to increment 2: `sweep.py`, full concurrency client, `workloads/structured_json`,
`analysis/*`.

---

## 13. Design risks

| Risk | Mitigation |
|---|---|
| Prometheus counter names differ or move between engine versions | Adapter per engine, asserted against a known-good cell at probe time; τ recorded `null` rather than guessed if the mapping fails |
| Greedy decoding not bit-exact → spurious Test 1 failures | Probe at concurrency 1; threshold not equality; always report the raw match rate |
| Both spec and non-spec broken identically → Test 1 passes on nonsense | Test 3 task-accuracy floor as backstop |
| Server boot dominates GPU cost | Group `RunCell`s by `ServerCell`; one boot per group |
| Engine dependency conflicts across versions | Harness never imports engine code; HTTP only |
| A cell hangs instead of failing | Hard timeout on `wait_healthy` and on every request; `timeout` is a recorded status |
| Free-tier interruption mid-sweep | Atomic append + machine-independent `cell_id` + resume |
