"""
Modal GPU runner for the compatibility probe.

Setup:
1. uv tool install modal   (or pip install modal)
2. modal setup             # authenticate
3. modal run cloud/modal_probe.py --engine vllm

This runs the probe on an L4 (SM89, 24GB) — Ada-class, which is the compute
capability the study is about. A payment method is required for any GPU.
Results are persisted to a Modal Volume so they survive across invocations.
"""

import modal

# Modal app definition
app = modal.App("specfp8-probe")

# Persistent volumes for results and model cache
results_vol = modal.Volume.from_name("specfp8-results", create_if_missing=True)
model_vol = modal.Volume.from_name("specfp8-models", create_if_missing=True)

# A CUDA *devel* base is required, not a slim image: vLLM's engine core needs
# nvcc at /usr/local/cuda to compile kernels, and aborts with
# "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist"
# without it. Disabling compilation instead would change what is being
# measured, so the toolkit has to be present.
#
# The CUDA major version must match the one the engine's torch wheel was built
# against (torch 2.13.0+cu130 -> CUDA 13). Base image and engine version are
# both pinned so a third party gets the same stack.
CUDA_BASE = "nvidia/cuda:13.0.3-devel-ubuntu24.04"
VLLM_VERSION = "0.29.0"
SGLANG_VERSION = "0.5.20"

HARNESS_DEPS = ["httpx", "pydantic>=2.0", "pyyaml", "pandas"]


def _engine_image(engine_pkg: str) -> modal.Image:
    """CUDA devel base + one engine + the harness's own dependencies.

    torch is left to the engine's own pin rather than requested separately,
    so the resolver cannot pair the engine with a mismatched build.
    """
    return (
        modal.Image.from_registry(CUDA_BASE, add_python="3.11")
        .apt_install("git")
        .pip_install(engine_pkg, *HARNESS_DEPS)
    )


vllm_image = _engine_image(f"vllm=={VLLM_VERSION}")
sglang_image = _engine_image(f"sglang[all]=={SGLANG_VERSION}")

# Cost guardrails. GPU time bills per second, so every GPU function is capped:
# one container so a mistake can never fan out across GPUs, one input per
# container so none is kept warm between calls, and a short idle window.
# Timeouts are sized to the work rather than left permissive — the probe
# resumes from the results volume, so hitting one costs a re-run of the
# in-flight cell, not the sweep.
GPU_GUARDRAILS = dict(
    max_containers=1, single_use_containers=True, scaledown_window=60
)


def _commit_periodically(stop_event, every_s: int = 60):
    """Commit the results volume while a long run is in progress.

    Committing only after the sweep finishes means a timeout, a crash, or a
    preemption throws away every cell already completed and pays for them
    again on the next run. Committing as we go caps that loss at the cell
    currently in flight.
    """
    while not stop_event.wait(every_s):
        try:
            results_vol.commit()
        except Exception as e:  # a failed commit must not kill the probe
            print(f"  (periodic volume commit failed: {e})")


def _setup_repo():
    """Clone (or pull) the repo and install the harness."""
    import subprocess
    import os

    repo_dir = "/root/spec-fp8-study"
    if not os.path.exists(repo_dir):
        subprocess.run(
            ["git", "clone",
             "https://github.com/akshathtiwari/spec-fp8-study.git",
             repo_dir],
            check=True,
        )
    else:
        subprocess.run(["git", "-C", repo_dir, "pull"], check=True)

    os.chdir(repo_dir)
    subprocess.run(["pip", "install", "-e", ".", "-q"], check=True)

    # An editable install landing in site-packages is not visible to the
    # already-running interpreter, so callers that import specfp8 in-process
    # (rather than spawning `python -m specfp8.probe`) would not find it.
    import sys
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    return repo_dir


def _print_gpu_info():
    """Print GPU info for the log."""
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {name} (SM{cap[0]}{cap[1]}, {vram:.0f}GB)")
    else:
        print("WARNING: No CUDA GPU detected")


def _print_results(results_path="/results/cells.jsonl"):
    """Print a summary of results."""
    import json
    import os

    if not os.path.exists(results_path):
        print("No results yet.")
        return 0

    # cells.jsonl is append-only, so a re-run leaves the superseded record in
    # place. Keep only the last record per cell_id or retries look like
    # extra cells.
    latest = {}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            latest[rec["cell_id"]] = rec

    print(f"\nCompleted {len(latest)} cells:")
    for rec in latest.values():
        cfg = rec["config"]
        status = rec["outcome"]["status"]
        tau = rec.get("spec", {}).get("tau", "-")
        tau_str = f"τ={tau:.2f}" if isinstance(tau, (int, float)) else ""
        marker = "✓" if status == "ok" else "✗"
        print(f"  {marker} {cfg['engine']}|{cfg['mechanism']}|"
              f"{cfg['weight_precision']}|{cfg['kv_cache_dtype']} "
              f"→ {status} {tau_str}")

    return len(latest)


@app.function(
    image=vllm_image,
    gpu="L4",
    timeout=150 * 60,
    **GPU_GUARDRAILS,
    volumes={"/results": results_vol, "/models": model_vol},
)
def run_vllm_sweep(sweep_file: str = "perf", force: bool = False,
                   boot_repeats: int = 1):
    """Phase 2 performance sweep: boot once per ServerCell, run many."""
    import os, subprocess, threading

    os.environ["SPECFP8_MODEL_CACHE"] = "/models"
    _print_gpu_info()
    _setup_repo()

    source = f"sweeps/{sweep_file}.yaml"
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"No sweep at {source}. Available: {sorted(os.listdir('sweeps'))}")
    print(f"\n=== Phase 2 sweep: {source} ===")

    stop = threading.Event()
    committer = threading.Thread(target=_commit_periodically, args=(stop,),
                                 daemon=True)
    committer.start()
    try:
        for attempt in range(max(1, boot_repeats)):
            if boot_repeats > 1:
                print(f"\n########## BOOT PASS {attempt + 1} of "
                      f"{boot_repeats} ##########")
            result = subprocess.run(
                ["python", "-u", "-m", "specfp8.sweep",
                 "--sweep", source, "--results", "/results"]
                # Every pass must re-measure, otherwise pass 2 sees pass 1's
                # records and skips -- which would defeat the whole test.
                + (["--force"] if (force or boot_repeats > 1) else []),
            )
    finally:
        stop.set()
        committer.join(timeout=5)
        results_vol.commit()

    if result.returncode != 0:
        print(f"\nSweep exited with code {result.returncode}")
    return result.returncode


@app.function(
    image=vllm_image,
    gpu="L4",
    # 120 min: the quality stage adds several minutes per cell on top of boot
    # and gate. Resume plus per-60s volume commits cap the cost of hitting
    # this at one re-run cell, so headroom is cheaper than a clipped sweep.
    timeout=120 * 60,
    **GPU_GUARDRAILS,
    volumes={
        "/results": results_vol,
        "/models": model_vol,
    },
)
def run_vllm_probe(sweep_file: str = "compat", retry_failed: bool = False,
                   quality: bool = False, repeats: int = 1,
                   force: bool = False):
    """Run the probe with vLLM engine."""
    import os

    os.environ["SPECFP8_MODEL_CACHE"] = "/models"
    _print_gpu_info()
    repo_dir = _setup_repo()

    # Accept any sweep by name; "mini" is an alias kept for muscle memory.
    named = {"mini": "sweeps/test_mini.yaml"}
    source = named.get(sweep_file, f"sweeps/{sweep_file}.yaml")
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"No sweep at {source}. Available: "
            f"{sorted(os.listdir('sweeps'))}"
        )

    # This container only has vLLM installed, so drop any sglang cells rather
    # than letting them fail as if they were incompatible configurations.
    import yaml
    with open(source) as f:
        spec = yaml.safe_load(f)
    engines = spec.get("axes", {}).get("engine", [])
    if list(engines) != ["vllm"]:
        spec["axes"]["engine"] = ["vllm"]
        sweep_path = f"/tmp/{os.path.basename(source)}"
        with open(sweep_path, "w") as f:
            yaml.dump(spec, f)
        print(f"\n=== Running {source} (vLLM cells only) ===")
    else:
        sweep_path = source
        print(f"\n=== Running {source} ===")

    import subprocess
    import threading
    stop_commits = threading.Event()
    committer = threading.Thread(
        target=_commit_periodically, args=(stop_commits,), daemon=True
    )
    committer.start()
    try:
        for attempt in range(max(1, repeats)):
            if repeats > 1:
                print(f"\n===== repeat {attempt + 1} of {repeats} =====")
            result = subprocess.run(
            # -u is load-bearing: Python block-buffers stdout when it is a
            # pipe, so without it the probe's per-cell progress does not
            # reach Modal's logs until the process exits. A GPU run that
            # cannot be observed while it bills is indistinguishable from a
            # hung one.
            ["python", "-u", "-m", "specfp8.probe",
             "--sweep", sweep_path,
             "--results", "/results"]
            + (["--retry-failed"] if retry_failed else [])
            + (["--quality"] if quality else [])
            # Every repeat after the first must bypass the staleness skip,
            # otherwise the cells it just completed are treated as done.
            + (["--force"] if (force or repeats > 1) else []),
            )
    finally:
        stop_commits.set()
        committer.join(timeout=5)
        results_vol.commit()

    count = _print_results()

    # Check budget
    budget_path = "/results/budget.log"
    if os.path.exists(budget_path):
        import json
        print("\nBudget log:")
        with open(budget_path) as f:
            for line in f:
                rec = json.loads(line)
                print(f"  {rec['phase']}: {rec['session_gpu_s']:.0f}s "
                      f"(cumulative: {rec['cumulative_gpu_s']:.0f}s / "
                      f"{rec['cumulative_gpu_s']/3600:.1f}h)")

    if result.returncode != 0:
        print(f"\nProbe exited with code {result.returncode}")

    return count


@app.function(
    image=sglang_image,
    gpu="L4",
    timeout=90 * 60,
    **GPU_GUARDRAILS,
    volumes={
        "/results": results_vol,
        "/models": model_vol,
    },
)
def run_sglang_probe(retry_failed: bool = False):
    """Run the probe with SGLang engine."""
    import os
    import subprocess

    os.environ["SPECFP8_MODEL_CACHE"] = "/models"
    _print_gpu_info()
    repo_dir = _setup_repo()

    import yaml
    with open("sweeps/compat.yaml") as f:
        spec = yaml.safe_load(f)
    spec["axes"]["engine"] = ["sglang"]
    sweep_path = "/tmp/compat_sglang.yaml"
    with open(sweep_path, "w") as f:
        yaml.dump(spec, f)

    print("\n=== Running SGLang probe ===")
    # Resume: existing vLLM results in the volume get skipped automatically
    import threading
    stop_commits = threading.Event()
    committer = threading.Thread(
        target=_commit_periodically, args=(stop_commits,), daemon=True
    )
    committer.start()
    try:
        for attempt in range(max(1, repeats)):
            if repeats > 1:
                print(f"\n===== repeat {attempt + 1} of {repeats} =====")
            result = subprocess.run(
            # -u is load-bearing: Python block-buffers stdout when it is a
            # pipe, so without it the probe's per-cell progress does not
            # reach Modal's logs until the process exits. A GPU run that
            # cannot be observed while it bills is indistinguishable from a
            # hung one.
            ["python", "-u", "-m", "specfp8.probe",
             "--sweep", sweep_path,
             "--results", "/results"]
            + (["--retry-failed"] if retry_failed else []),
        )
    finally:
        stop_commits.set()
        committer.join(timeout=5)
        results_vol.commit()
    _print_results()


@app.function(
    image=vllm_image,
    gpu="L4",
    timeout=30 * 60,
    **GPU_GUARDRAILS,
    volumes={"/models": model_vol},
)
def diagnose(mechanism: str = "ngram") -> str:
    """Boot one cell and dump the raw signals the probe depends on.

    The probe reported KV capacity None, tau null, and 0% task accuracy.
    Each of those depends on a string the harness expects the engine to
    emit — a log line, a counter name, an answer format — and all three are
    currently guesses. This boots a server once and prints what the engine
    actually produces, so they can be fixed against evidence rather than
    re-guessed one GPU run at a time.
    """
    import asyncio
    import os

    os.environ["SPECFP8_MODEL_CACHE"] = "/models"
    _setup_repo()

    from specfp8.cells import expand, cell_id
    from specfp8.client import probe_run
    from specfp8.correctness import get_all_prompts, get_gsm8k_answers
    from specfp8.launchers.base import find_free_port
    from specfp8.launchers.vllm import VllmLauncher
    from specfp8.metrics.base import scrape_prometheus

    cell = next(
        c for c in expand("sweeps/test_mini.yaml") if c.mechanism == mechanism
    )
    launcher = VllmLauncher()
    port = find_free_port()
    out: list[str] = [f"### cell: {cell.mechanism} {cell.weight_precision}"]
    out.append("argv: " + " ".join(launcher.argv(cell, port)))

    handle = launcher.start(cell, cell_id(cell), port, "logs")
    try:
        health = launcher.wait_healthy(handle, timeout_s=600)
        out.append(f"healthy={health.ok} status={health.status}")
        if not health.ok:
            return "\n".join(out)

        # (1) KV capacity: which startup lines actually mention blocks/cache?
        out.append("\n### startup log lines mentioning blocks/cache/memory")
        with open(handle.log_path) as f:
            for line in f:
                low = line.lower()
                if any(k in low for k in
                       ("gpu block", "kv cache", "num_gpu_blocks",
                        "kv_cache_size", "concurrency", "gpu memory")):
                    out.append("  " + line.rstrip()[:200])

        # (2) tau: the real counter names, before and after load
        raw_before = scrape_prometheus(f"{handle.base_url}/metrics")
        prompts = get_all_prompts()[:5]
        result = asyncio.run(
            probe_run(handle.base_url, cell.model, prompts,
                      {"temperature": 0, "max_tokens": 256},
                      n_warmup=1, timeout_s=180.0)
        )
        raw_after = scrape_prometheus(f"{handle.base_url}/metrics")

        out.append("\n### spec/accept/draft counters (before -> after)")
        keys = sorted(
            k for k in set(raw_before) | set(raw_after)
            if any(t in k.lower()
                   for t in ("spec", "accept", "draft", "num_token"))
        )
        for k in keys or ["<none matched>"]:
            out.append(f"  {k}: {raw_before.get(k)} -> {raw_after.get(k)}")
        out.append(f"  (total counters exposed: {len(raw_after)})")

        # (3) accuracy: what the model actually returns vs what is expected
        answers = get_gsm8k_answers()[:5]
        out.append("\n### generated outputs vs expected answers")
        for i, (text, want) in enumerate(zip(result.outputs, answers)):
            req = result.requests[i]
            out.append(f"\n--- [{i}] ok={req.ok} tokens={req.output_tokens} "
                       f"expected={want!r}")
            out.append(f"    {text[:700]!r}")
    finally:
        launcher.stop(handle)

    return "\n".join(out)


@app.function(
    image=vllm_image,
    timeout=60 * 60,
    max_containers=1,
    volumes={"/models": model_vol},
)
def prefetch_models(sweep_file: str = "h1") -> str:
    """Download every checkpoint a sweep needs, on CPU, before any GPU run.

    Weights were being pulled inside the GPU container: the first H1 cell
    spent its entire 300s health budget downloading ~8GB of Qwen3-4B while
    an L4 billed by the second, then timed out without ever serving. The
    bytes are identical whichever container fetches them, so fetch them
    where the GPU is not running and let the cells start warm.
    """
    import os
    _setup_repo()
    from specfp8.cells import expand

    named = {"mini": "sweeps/test_mini.yaml"}
    source = named.get(sweep_file, f"sweeps/{sweep_file}.yaml")
    cells = expand(source)
    wanted = sorted(
        {c.model for c in cells}
        | {c.draft_model for c in cells if c.draft_model}
    )

    from huggingface_hub import snapshot_download
    out = [f"{source}: {len(wanted)} checkpoint(s) to warm"]
    for repo in wanted:
        try:
            # cache_dir matches what the launcher passes as --download-dir,
            # so the engine finds these instead of re-downloading.
            path = snapshot_download(repo_id=repo, cache_dir="/models")
            size = sum(
                os.path.getsize(os.path.join(d, f))
                for d, _, fs in os.walk(path) for f in fs
                if os.path.exists(os.path.join(d, f))
            )
            out.append(f"  OK      {repo}  ({size / 1024**3:.2f} GiB)")
        except Exception as e:
            out.append(f"  FAILED  {repo}  -> {type(e).__name__}: {e}")

    model_vol.commit()
    return "\n".join(out)


@app.function(
    image=vllm_image,
    gpu="L4",
    timeout=45 * 60,
    **GPU_GUARDRAILS,
    volumes={"/models": model_vol},
)
def determinism_check() -> str:
    """T18a: locate the nondeterminism that broke Test 1's premise.

    Speculative and non-speculative outputs agreed on only 34.4% of prompts
    at temperature 0, but that number alone cannot say whether speculation
    caused it. Repeat the *same* non-speculative config twice within one
    boot and once across a restart:

      A vs B differ            -> request-level (batching, prefix cache,
                                  server state); nothing to do with
                                  speculation, and Test 1 is unusable as
                                  specified
      A vs B same, A vs C differ -> boot-level (autotuning, compile choices);
                                  Test 1 works only within a single server
      both identical           -> nondeterminism is speculation-specific,
                                  and Test 1's premise survives for
                                  same-boot comparisons
    """
    import asyncio
    import os

    os.environ["SPECFP8_MODEL_CACHE"] = "/models"
    _setup_repo()

    from specfp8.cells import expand, cell_id
    from specfp8.client import probe_run
    from specfp8.correctness import get_all_prompts
    from specfp8.launchers.base import find_free_port
    from specfp8.launchers.vllm import VllmLauncher
    from specfp8.probe import SAMPLING_PARAMS

    cell = next(c for c in expand("sweeps/test_mini.yaml") if c.mechanism == "none")
    launcher = VllmLauncher()
    prompts = get_all_prompts()
    cid = cell_id(cell)
    out: list[str] = [
        f"config: {cell.mechanism}|{cell.weight_precision}|{cell.kv_cache_dtype}",
        f"model: {cell.model}",
        f"sampling: {SAMPLING_PARAMS}",
    ]

    def run_pass(handle):
        return asyncio.run(
            probe_run(handle.base_url, cell.model, prompts,
                      SAMPLING_PARAMS, n_warmup=3, timeout_s=180.0)
        ).outputs

    def compare(label, x, y):
        n = min(len(x), len(y))
        same = sum(1 for i in range(n) if x[i] == y[i])
        out.append(f"\n### {label}: {same}/{n} identical = {same/max(n,1):.1%}")
        for i in range(n):
            if x[i] != y[i]:
                p = next((k for k in range(min(len(x[i]), len(y[i])))
                          if x[i][k] != y[i][k]), 0)
                out.append(f"  first divergence at prompt {i}, char {p}")
                out.append(f"    run1: {x[i][p:p+80]!r}")
                out.append(f"    run2: {y[i][p:p+80]!r}")
                break
        return same / max(n, 1)

    # --- boot 1: two passes against the same server ---
    port = find_free_port()
    handle = launcher.start(cell, cid, port, "logs")
    try:
        h = launcher.wait_healthy(handle, timeout_s=600)
        if not h.ok:
            return "\n".join(out + [f"boot 1 failed: {h.status}"])

        # Prefix caching reuses KV across requests and is a prime suspect if
        # the two same-server passes disagree.
        with open(handle.log_path) as f:
            for line in f:
                if "prefix" in line.lower() and "cach" in line.lower():
                    out.append("  log: " + line.rstrip()[:160])

        a = run_pass(handle)
        b = run_pass(handle)
    finally:
        launcher.stop(handle)

    # --- boot 2: same config, fresh server ---
    port = find_free_port()
    handle2 = launcher.start(cell, cid, port, "logs")
    try:
        h2 = launcher.wait_healthy(handle2, timeout_s=600)
        if not h2.ok:
            return "\n".join(out + [f"boot 2 failed: {h2.status}"])
        c = run_pass(handle2)
    finally:
        launcher.stop(handle2)

    within = compare("A vs B  (same server, repeated)", a, b)
    across = compare("A vs C  (same config, fresh server)", a, c)

    out.append("\n### verdict")
    if within < 0.99:
        out.append("  Request-level nondeterminism: the same server returns")
        out.append("  different text for identical input. Speculation is not")
        out.append("  the cause, and exact-match Test 1 cannot work as specified.")
    elif across < 0.99:
        out.append("  Boot-level nondeterminism: deterministic within a server,")
        out.append("  not across restarts. Test 1 is valid only when both arms")
        out.append("  are measured against the same running server.")
    else:
        out.append("  Deterministic within and across boots, so the 34.4%")
        out.append("  spec-vs-nonspec gap is speculation-specific and Test 1's")
        out.append("  premise holds for same-boot comparisons.")
    return "\n".join(out)


@app.function(
    image=vllm_image, volumes={"/results": results_vol}, timeout=15 * 60
)
def backend_report() -> str:
    """Report which attention backend served each cell. CPU only."""
    _setup_repo()
    from specfp8.analysis.backend_report import analyse, format_report
    return format_report(analyse("/results"))


@app.function(image=vllm_image, timeout=15 * 60, max_containers=1)
def list_attention_backends() -> str:
    """Enumerate the attention backends this vLLM accepts.

    H1 is a claim about FlashInfer specifically, but every cell so far ran
    with attn_backend=auto, so the backend that actually served is unknown
    and FlashInfer's FP8 path may never have been exercised. Naming the
    backends is the prerequisite for testing them, and guessing the spelling
    would burn a GPU run on an argparse error.

    Run in a subprocess rather than imported here: the enum lives in vLLM,
    and importing it would pull engine code into the runner process.
    """
    import subprocess
    import sys

    script = (
        "from vllm.v1.attention.backends.registry import AttentionBackendEnum as E\n"
        "print('\\n'.join(sorted(m.name for m in E)))\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    # Whether FlashInfer is even importable decides what a FLASHINFER cell
    # failure would mean: a missing SM89 kernel (H1's claim) or simply an
    # absent optional dependency (no evidence either way).
    probe = (
        "import importlib\n"
        "for mod in ('flashinfer','flash_attn','triton'):\n"
        "    try:\n"
        "        m = importlib.import_module(mod)\n"
        "        print(f'  {mod}: {getattr(m, \"__version__\", \"?\")}')\n"
        "    except Exception as e:\n"
        "        print(f'  {mod}: ABSENT ({type(e).__name__})')\n"
    )
    r2 = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    return (
        f"### AttentionBackendEnum members\n{r.stdout}\n"
        f"### backend libraries present\n{r2.stdout}{r2.stderr[-500:]}\n"
        f"### stderr\n{r.stderr[-800:]}"
    )


@app.function(
    image=vllm_image, volumes={"/results": results_vol}, timeout=15 * 60
)
def correctness_matrix(model: str = "Qwen/Qwen3-4B") -> str:
    """Run Tests 1 and 2 over stored outputs. CPU only — no GPU needed."""
    _setup_repo()
    from specfp8.analysis.correctness_matrix import analyse, format_report
    return format_report(analyse("/results", model=model or None))


@app.function(volumes={"/results": results_vol}, timeout=10 * 60)
def compare_outputs() -> str:
    """Compare stored per-request outputs across cells, pairwise by index.

    This is Test 1 in miniature: at temperature 0, speculative decoding is
    supposed to be output-equivalent to non-speculative decoding, and the
    design leans on that (exact-match threshold 0.95) as the sharp
    instrument for detecting a broken kernel. Worth checking directly
    against stored text rather than inferring it from aggregate accuracy.
    """
    import json
    import os

    req_dir = "/results/requests"
    if not os.path.isdir(req_dir):
        return "No per-request outputs stored yet."

    cells = {}
    for fn in sorted(os.listdir(req_dir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(req_dir, fn)) as f:
            cells[fn[:-6]] = [json.loads(line) for line in f if line.strip()]

    # Label each cell by its config so the comparison is readable.
    labels = {}
    cells_path = "/results/cells.jsonl"
    if os.path.exists(cells_path):
        with open(cells_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                c = r["config"]
                labels[r["cell_id"]] = (
                    f"{c['mechanism']}|{c['weight_precision']}|{c['kv_cache_dtype']}"
                )

    out = [f"{len(cells)} cells with stored outputs"]
    for cid, recs in cells.items():
        out.append(f"  {cid} ({labels.get(cid, '?')}): {len(recs)} requests")

    ids = list(cells)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = cells[ids[i]], cells[ids[j]]
            la = labels.get(ids[i], ids[i])
            lb = labels.get(ids[j], ids[j])
            n = min(len(a), len(b))
            same = sum(
                1 for k in range(n)
                if a[k]["output_text"] == b[k]["output_text"]
            )
            out.append(f"\n### {la}  vs  {lb}")
            out.append(f"  exact match: {same}/{n} = {same/max(n,1):.1%}")
            shown = 0
            for k in range(n):
                if a[k]["output_text"] == b[k]["output_text"]:
                    continue
                if shown >= 2:
                    break
                shown += 1
                ta, tb = a[k]["output_text"], b[k]["output_text"]
                # Report where they first diverge, not just that they did.
                p = next(
                    (x for x in range(min(len(ta), len(tb))) if ta[x] != tb[x]),
                    min(len(ta), len(tb)),
                )
                out.append(f"  [{k}] diverges at char {p}")
                out.append(f"      common: ...{ta[max(0,p-60):p]!r}")
                out.append(f"      {la}: {ta[p:p+90]!r}")
                out.append(f"      {lb}: {tb[p:p+90]!r}")
    return "\n".join(out)


@app.function(
    volumes={"/results": results_vol},
    timeout=10 * 60,
)
def check_results():
    """Check results from the Modal volume."""
    import json
    import os
    results_path = "/results/cells.jsonl"
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                rec = json.loads(line)
                print(json.dumps(rec, indent=2))
                print("---")
    else:
        print("No results file found.")
    _print_results()


@app.function(image=vllm_image, gpu="L4", timeout=900)
def dump_vllm_help() -> str:
    """Introspect vLLM's CLI surface so launcher flags can be verified
    against the exact installed version. Needs a GPU: vLLM infers the
    device type while constructing the arg parser."""
    import subprocess
    import vllm

    out = [f"VLLM_VERSION={vllm.__version__}", "=" * 70]

    # Authoritative flag list: every --flag vllm serve accepts
    r = subprocess.run(
        ["vllm", "serve", "--help=all"], capture_output=True, text=True
    )
    import re
    flags = sorted(set(re.findall(r"--[a-z0-9][a-z0-9-]+", r.stdout)))
    out += ["## all vllm serve flags", *(f"  {f}" for f in flags), "=" * 70]

    # Choices for the flags this study actually varies
    for section in ["dtype", "seed", "quantization", "revision",
                    "kv-cache-dtype", "attention-backend", "download-dir"]:
        rr = subprocess.run(
            ["vllm", "serve", f"--help={section}"], capture_output=True, text=True
        )
        out += [f"## --help={section}", rr.stdout.strip()[:1200], "-" * 50]
    out.append("=" * 70)

    # SpeculativeConfig fields — needed to build --speculative-config JSON
    try:
        from vllm.config import SpeculativeConfig
        import dataclasses
        out.append("## SpeculativeConfig fields")
        for f in dataclasses.fields(SpeculativeConfig):
            out.append(f"  {f.name}: {f.type}")
    except Exception as e:
        out.append(f"SpeculativeConfig introspection failed: {e}")

    return "\n".join(out)


@app.local_entrypoint()
def main(
    engine: str = "vllm",
    sweep: str = "compat",
    retry_failed: bool = False,
    quality: bool = False,
    repeats: int = 1,
    force: bool = False,
):
    """Entry point: modal run cloud/modal_probe.py [--engine vllm|sglang|status|help] [--sweep mini|compat]"""
    if engine == "help":
        print(dump_vllm_help.remote())
        return
    if engine == "which-backend":
        print(backend_report.remote())
        return
    if engine == "backends":
        print(list_attention_backends.remote())
        return
    if engine == "correctness":
        print(correctness_matrix.remote(model=sweep if sweep != "compat" else "Qwen/Qwen3-4B"))
        return
    if engine == "sweep":
        rc = run_vllm_sweep.remote(sweep_file=sweep, force=force,
                                   boot_repeats=repeats)
        print(f"\nSweep finished (exit {rc}).")
        return
    if engine == "prefetch":
        print(prefetch_models.remote(sweep_file=sweep))
        return
    if engine == "determinism":
        print(determinism_check.remote())
        return
    if engine == "compare":
        print(compare_outputs.remote())
        return
    if engine == "diagnose":
        print(diagnose.remote(mechanism=sweep if sweep != "compat" else "ngram"))
        return
    if engine == "vllm":
        count = run_vllm_probe.remote(sweep_file=sweep, retry_failed=retry_failed,
                                      quality=quality, repeats=repeats,
                                      force=force)
        print(f"\nDone. {count} cells completed.")
    elif engine == "sglang":
        run_sglang_probe.remote(retry_failed=retry_failed)
    elif engine == "status":
        check_results.remote()
    else:
        print(f"Unknown engine: {engine}. Use 'vllm', 'sglang', or 'status'.")
