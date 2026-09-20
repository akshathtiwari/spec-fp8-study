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
    timeout=4 * 3600,
    volumes={
        "/results": results_vol,
        "/models": model_vol,
    },
)
def run_vllm_probe(sweep_file: str = "compat", retry_failed: bool = False):
    """Run the probe with vLLM engine."""
    import os

    os.environ["SPECFP8_MODEL_CACHE"] = "/models"
    _print_gpu_info()
    repo_dir = _setup_repo()

    if sweep_file == "mini":
        # Smoke test: 2 cells, ~10 min
        sweep_path = "sweeps/test_mini.yaml"
        print("\n=== Running smoke test (test_mini.yaml) ===")
    else:
        # Full vLLM probe: create vLLM-only sweep
        import yaml
        with open("sweeps/compat.yaml") as f:
            spec = yaml.safe_load(f)
        spec["axes"]["engine"] = ["vllm"]
        sweep_path = "/tmp/compat_vllm.yaml"
        with open(sweep_path, "w") as f:
            yaml.dump(spec, f)
        print("\n=== Running full vLLM probe ===")

    import subprocess
    result = subprocess.run(
        ["python", "-m", "specfp8.probe",
         "--sweep", sweep_path,
         "--results", "/results"]
        + (["--retry-failed"] if retry_failed else []),
    )

    # Commit volume so results persist
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
    timeout=4 * 3600,
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
    result = subprocess.run(
        ["python", "-m", "specfp8.probe",
         "--sweep", sweep_path,
         "--results", "/results"]
        + (["--retry-failed"] if retry_failed else []),
    )

    results_vol.commit()
    _print_results()


@app.function(
    volumes={"/results": results_vol},
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
):
    """Entry point: modal run cloud/modal_probe.py [--engine vllm|sglang|status|help] [--sweep mini|compat]"""
    if engine == "help":
        print(dump_vllm_help.remote())
        return
    if engine == "vllm":
        count = run_vllm_probe.remote(sweep_file=sweep, retry_failed=retry_failed)
        print(f"\nDone. {count} cells completed.")
    elif engine == "sglang":
        run_sglang_probe.remote(retry_failed=retry_failed)
    elif engine == "status":
        check_results.remote()
    else:
        print(f"Unknown engine: {engine}. Use 'vllm', 'sglang', or 'status'.")
