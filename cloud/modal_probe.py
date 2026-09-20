"""
Modal GPU runner for the compatibility probe.

Setup:
1. uv tool install modal   (or pip install modal)
2. modal setup             # authenticate
3. modal run cloud/modal_probe.py --engine vllm

This runs the probe on an L40S (SM89, 48GB) using Modal's free $30/mo credits.
Results are persisted to a Modal Volume so they survive across invocations.
"""

import modal

# Modal app definition
app = modal.App("specfp8-probe")

# Persistent volumes for results and model cache
results_vol = modal.Volume.from_name("specfp8-results", create_if_missing=True)
model_vol = modal.Volume.from_name("specfp8-models", create_if_missing=True)

# Container image with vLLM
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "httpx",
        "pydantic>=2.0",
        "pyyaml",
        "pandas",
        "torch",
    )
)

# Container image with SGLang
sglang_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sglang[all]",
        "httpx",
        "pydantic>=2.0",
        "pyyaml",
        "pandas",
        "torch",
    )
)


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

    with open(results_path) as f:
        lines = f.readlines()

    print(f"\nCompleted {len(lines)} cells:")
    for line in lines:
        rec = json.loads(line)
        cfg = rec["config"]
        status = rec["outcome"]["status"]
        tau = rec.get("spec", {}).get("tau", "-")
        tau_str = f"τ={tau:.2f}" if isinstance(tau, (int, float)) else ""
        marker = "✓" if status == "ok" else "✗"
        print(f"  {marker} {cfg['engine']}|{cfg['mechanism']}|"
              f"{cfg['weight_precision']}|{cfg['kv_cache_dtype']} "
              f"→ {status} {tau_str}")

    return len(lines)


@app.function(
    image=vllm_image,
    gpu=modal.gpu.L40S(),
    timeout=4 * 3600,
    volumes={
        "/results": results_vol,
        "/models": model_vol,
    },
)
def run_vllm_probe(sweep_file: str = "compat"):
    """Run the probe with vLLM engine on L40S."""
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
         "--results", "/results"],
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
    gpu=modal.gpu.L40S(),
    timeout=4 * 3600,
    volumes={
        "/results": results_vol,
        "/models": model_vol,
    },
)
def run_sglang_probe():
    """Run the probe with SGLang engine on L40S."""
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
         "--results", "/results"],
    )

    results_vol.commit()
    _print_results()


@app.function(
    volumes={"/results": results_vol},
)
def check_results():
    """Check results from the Modal volume."""
    _print_results()


@app.local_entrypoint()
def main(
    engine: str = "vllm",
    sweep: str = "compat",
):
    """Entry point: modal run cloud/modal_probe.py [--engine vllm|sglang|status] [--sweep mini|compat]"""
    if engine == "vllm":
        count = run_vllm_probe.remote(sweep_file=sweep)
        print(f"\nDone. {count} cells completed.")
    elif engine == "sglang":
        run_sglang_probe.remote()
    elif engine == "status":
        check_results.remote()
    else:
        print(f"Unknown engine: {engine}. Use 'vllm', 'sglang', or 'status'.")
