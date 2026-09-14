"""
Modal GPU runner for the compatibility probe.

Setup:
1. pip install modal
2. modal token new          # authenticate
3. modal run cloud/modal_probe.py

This runs the probe on an L40S (SM89, 48GB) using Modal's free $30/mo credits.
Results are persisted to a Modal Volume so they survive across invocations.
"""

import modal

# Modal app definition
app = modal.App("specfp8-probe")

# Persistent volume for results + model cache
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
    .run_commands("pip install -e /root/spec-fp8-study || true")
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


@app.function(
    image=vllm_image,
    gpu=modal.gpu.L40S(),
    timeout=4 * 3600,  # 4 hour max
    volumes={
        "/results": results_vol,
        "/models": model_vol,
    },
)
def run_vllm_probe():
    """Run the probe with vLLM engine on L40S."""
    import subprocess
    import os

    os.environ["SPECFP8_MODEL_CACHE"] = "/models"

    # Clone and install if not present
    if not os.path.exists("/root/spec-fp8-study"):
        subprocess.run(
            ["git", "clone", "https://github.com/akshathtiwari/spec-fp8-study.git",
             "/root/spec-fp8-study"],
            check=True,
        )
    os.chdir("/root/spec-fp8-study")
    subprocess.run(["pip", "install", "-e", "."], check=True)

    # Create vLLM-only sweep
    import yaml
    with open("sweeps/compat.yaml") as f:
        spec = yaml.safe_load(f)
    spec["axes"]["engine"] = ["vllm"]
    with open("/tmp/compat_vllm.yaml", "w") as f:
        yaml.dump(spec, f)

    # Run probe (results go to volume for persistence)
    subprocess.run(
        ["python", "-m", "specfp8.probe",
         "--sweep", "/tmp/compat_vllm.yaml",
         "--results", "/results"],
        check=True,
    )

    # Commit volume
    results_vol.commit()

    # Print summary
    import json
    with open("/results/cells.jsonl") as f:
        lines = f.readlines()
    print(f"\nCompleted {len(lines)} cells")
    for line in lines:
        rec = json.loads(line)
        print(f"  {rec['config']['mechanism']}|{rec['config']['weight_precision']}|"
              f"{rec['config']['kv_cache_dtype']} → {rec['outcome']['status']}")


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
    import subprocess
    import os

    os.environ["SPECFP8_MODEL_CACHE"] = "/models"

    if not os.path.exists("/root/spec-fp8-study"):
        subprocess.run(
            ["git", "clone", "https://github.com/akshathtiwari/spec-fp8-study.git",
             "/root/spec-fp8-study"],
            check=True,
        )
    os.chdir("/root/spec-fp8-study")
    subprocess.run(["pip", "install", "-e", "."], check=True)

    import yaml
    with open("sweeps/compat.yaml") as f:
        spec = yaml.safe_load(f)
    spec["axes"]["engine"] = ["sglang"]
    with open("/tmp/compat_sglang.yaml", "w") as f:
        yaml.dump(spec, f)

    # Resume: existing vLLM results in the volume get skipped automatically
    subprocess.run(
        ["python", "-m", "specfp8.probe",
         "--sweep", "/tmp/compat_sglang.yaml",
         "--results", "/results"],
        check=True,
    )

    results_vol.commit()


@app.function(
    volumes={"/results": results_vol},
)
def download_results():
    """Download results from the Modal volume to local disk."""
    import json
    import os

    cells_path = "/results/cells.jsonl"
    if not os.path.exists(cells_path):
        print("No results yet.")
        return

    with open(cells_path) as f:
        lines = f.readlines()

    print(f"Total cells: {len(lines)}")
    for line in lines:
        rec = json.loads(line)
        print(f"  {rec['cell_id']}: {rec['config']['engine']}|"
              f"{rec['config']['mechanism']}|{rec['config']['weight_precision']}|"
              f"{rec['config']['kv_cache_dtype']} → {rec['outcome']['status']}")

    # Also check budget
    budget_path = "/results/budget.log"
    if os.path.exists(budget_path):
        print("\nBudget log:")
        with open(budget_path) as f:
            for line in f:
                rec = json.loads(line)
                print(f"  {rec['phase']}: {rec['session_gpu_s']:.0f}s "
                      f"(cumulative: {rec['cumulative_gpu_s']:.0f}s / "
                      f"{rec['cumulative_gpu_s']/3600:.1f}h)")


@app.local_entrypoint()
def main(engine: str = "vllm"):
    """Entry point: modal run cloud/modal_probe.py --engine vllm"""
    if engine == "vllm":
        run_vllm_probe.remote()
    elif engine == "sglang":
        run_sglang_probe.remote()
    elif engine == "status":
        download_results.remote()
    else:
        print(f"Unknown engine: {engine}. Use 'vllm', 'sglang', or 'status'.")
