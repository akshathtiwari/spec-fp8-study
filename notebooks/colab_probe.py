"""
Colab Probe Runner — copy-paste this into a Colab notebook cell by cell.

Prerequisites:
- Colab Pro with L4 GPU runtime selected
- Runtime > Change runtime type > L4 GPU

This script is structured as sequential cells for a Colab notebook.
Each # %% marks a new cell.
"""

# %% Cell 1: Verify GPU
# Run this first to confirm you have an L4
import subprocess
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout)

# Check compute capability
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    props = torch.cuda.get_device_properties(0)
    print(f"\nCompute capability: {cap[0]}.{cap[1]}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
    if cap >= (8, 9):
        print("\n✓ GPU is Ada-class (SM89+). Full FP8 probe will run.")
    else:
        print(f"\n⚠ GPU is SM{cap[0]}{cap[1]}, not SM89+.")
        print("  FP8 cells will be skipped. BF16 smoke test will still work.")
else:
    print("✗ No CUDA GPU. Change runtime to L4.")

# %% Cell 2: Clone repo and install harness
!git clone https://github.com/akshathtiwari/spec-fp8-study.git
%cd spec-fp8-study
!pip install -e . -q

# %% Cell 3: Install vLLM (for vLLM cells)
# This takes 3-5 minutes
# Use cu124 index to avoid CUDA 13 mismatch on newer Colab runtimes
!pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124 -q
import vllm
print(f"vLLM version: {vllm.__version__}")

# %% Cell 4: Smoke test with mini sweep
# 2 cells (none + ngram), BF16 only, ~10 minutes
!python -m specfp8.probe --sweep sweeps/test_mini.yaml --results results/

# Check results
import json
with open("results/cells.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        print(f"{rec['cell_id']}: {rec['outcome']['status']} "
              f"(mechanism={rec['config']['mechanism']})")

# %% Cell 5: Run vLLM probe cells
# Split the full compat.yaml to run only vLLM cells
# This avoids needing SGLang installed in the same env
import yaml

with open("sweeps/compat.yaml") as f:
    spec = yaml.safe_load(f)

# Override to vLLM only
spec["axes"]["engine"] = ["vllm"]

with open("sweeps/compat_vllm.yaml", "w") as f:
    yaml.dump(spec, f, default_flow_style=False)

print("Created sweeps/compat_vllm.yaml with vLLM-only cells")
!python -m specfp8.probe --sweep sweeps/compat_vllm.yaml --results results/

# %% Cell 6: Check results so far
import json
statuses = {}
with open("results/cells.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        key = f"{rec['config']['mechanism']}|{rec['config']['weight_precision']}|{rec['config']['kv_cache_dtype']}"
        statuses[key] = rec['outcome']['status']

print(f"\n{'Config':<35} {'Status'}")
print("-" * 50)
for k, v in sorted(statuses.items()):
    marker = "✓" if v == "ok" else "✗"
    print(f"{k:<35} {marker} {v}")

# %% Cell 7: Save results to Google Drive (persistence across sessions)
from google.colab import drive
drive.mount('/content/drive')
!cp -r results/ /content/drive/MyDrive/specfp8-results/
print("Results backed up to Google Drive")

# %% Cell 8: Download results locally
from google.colab import files
!tar czf probe_results.tar.gz results/
files.download("probe_results.tar.gz")
"""

# =============================================================
# FOR SGLANG CELLS:
# Start a NEW Colab session (fresh runtime) and run:
# =============================================================

# %% SGLang Cell 1: Install SGLang instead of vLLM
# !pip install sglang[all] -q

# %% SGLang Cell 2: Upload previous results
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r /content/drive/MyDrive/specfp8-results/ results/

# %% SGLang Cell 3: Run SGLang cells (resume skips vLLM cells)
# Create SGLang-only sweep
# import yaml
# with open("sweeps/compat.yaml") as f:
#     spec = yaml.safe_load(f)
# spec["axes"]["engine"] = ["sglang"]
# with open("sweeps/compat_sglang.yaml", "w") as f:
#     yaml.dump(spec, f)
# !python -m specfp8.probe --sweep sweeps/compat_sglang.yaml --results results/
"""
