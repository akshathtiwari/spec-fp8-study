#!/usr/bin/env bash
# Cloud GPU setup script — works on Lightning AI, Paperspace, Vast.ai, RunPod
# Usage: bash cloud/setup.sh [vllm|sglang|both]
set -euo pipefail

ENGINE="${1:-both}"
REPO_DIR="${HOME}/spec-fp8-study"
MODEL_CACHE="${HOME}/models"

echo "============================================"
echo "specfp8 cloud setup"
echo "Engine: ${ENGINE}"
echo "============================================"

# 1. Verify GPU
echo ""
echo "--- GPU check ---"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
python3 -c "
import torch
cap = torch.cuda.get_device_capability()
print(f'Compute capability: {cap[0]}.{cap[1]}')
assert cap[0] >= 8, f'Need SM80+ for FP8. Got SM{cap[0]}{cap[1]}'
print('✓ GPU supports FP8')
if cap == (8, 9):
    print('✓ SM89 (Ada) — this is the H1 target hardware')
"

# 2. Clone repo (or pull if exists)
echo ""
echo "--- Repo setup ---"
if [ -d "${REPO_DIR}" ]; then
    cd "${REPO_DIR}" && git pull
    echo "Updated existing repo"
else
    git clone https://github.com/akshathtiwari/spec-fp8-study.git "${REPO_DIR}"
    echo "Cloned repo"
fi
cd "${REPO_DIR}"

# 3. Install harness
pip install -e . -q
echo "✓ Harness installed"

# 4. Install engine(s)
if [ "${ENGINE}" = "vllm" ] || [ "${ENGINE}" = "both" ]; then
    echo ""
    echo "--- Installing vLLM ---"
    pip install vllm -q
    python3 -c "import vllm; print(f'vLLM {vllm.__version__}')"
fi

if [ "${ENGINE}" = "sglang" ] || [ "${ENGINE}" = "both" ]; then
    echo ""
    echo "--- Installing SGLang ---"
    pip install "sglang[all]" -q
    python3 -c "import sglang; print(f'SGLang installed')"
fi

# 5. Pre-download models
echo ""
echo "--- Model download ---"
export SPECFP8_MODEL_CACHE="${MODEL_CACHE}"
mkdir -p "${MODEL_CACHE}"

python3 -c "
from huggingface_hub import snapshot_download
import os

cache = os.environ['SPECFP8_MODEL_CACHE']
models = [
    'Qwen/Qwen3-4B',
    'z-lab/Qwen3-4B-DFlash-b16',
    # Add EAGLE-3 checkpoint once confirmed (U3)
]
for m in models:
    print(f'Downloading {m}...')
    try:
        snapshot_download(m, cache_dir=cache)
        print(f'  ✓ {m}')
    except Exception as e:
        print(f'  ✗ {m}: {e}')
"

# 6. Smoke test
echo ""
echo "--- Smoke test ---"
export SPECFP8_MODEL_CACHE="${MODEL_CACHE}"
python3 -c "
from specfp8.cells import expand
cells = expand('sweeps/test_mini.yaml')
print(f'test_mini.yaml expands to {len(cells)} cells')
from specfp8.env import capture
try:
    env = capture()
    print(f'GPU: {env.gpu} (SM{env.compute_cap})')
    print(f'VRAM: {env.vram_gb} GB')
except Exception as e:
    print(f'Env capture partial: {e}')
print('✓ Harness ready')
"

echo ""
echo "============================================"
echo "Setup complete. Next steps:"
echo ""
echo "  # Smoke test (2 cells, ~10 min):"
echo "  ./run.sh probe --sweep sweeps/test_mini.yaml"
echo ""
echo "  # Full probe (~40 cells, ~3-4 hrs):"
echo "  ./run.sh probe --sweep sweeps/compat.yaml"
echo ""
echo "  # Check results:"
echo "  cat results/cells.jsonl | python3 -m json.tool"
echo "============================================"
