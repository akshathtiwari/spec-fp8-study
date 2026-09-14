"""GPU / driver / CUDA / engine version capture.

Never hand-write env fields — everything is queried at runtime so results
are machine-truthful.
"""

from __future__ import annotations

import subprocess
import json
from pydantic import BaseModel


class EnvInfo(BaseModel):
    """Snapshot of the hardware + software environment at measurement time."""

    gpu: str                    # e.g. "NVIDIA GeForce RTX 4090"
    compute_cap: str            # e.g. "8.9" — THE hypothesis variable
    vram_gb: float              # total GPU memory in GB
    driver: str                 # e.g. "560.35.03"
    cuda: str                   # CUDA runtime version
    torch: str                  # PyTorch version (from the harness env)


def capture() -> EnvInfo:
    """Capture the current environment. Requires nvidia-smi and torch."""
    gpu_info = _query_nvidia_smi()
    torch_version, compute_cap = _query_torch()

    return EnvInfo(
        gpu=gpu_info["name"],
        compute_cap=compute_cap,
        vram_gb=round(gpu_info["vram_mb"] / 1024, 1),
        driver=gpu_info["driver"],
        cuda=gpu_info["cuda"],
        torch=torch_version,
    )


def _query_nvidia_smi() -> dict:
    """Query nvidia-smi for GPU name, VRAM, driver, and CUDA version."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Take the first GPU if multiple
    line = result.stdout.strip().split("\n")[0]
    parts = [p.strip() for p in line.split(",")]

    # CUDA version from nvidia-smi
    cuda_cmd = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    # Actually get CUDA version from nvcc or nvidia-smi header
    cuda_version = _get_cuda_version()

    return {
        "name": parts[0],
        "vram_mb": float(parts[1]),
        "driver": parts[2],
        "cuda": cuda_version,
    }


def _get_cuda_version() -> str:
    """Get CUDA runtime version, trying nvcc first, then nvidia-smi."""
    # Try nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.split("\n"):
            if "release" in line.lower():
                # "Cuda compilation tools, release 12.4, V12.4.131"
                parts = line.split("release")[-1].strip()
                return parts.split(",")[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: parse nvidia-smi header
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.split("\n"):
            if "CUDA Version" in line:
                # "| NVIDIA-SMI 560.35.03    Driver Version: 560.35.03    CUDA Version: 12.6 |"
                return line.split("CUDA Version:")[-1].strip().rstrip("|").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return "unknown"


def _query_torch() -> tuple[str, str]:
    """Get torch version and compute capability. Imports torch lazily."""
    try:
        import torch  # noqa: delayed import — harness doesn't depend on torch at install

        version = torch.__version__
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            compute_cap = f"{cap[0]}.{cap[1]}"
        else:
            compute_cap = "no-cuda"
        return version, compute_cap
    except ImportError:
        return "not-installed", "unknown"
