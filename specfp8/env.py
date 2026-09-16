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
    """Query nvidia-smi for GPU name, VRAM, driver, and CUDA version.

    Tries the structured --query-gpu first; if that fails (exit 12 on
    some driver versions), falls back to parsing the plain nvidia-smi
    header table.
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        name, vram_mb, driver = parts[0], float(parts[1]), parts[2]
    except (subprocess.CalledProcessError, ValueError, IndexError):
        # Fallback: parse the plain nvidia-smi output
        name, vram_mb, driver = _parse_nvidia_smi_plain()

    cuda_version = _get_cuda_version()

    return {
        "name": name,
        "vram_mb": vram_mb,
        "driver": driver,
        "cuda": cuda_version,
    }


def _parse_nvidia_smi_plain() -> tuple[str, float, str]:
    """Parse GPU info from plain nvidia-smi output (no --query-gpu)."""
    result = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=True
    )
    lines = result.stdout.split("\n")

    name = "unknown"
    vram_mb = 0.0
    driver = "unknown"

    for line in lines:
        # Driver line: "| NVIDIA-SMI 560.35.03    Driver Version: 560.35.03    CUDA Version: 12.6 |"
        if "Driver Version:" in line:
            try:
                driver = line.split("Driver Version:")[1].split()[0].strip()
            except (IndexError, ValueError):
                pass
        # GPU name line: "|   0  Tesla T4   ..." or "|   0  NVIDIA L4   ..."
        if "MiB" in line and "|" in line:
            try:
                # Memory line: "| N/A   47C    P0    27W /  70W |    0MiB / 15360MiB |      0%      Default |"
                parts = line.split("|")
                for part in parts:
                    if "MiB" in part and "/" in part:
                        total_str = part.split("/")[1].strip().replace("MiB", "").strip()
                        vram_mb = float(total_str)
                        break
            except (IndexError, ValueError):
                pass

    # Try to get GPU name from torch if nvidia-smi parsing is tricky
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except ImportError:
        # Last resort: grep nvidia-smi output for known GPU names
        for line in lines:
            for gpu in ["T4", "L4", "L40S", "A100", "H100", "RTX 4090", "RTX 3090"]:
                if gpu in line and "%" not in line:
                    name = line.split("|")[1].strip() if "|" in line else gpu
                    break

    return name, vram_mb, driver


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
