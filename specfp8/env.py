"""GPU / driver / CUDA / engine version capture.

Never hand-write env fields — everything is queried at runtime so results
are machine-truthful.
"""

from __future__ import annotations

import subprocess
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
    """Capture the current environment.

    Tries nvidia-smi first, falls back to torch.cuda if nvidia-smi is
    broken (common on Colab where nvidia-smi returns exit 12 from
    subprocesses).
    """
    gpu_info = _query_nvidia_smi()
    torch_version, compute_cap = _query_torch()

    # If nvidia-smi failed, fill gaps from torch
    if gpu_info["name"] == "unknown":
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["name"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass

    if gpu_info["vram_mb"] == 0.0:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["vram_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        except (ImportError, AttributeError):
            pass

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

    Tries the structured --query-gpu first; if that fails, tries plain
    nvidia-smi; if that also fails (Colab exit 12), returns defaults
    that capture() fills from torch.cuda.
    """
    # Attempt 1: structured query
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        return {
            "name": parts[0],
            "vram_mb": float(parts[1]),
            "driver": parts[2],
            "cuda": _get_cuda_version(),
        }
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
        pass

    # Attempt 2: parse plain nvidia-smi output
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        )
        return _parse_nvidia_smi_output(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Attempt 3: nvidia-smi broken — return defaults, let capture() fill from torch
    return {
        "name": "unknown",
        "vram_mb": 0.0,
        "driver": _get_driver_version_fallback(),
        "cuda": _get_cuda_version(),
    }


def _parse_nvidia_smi_output(stdout: str) -> dict:
    """Parse GPU info from plain nvidia-smi output."""
    name = "unknown"
    vram_mb = 0.0
    driver = "unknown"
    cuda = "unknown"

    for line in stdout.split("\n"):
        if "Driver Version:" in line:
            try:
                driver = line.split("Driver Version:")[1].split()[0].strip()
            except (IndexError, ValueError):
                pass
            if "CUDA Version:" in line:
                try:
                    cuda = line.split("CUDA Version:")[1].strip().rstrip("|").strip()
                except (IndexError, ValueError):
                    pass

        if "MiB" in line and "/" in line and "|" in line:
            try:
                for part in line.split("|"):
                    if "MiB" in part and "/" in part:
                        total_str = part.split("/")[1].strip().replace("MiB", "").strip()
                        vram_mb = float(total_str)
                        break
            except (IndexError, ValueError):
                pass

    # GPU name from torch (more reliable than parsing nvidia-smi table)
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    if cuda == "unknown":
        cuda = _get_cuda_version()

    return {"name": name, "vram_mb": vram_mb, "driver": driver, "cuda": cuda}


def _get_cuda_version() -> str:
    """Get CUDA runtime version, trying nvcc first, then torch."""
    # Try nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.split("\n"):
            if "release" in line.lower():
                parts = line.split("release")[-1].strip()
                return parts.split(",")[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: torch.version.cuda
    try:
        import torch
        if torch.version.cuda:
            return torch.version.cuda
    except (ImportError, AttributeError):
        pass

    return "unknown"


def _get_driver_version_fallback() -> str:
    """Try to get driver version without nvidia-smi."""
    # Check /proc/driver/nvidia/version
    try:
        with open("/proc/driver/nvidia/version") as f:
            for line in f:
                if "Kernel Module" in line:
                    # "NVRM version: NVIDIA UNIX x86_64 Kernel Module  560.35.03  ..."
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "Module":
                            return parts[i + 1]
    except (FileNotFoundError, IndexError):
        pass

    return "unknown"


def _query_torch() -> tuple[str, str]:
    """Get torch version and compute capability. Imports torch lazily."""
    try:
        import torch

        version = torch.__version__
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            compute_cap = f"{cap[0]}.{cap[1]}"
        else:
            compute_cap = "no-cuda"
        return version, compute_cap
    except ImportError:
        return "not-installed", "unknown"
