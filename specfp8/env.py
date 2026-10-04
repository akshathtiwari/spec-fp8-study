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

    # Physical-device identity and host power/clock configuration.
    #
    # Added 2026-09-29 to close item 5 of docs/reproducibility.md, which is
    # the paper's largest open question. Section 4 reports a 13.92% CV in
    # speculative throughput across boots of an identical configuration and
    # has to concede that boot-to-boot and host-to-host components cannot be
    # separated, because runs used rented instances and nothing recorded
    # WHICH physical GPU served them.
    #
    # gpu_uuid is the field that settles it. Two boots sharing a uuid that
    # still differ by 14% is a boot effect. Dispersion that tracks uuid
    # changes is a host effect. Without it the question is unanswerable no
    # matter how many boots are collected, which is why 12 boots did not
    # settle it.
    #
    # All optional with None defaults: the store is append-only and every
    # record written before this date lacks these keys. Making them required
    # would break parsing of the existing corpus and fail
    # check_provenance.py's id-recomputation rule. None means "not recorded",
    # never "not applicable".
    #
    # These are env fields, not ServerCell fields, so they are outside
    # _ID_SCHEMA_V1 and cell_id is unaffected. That is deliberate: a cell is
    # a declared configuration, and which physical card served it is a
    # property of the measurement rather than of the configuration.
    gpu_uuid: str | None = None          # per-device, e.g. "GPU-1a2b3c4d-..."
    gpu_serial: str | None = None        # often unavailable on cloud cards
    sm_clock_max_mhz: float | None = None
    power_limit_w: float | None = None


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
        gpu_uuid=gpu_info.get("uuid"),
        gpu_serial=gpu_info.get("serial"),
        sm_clock_max_mhz=gpu_info.get("sm_clock_max_mhz"),
        power_limit_w=gpu_info.get("power_limit_w"),
    )


def _na(value: str) -> str | None:
    """Normalise nvidia-smi's unavailable markers to None.

    nvidia-smi prints "[N/A]" or "N/A" for fields a card does not expose --
    serial is commonly unavailable on cloud instances. Storing the literal
    string would make a missing reading look like a recorded one.
    """
    v = value.strip()
    if not v or v.upper().strip("[]") in ("N/A", "NOT SUPPORTED", "UNKNOWN"):
        return None
    return v


def _float_or_none(value: str) -> float | None:
    v = _na(value)
    if v is None:
        return None
    try:
        return float(v.split()[0])
    except (ValueError, IndexError):
        return None


def _query_nvidia_smi() -> dict:
    """Query nvidia-smi for GPU name, VRAM, driver, and CUDA version.

    Tries the structured --query-gpu first; if that fails, tries plain
    nvidia-smi; if that also fails (Colab exit 12), returns defaults
    that capture() fills from torch.cuda.
    """
    # Attempt 1: structured query including physical-device identity.
    #
    # Tried before the three-field query below because uuid is what separates
    # boot effects from host effects (see EnvInfo). If this form fails on an
    # older nvidia-smi that does not know one of these keys, the next attempt
    # asks only for the original three, so a new field can never cost us the
    # fields we already had.
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,uuid,serial,"
            "clocks.max.sm,power.limit",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            return {
                "name": parts[0],
                "vram_mb": float(parts[1]),
                "driver": parts[2],
                "cuda": _get_cuda_version(),
                "uuid": _na(parts[3]),
                "serial": _na(parts[4]),
                "sm_clock_max_mhz": _float_or_none(parts[5]),
                "power_limit_w": _float_or_none(parts[6]),
            }
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, IndexError):
        pass

    # Attempt 1b: the original three-field query, unchanged.
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


# ---------------------------------------------------------------------------
# Achieved clocks, power and throttle reasons DURING a measurement.
#
# EnvInfo records configured limits: clocks.max.sm and power.limit. Those are
# what the card is allowed to do, not what it did. F035 proposes that the
# CUDA-graph deficit on the L4 is power throttling -- a 72 W part at 2040 MHz
# against a 150 W A10 at 1695 MHz, same 22.5 GiB -- and that hypothesis is
# precisely the one limits cannot test.
#
# nvidia-smi reports throttle REASONS, so this does not have to infer
# throttling from a clock curve: clocks_throttle_reasons.sw_power_cap is the
# driver saying "I am clamping this because of the power cap". That is a
# direct answer where a clock drop would be circumstantial.
# ---------------------------------------------------------------------------

_SAMPLE_FIELDS = (
    "clocks.sm", "power.draw", "temperature.gpu",
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
)


class GpuSampler:
    """Poll the GPU in a background thread for the duration of a `with` block.

    Sampling is deliberately cheap and deliberately recorded: an instrument
    that perturbs the thing it measures has to be able to say by how much, so
    `interval_s` and `n_samples` go into the record alongside the readings.

    Every failure mode degrades to "no samples" rather than to a wrong number.
    A card that does not expose throttle reasons yields None for them, which
    is distinguishable from "not throttling" -- the distinction F018 and F031
    were both about.
    """

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self._rows: list[dict] = []
        self._stop = None
        self._thread = None

    def __enter__(self):
        import threading
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return False

    def _loop(self) -> None:
        while self._stop is not None and not self._stop.is_set():
            row = _sample_gpu()
            if row:
                self._rows.append(row)
            if self._stop.wait(self.interval_s):
                break

    def summary(self) -> dict:
        """Aggregate. Returns `{}` when nothing was sampled, never zeros."""
        if not self._rows:
            return {"gpu_samples": 0}

        def _stat(key, fn):
            vals = [r[key] for r in self._rows if r.get(key) is not None]
            return round(fn(vals), 1) if vals else None

        throttled = [r for r in self._rows
                     if r.get("throttle_sw_power_cap") is True]
        thermal = [r for r in self._rows
                   if r.get("throttle_sw_thermal") is True]
        hw = [r for r in self._rows if r.get("throttle_hw_slowdown") is True]
        return {
            "gpu_samples": len(self._rows),
            "gpu_sample_interval_s": self.interval_s,
            "sm_clock_mean_mhz": _stat("sm_clock_mhz", lambda v: sum(v) / len(v)),
            "sm_clock_min_mhz": _stat("sm_clock_mhz", min),
            "sm_clock_max_mhz": _stat("sm_clock_mhz", max),
            "power_draw_mean_w": _stat("power_w", lambda v: sum(v) / len(v)),
            "power_draw_max_w": _stat("power_w", max),
            "temperature_max_c": _stat("temp_c", max),
            # Fractions, not booleans: "throttled for 3% of the run" and
            # "throttled for 90%" are different findings.
            "frac_power_capped": round(len(throttled) / len(self._rows), 3),
            "frac_thermal_throttled": round(len(thermal) / len(self._rows), 3),
            "frac_hw_slowdown": round(len(hw) / len(self._rows), 3),
        }


def _sample_gpu() -> dict | None:
    """One nvidia-smi reading. None if unavailable, never a fabricated row."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(_SAMPLE_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip().split("\n")[0]
    except Exception:
        return None
    p = [x.strip() for x in out.split(",")]
    if len(p) < len(_SAMPLE_FIELDS):
        return None

    def _b(v: str):
        v = v.strip().lower()
        if v in ("active", "1", "true"):
            return True
        if v in ("not active", "0", "false"):
            return False
        return None          # unsupported -- NOT the same as "not throttling"

    return {
        "sm_clock_mhz": _float_or_none(p[0]),
        "power_w": _float_or_none(p[1]),
        "temp_c": _float_or_none(p[2]),
        "throttle_sw_power_cap": _b(p[3]),
        "throttle_hw_slowdown": _b(p[4]),
        "throttle_sw_thermal": _b(p[5]),
    }
