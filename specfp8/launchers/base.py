"""Launcher protocol and shared types.

The harness never imports vLLM or SGLang. It launches them as subprocesses
and talks HTTP. This file defines the interface both engine launchers implement.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Protocol, runtime_checkable
from pydantic import BaseModel

from specfp8.cells import ServerCell


class ServerHandle(BaseModel):
    """A running engine server."""

    cell_id: str
    process_pid: int
    port: int
    base_url: str
    log_path: str
    boot_time_s: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    # Store the process object outside pydantic (not serializable)
    _process: subprocess.Popen | None = None

    def set_process(self, proc: subprocess.Popen) -> None:
        object.__setattr__(self, "_process", proc)

    def get_process(self) -> subprocess.Popen | None:
        return getattr(self, "_process", None)


class HealthResult(BaseModel):
    """Result of attempting to boot and health-check a server."""

    ok: bool
    status: str  # "ok", "launch_failed", "oom", "timeout", "unsupported_config"
    error_verbatim: str | None = None   # R1: exact error string, never summarised
    log_path: str = ""


@runtime_checkable
class Launcher(Protocol):
    """Interface that engine-specific launchers implement."""

    def argv(self, cell: ServerCell, port: int) -> list[str]:
        """Build the command-line arguments for launching the server."""
        ...

    def start(
        self, cell: ServerCell, cell_id: str, port: int, log_dir: str | Path
    ) -> ServerHandle:
        """Spawn the server process, stream logs to file."""
        ...

    def wait_healthy(
        self, handle: ServerHandle, timeout_s: int = 300
    ) -> HealthResult:
        """Poll health endpoint until ready or failed."""
        ...

    def engine_version(self, handle: ServerHandle) -> str | None:
        """Version of the engine actually serving, read over HTTP.

        Asked of the running server rather than the harness environment: the
        harness never imports the engine, and the sweep's `engine_ref` is a
        declared intent that nothing verifies. This is what actually answered.
        """
        ...

    def selected_backend(self, handle: ServerHandle) -> str | None:
        """Attention backend the engine actually selected, read from its log.

        Requested is not selected. See findings/F018.
        """
        ...

    def kv_capacity(self, handle: ServerHandle) -> int | None:
        """Parse KV-token capacity from startup log. None if unavailable."""
        ...

    def stop(self, handle: ServerHandle) -> None:
        """Stop the server: SIGTERM, wait 10s, SIGKILL if needed."""
        ...


def find_free_port() -> int:
    """Find a free TCP port for the server."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def poll_health(
    url: str, process: subprocess.Popen | None, timeout_s: int, log_path: str
) -> HealthResult:
    """Generic health polling — shared by both launchers.

    Polls the health endpoint every 2s. Simultaneously watches the child process.
    If the process dies, captures the last ~200 stderr lines.
    """
    import httpx

    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        # Check if process died
        if process is not None and process.poll() is not None:
            error_text = _tail_log(log_path, 200)
            status = "oom" if "out of memory" in error_text.lower() else "launch_failed"
            return HealthResult(
                ok=False,
                status=status,
                error_verbatim=error_text,
                log_path=log_path,
            )

        # Try health endpoint
        try:
            resp = httpx.get(url, timeout=5)
            if resp.status_code == 200:
                return HealthResult(ok=True, status="ok", log_path=log_path)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
            pass

        time.sleep(2)

    # Timeout
    error_text = _tail_log(log_path, 200)
    return HealthResult(
        ok=False,
        status="timeout",
        error_verbatim=error_text,
        log_path=log_path,
    )


def stop_process(handle: ServerHandle) -> None:
    """Terminate the server's whole process group: SIGTERM, then SIGKILL.

    Signalling only the direct child is not enough. Both engines fork
    separate worker processes (vLLM 0.29 runs `EngineCore` and `APIServer`
    as distinct pids), and those workers are the ones holding GPU memory.
    Killing the parent alone orphans them: they survive, keep ~20GB
    resident, and every later cell fails to allocate — which then gets
    recorded as if the *configuration* were unsupported.

    Launchers start servers with `start_new_session=True`, so the whole
    tree shares a process group that can be signalled at once.
    """
    import os
    import signal

    proc = handle.get_process()
    if proc is None or proc.poll() is not None:
        return

    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        pgid = None

    def signal_tree(sig) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass

    signal_tree(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        signal_tree(signal.SIGKILL)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    # The parent can exit while workers are still tearing down, so make sure
    # nothing is left in the group before the caller boots the next server.
    if pgid is not None:
        for _ in range(20):
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.5)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def gpu_free_mib() -> float | None:
    """Free GPU memory in MiB, or None if it cannot be determined.

    Uses nvidia-smi rather than torch.cuda.mem_get_info on purpose: the
    torch call would create a CUDA context inside the harness process and
    consume a few hundred MiB of the very memory being measured.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return float(result.stdout.strip().split("\n")[0])
    except Exception:
        return None


def wait_for_gpu_free(min_mib: float, timeout_s: float = 120.0) -> float | None:
    """Wait for at least `min_mib` of GPU memory to be free.

    A killed server does not release its CUDA context instantly, so booting
    the next cell immediately can fail on memory that is about to come back.
    Returns the last observed free memory (None if unknown).
    """
    deadline = time.monotonic() + timeout_s
    free = gpu_free_mib()
    while free is not None and free < min_mib and time.monotonic() < deadline:
        time.sleep(2)
        free = gpu_free_mib()
    return free


def _tail_log(log_path: str, n_lines: int) -> str:
    """Read the last n_lines from a log file."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except (FileNotFoundError, PermissionError):
        return ""
