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
    """SIGTERM, wait 10s, SIGKILL if still alive."""
    import signal

    proc = handle.get_process()
    if proc is None or proc.poll() is not None:
        return

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _tail_log(log_path: str, n_lines: int) -> str:
    """Read the last n_lines from a log file."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except (FileNotFoundError, PermissionError):
        return ""
