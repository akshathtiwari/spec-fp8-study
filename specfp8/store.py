"""Atomic JSONL append + resume.

Results are written per-cell with fsync + os.replace so an interrupted sweep
loses at most the in-flight cell, never corrupts completed records. Resume
works by scanning completed IDs at startup and subtracting them from the
full cell list.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone


def completed_ids(results_dir: str | Path) -> set[str]:
    """Scan cells.jsonl once and return the set of completed cell_ids."""
    return set(completed_cells(results_dir))


def completed_cells(results_dir: str | Path) -> dict[str, dict]:
    """Map cell_id -> {"argv_fingerprint", "status"} for the latest record.

    A cell_id hashes the ServerCell config only, so it cannot distinguish
    "this configuration is genuinely unsupported" from "the harness built a
    bad command line". The fingerprint closes part of that gap: when the
    launcher starts emitting a different command for the same config, the
    old record is stale and the cell is re-run.

    It does not close all of it. An environment fault — a missing CUDA
    toolkit, an OOM, a bad driver — fails the cell without changing the
    command, so such a record is indistinguishable from a genuine
    incompatibility by fingerprint alone. `status` is surfaced here so the
    caller can choose to retry failures explicitly.
    """
    cells_path = Path(results_dir) / "cells.jsonl"
    seen: dict[str, dict] = {}

    if not cells_path.exists():
        return seen

    with open(cells_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines (e.g., from a partial write that
                # somehow survived — shouldn't happen with atomic rename,
                # but defensive)
                continue
            if "cell_id" in record:
                # Later records win: a re-run supersedes the stale result.
                seen[record["cell_id"]] = {
                    "argv_fingerprint": record.get("argv_fingerprint"),
                    "sampling_params": record.get("sampling_params"),
                    "prompt_set_fingerprint": record.get("prompt_set_fingerprint"),
                    "status": record.get("outcome", {}).get("status"),
                }

    return seen


#: argv flags whose values vary between runs without changing what is measured.
_VOLATILE_FLAGS = {"--port", "--download-dir"}


def argv_fingerprint(argv: list[str]) -> str:
    """Stable hash of a launch command, ignoring run-to-run noise.

    Port and download directory change on every invocation and across
    machines, so they are excluded — two runs of the same configuration on
    different hosts must produce the same fingerprint.
    """
    kept: list[str] = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in _VOLATILE_FLAGS:
            skip_next = True
            continue
        kept.append(tok)

    joined = "\x00".join(kept)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def append(record: dict, results_dir: str | Path) -> None:
    """Atomically append one result record to cells.jsonl.

    Write to a temp file, fsync, then os.replace to achieve atomic append.
    This means a kill mid-write leaves the last good record intact.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cells_path = results_dir / "cells.jsonl"

    line = json.dumps(record, separators=(",", ":")) + "\n"

    # Write to temp file in the same directory (same filesystem for rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=results_dir, prefix=".cell_", suffix=".tmp"
    )
    try:
        os.write(fd, line.encode())
        os.fsync(fd)
        os.close(fd)
        fd = -1  # mark as closed

        # Append by reading existing + new line, then atomic replace
        existing = ""
        if cells_path.exists():
            existing = cells_path.read_text()

        # Write combined to a new temp
        fd2, tmp2 = tempfile.mkstemp(
            dir=results_dir, prefix=".cells_", suffix=".tmp"
        )
        os.write(fd2, (existing + line).encode())
        os.fsync(fd2)
        os.close(fd2)

        os.replace(tmp2, cells_path)
    finally:
        if fd >= 0:
            os.close(fd)
        # Clean up first temp
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def append_requests(
    cell_id: str, records: list[dict], results_dir: str | Path
) -> str:
    """Write per-request records for a cell. Returns the path written."""
    results_dir = Path(results_dir)
    req_dir = results_dir / "requests"
    req_dir.mkdir(parents=True, exist_ok=True)

    out_path = req_dir / f"{cell_id}.jsonl"

    fd, tmp_path = tempfile.mkstemp(dir=req_dir, prefix=f".{cell_id}_", suffix=".tmp")
    try:
        content = "".join(
            json.dumps(r, separators=(",", ":")) + "\n" for r in records
        )
        os.write(fd, content.encode())
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp_path, out_path)
    finally:
        if fd >= 0:
            os.close(fd)

    return str(out_path)


def persist_log(
    cell_id: str,
    log_path: str,
    results_dir: str | Path,
    max_bytes: int = 4_000_000,
) -> str | None:
    """Copy a server's launch log into the results directory.

    Launchers write logs beside the working directory, which on rented
    hardware vanishes when the container exits. Every record stores a
    `log_path`, so without this a result points at evidence that no longer
    exists — including the attention-backend selection lines and the
    verbatim error behind each failed cell, which are precisely what a
    reader needs to check a compatibility claim.

    Oversized logs are truncated in the middle rather than at one end: the
    head carries configuration and backend selection, the tail carries the
    failure, and the interesting parts are at both ends.
    """
    src = Path(log_path)
    if not src.is_file():
        return None

    out_dir = Path(results_dir) / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{cell_id}.log"

    try:
        data = src.read_bytes()
    except OSError:
        return None

    if len(data) > max_bytes:
        head = max_bytes * 3 // 4
        tail = max_bytes - head
        omitted = len(data) - max_bytes
        data = (
            data[:head]
            + f"\n\n... [{omitted} bytes omitted by persist_log] ...\n\n".encode()
            + data[-tail:]
        )

    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=f".{cell_id}_", suffix=".tmp")
    try:
        os.write(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp, dest)
    finally:
        if fd >= 0:
            os.close(fd)
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass

    return str(dest)


def append_budget_log(
    phase: str,
    session_gpu_s: float,
    cumulative_gpu_s: float,
    results_dir: str | Path,
) -> None:
    """Append one line to budget.log with timing info."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "budget.log"

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "session_gpu_s": round(session_gpu_s, 1),
        "cumulative_gpu_s": round(cumulative_gpu_s, 1),
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")
