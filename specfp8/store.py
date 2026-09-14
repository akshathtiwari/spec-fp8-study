"""Atomic JSONL append + resume.

Results are written per-cell with fsync + os.replace so an interrupted sweep
loses at most the in-flight cell, never corrupts completed records. Resume
works by scanning completed IDs at startup and subtracting them from the
full cell list.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone


def completed_ids(results_dir: str | Path) -> set[str]:
    """Scan cells.jsonl once and return the set of completed cell_ids."""
    cells_path = Path(results_dir) / "cells.jsonl"
    ids: set[str] = set()

    if not cells_path.exists():
        return ids

    with open(cells_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "cell_id" in record:
                    ids.add(record["cell_id"])
            except json.JSONDecodeError:
                # Skip malformed lines (e.g., from a partial write that
                # somehow survived — shouldn't happen with atomic rename,
                # but defensive)
                continue

    return ids


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
