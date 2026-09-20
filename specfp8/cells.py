"""Cell model, stable ID, and sweep expansion.

A ServerCell is one server boot configuration. A RunCell is one measurement
against a booted server. cell_id is a stable hash of the canonical JSON
representation — sorted keys, no whitespace — so results from different
machines merge cleanly.
"""

from __future__ import annotations

import hashlib
import json
import itertools
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class ServerCell(BaseModel):
    """One server boot configuration."""

    engine: Literal["sglang", "vllm"]
    engine_ref: str                         # version tag or commit
    model: str
    model_revision: str
    mechanism: Literal["none", "ngram", "eagle3", "dflash", "mtp"]
    draft_model: str | None = None
    draft_revision: str | None = None
    weight_precision: Literal["bf16", "fp8"]
    kv_cache_dtype: Literal["auto", "fp8_e4m3", "fp8_e5m2"]
    attn_backend: str = "auto"
    spec_tokens: int | None = None
    block_size: int | None = None
    guided_decoding: bool = False


class RunCell(BaseModel):
    """One measurement against a booted server."""

    server: ServerCell
    workload: str
    concurrency: int
    seed: int
    repeat: int


def cell_id(cell: ServerCell | RunCell) -> str:
    """Stable, machine-independent ID from canonical JSON.

    Canonical = keys sorted recursively, no whitespace, no trailing commas.
    """
    canonical = json.dumps(
        cell.model_dump(), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def expand(yaml_path: str | Path) -> list[ServerCell]:
    """Expand a sweep YAML into a deduplicated list of ServerCells.

    Steps:
    1. Cartesian product of all `axes` lists.
    2. Merge `defaults`, then `per_mechanism` overrides for matching mechanism.
    3. Drop any cell matching an `exclude` entry.
    4. Deduplicate by cell_id.
    """
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)

    # A section that is present but comment-only parses as None, not as an
    # empty collection — normalise so callers always get something iterable.
    axes = spec["axes"]
    defaults = spec.get("defaults") or {}
    per_mechanism = spec.get("per_mechanism") or {}
    excludes = spec.get("exclude") or []
    top_level = {
        k: spec[k]
        for k in ("model", "model_revision", "spec_tokens")
        if k in spec
    }

    # Cartesian product of axes
    axis_names = list(axes.keys())
    axis_values = [axes[k] if isinstance(axes[k], list) else [axes[k]] for k in axis_names]

    cells: list[ServerCell] = []
    seen_ids: set[str] = set()

    for combo in itertools.product(*axis_values):
        # Start with top-level fields
        fields: dict = {**top_level}

        # Apply defaults
        fields.update(defaults)

        # Apply axis values
        for name, val in zip(axis_names, combo):
            fields[name] = val

        # Apply per-mechanism overrides
        mechanism = fields.get("mechanism", "none")
        if mechanism in per_mechanism:
            fields.update(per_mechanism[mechanism])

        # Check excludes
        if _matches_any_exclude(fields, excludes):
            continue

        try:
            cell = ServerCell(**fields)
        except Exception:
            # Skip invalid combinations (e.g., missing required fields)
            continue

        cid = cell_id(cell)
        if cid not in seen_ids:
            seen_ids.add(cid)
            cells.append(cell)

    return cells


def expand_run_cells(yaml_path: str | Path) -> list[RunCell]:
    """Expand a performance sweep YAML into RunCells.

    The YAML must have `run_axes` with workload, concurrency, seed, repeat lists.
    Each ServerCell is paired with every combination of run axes.
    """
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)

    server_cells = expand(yaml_path)
    run_axes = spec.get("run_axes") or {}

    workloads = run_axes.get("workload", ["gsm8k"])
    concurrencies = run_axes.get("concurrency", [1])
    seeds = run_axes.get("seed", [42])
    repeats = run_axes.get("repeat", [0])

    run_cells: list[RunCell] = []
    for sc in server_cells:
        for wl, conc, seed, rep in itertools.product(
            workloads, concurrencies, seeds, repeats
        ):
            run_cells.append(
                RunCell(
                    server=sc,
                    workload=wl,
                    concurrency=conc,
                    seed=seed,
                    repeat=rep,
                )
            )

    return run_cells


def group_by_server(run_cells: list[RunCell]) -> dict[str, list[RunCell]]:
    """Group RunCells by ServerCell ID for boot-once-run-many."""
    groups: dict[str, list[RunCell]] = {}
    for rc in run_cells:
        sid = cell_id(rc.server)
        groups.setdefault(sid, []).append(rc)
    return groups


def _matches_any_exclude(fields: dict, excludes: list[dict]) -> bool:
    """Check if a field dict matches any exclude pattern."""
    for exc in excludes:
        if all(fields.get(k) == v for k, v in exc.items()):
            return True
    return False
