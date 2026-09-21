"""Regenerate results/runs.json from cells.jsonl.

Run manifests were first built by hand, which meant they went stale the moment
another sweep landed — an audit found them covering 18 of 22 cells. Deriving
them makes that impossible.

Runs are separated by gaps in time. That is an inference, not a record: the
harness does not yet stamp a run id on each cell, so every manifest here is
marked `reconstructed: true`. Stamping it live is the proper fix; until then
the inference is at least labelled as one.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
GAP_S = 1200  # a pause longer than this starts a new run


def _t(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _sweep_of(records: list[dict]) -> str:
    """Infer which sweep produced a group of records."""
    model = records[-1]["config"]["model"]
    backends = {r["config"].get("attn_backend", "auto") for r in records}
    mechs = {r["config"]["mechanism"] for r in records}
    has_quality = any(r.get("quality") for r in records)
    if "0.6B" in model:
        return "sweeps/test_mini.yaml"
    if has_quality:
        return "sweeps/quality.yaml"
    if len(backends) > 1:
        return "sweeps/backend.yaml"
    if mechs <= {"none", "dflash"}:
        return "sweeps/h1.yaml"
    return "sweeps/compat.yaml"


def build() -> list[dict]:
    records = [json.loads(l) for l in open(RESULTS / "cells.jsonl") if l.strip()]
    records.sort(key=lambda r: r["ts"])

    groups: list[list[dict]] = []
    current = [records[0]]
    for a, b in zip(records, records[1:]):
        if (_t(b["ts"]) - _t(a["ts"])).total_seconds() > GAP_S:
            groups.append(current)
            current = []
        current.append(b)
    groups.append(current)

    out = []
    for g in groups:
        env = g[-1].get("env", {})
        sweep = _sweep_of(g)
        out.append({
            "run_id": f"{_t(g[0]['ts']).strftime('%Y-%m-%dT%H-%MZ')}_"
                      f"{Path(sweep).stem}",
            "sweep": sweep,
            "engine": g[-1]["config"]["engine"],
            "engine_version": next(
                (r.get("engine_version") for r in reversed(g)
                 if r.get("engine_version")), None),
            "engine_ref_declared": g[-1]["config"].get("engine_ref"),
            "model": g[-1]["config"]["model"],
            "gpu": env.get("gpu"), "compute_cap": env.get("compute_cap"),
            "vram_gb": env.get("vram_gb"), "cuda": env.get("cuda"),
            "torch": env.get("torch"), "driver": env.get("driver"),
            "started": g[0]["ts"], "ended": g[-1]["ts"],
            "n_records": len(g),
            "n_unique_cells": len({r["cell_id"] for r in g}),
            "attn_backends": sorted(
                {r["config"].get("attn_backend", "auto") for r in g}),
            "has_quality": any(r.get("quality") for r in g),
            "status_counts": {
                s: sum(1 for r in g if r["outcome"]["status"] == s)
                for s in sorted({r["outcome"]["status"] for r in g})},
            "gpu_seconds": round(sum(r.get("wallclock_s", 0) for r in g), 1),
            "cell_ids": sorted({r["cell_id"] for r in g}),
            "reconstructed": True,
            "reconstruction_note": (
                f"Run boundaries inferred from gaps longer than {GAP_S}s. "
                "The harness does not stamp a run id on each cell; until it "
                "does, these groupings are an inference."),
        })
    return out


def main() -> None:
    runs = build()
    (RESULTS / "runs.json").write_text(
        json.dumps({"runs": runs}, indent=2))
    covered = set().union(*(set(r["cell_ids"]) for r in runs))
    print(f"  wrote results/runs.json: {len(runs)} runs, "
          f"{len(covered)} cells covered")
    for r in runs:
        print(f"    {r['run_id']:36} {Path(r['sweep']).name:18} "
              f"cells={r['n_unique_cells']:2} gpu_s={r['gpu_seconds']:.0f}"
              f"{'  +quality' if r['has_quality'] else ''}")


if __name__ == "__main__":
    main()
