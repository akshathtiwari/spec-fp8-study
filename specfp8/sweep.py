"""Phase 2 — performance sweep (R3-R10).

For every ServerCell that Phase 1 marked usable: boot once, then run every
workload x concurrency x repeat against that server, writing one summary
record plus one per-request file per RunCell.

Two design points carry most of the value:

**Boot once, run many.** Server boot is 3-5 minutes and a measurement is
2-3, so booting per RunCell would spend more GPU on startup than on
measuring. RunCells are grouped by ServerCell and the server is reused
across the group.

**Store raw, derive later.** Every request's timings are written out, not
just aggregates. Goodput at any SLO is then computed offline from the same
data, so sweeping the SLO threshold (R7) costs no GPU at all.

    python -m specfp8.sweep --sweep sweeps/perf.yaml --results results/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from specfp8 import cells as cells_mod
from specfp8.cells import RunCell, ServerCell, cell_id, group_by_server
from specfp8.client import load_run
from specfp8.env import EnvInfo, capture as capture_env
from specfp8.launchers.base import find_free_port, wait_for_gpu_free
from specfp8.metrics.base import scrape_spec_stats, delta as metrics_delta
from specfp8.probe import (
    MIN_FREE_GPU_MIB,
    HEALTH_TIMEOUT_S,
    get_launcher,
    get_metrics_scraper,
)
from specfp8.store import (
    append,
    append_budget_log,
    argv_fingerprint,
    completed_cells,
    persist_log,
)
from specfp8.workloads import get_workload


def _run_one_measurement(
    handle, server: ServerCell, run: RunCell, rid: str,
    env: EnvInfo, results_dir: Path, scraper,
) -> dict:
    """One workload x concurrency x repeat against an already-booted server."""
    workload = get_workload(run.workload)
    n_requests = min(run.concurrency * 8, workload.size)
    prompts = workload.prompts(n_requests, seed=run.seed)
    sampling = workload.sampling()

    stats_before = scrape_spec_stats(handle.base_url, scraper)
    started = time.monotonic()
    results = asyncio.run(
        load_run(handle.base_url, server.model, prompts, sampling,
                 concurrency=run.concurrency, n_warmup=min(4, n_requests),
                 timeout_s=600.0)
    )
    wall_s = time.monotonic() - started
    stats_after = scrape_spec_stats(handle.base_url, scraper)

    ok = [r for r in results if r.ok]
    correct = sum(
        1 for i, r in enumerate(results)
        if r.ok and workload.validate(i, r.output_text) is True
    )
    scored = sum(
        1 for i, r in enumerate(results)
        if r.ok and workload.validate(i, r.output_text) is not None
    )

    # Per-request rows are the point: goodput at any SLO is derived from
    # these offline, so adding an SLO later costs nothing.
    rows = [
        {
            "index": i,
            "ok": r.ok,
            "ttft_ms": round(r.ttft_ms, 3),
            "e2e_ms": round(r.e2e_ms, 3),
            "output_tokens": r.output_tokens,
            "itl_ms": [round(x, 3) for x in r.itl_ms],
            "correct": workload.validate(i, r.output_text) if r.ok else None,
            "error": r.error,
        }
        for i, r in enumerate(results)
    ]
    req_dir = Path(results_dir) / "runs"
    req_dir.mkdir(parents=True, exist_ok=True)
    (req_dir / f"{rid}.jsonl").write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    )

    spec = None
    if stats_before is not None and stats_after is not None:
        d = metrics_delta(stats_before, stats_after)
        spec = {
            "tau": round(d.tau, 4) if d.tau else None,
            "acceptance_rate": (round(d.acceptance_rate, 4)
                                if d.acceptance_rate is not None else None),
            "accepted_tokens": d.accepted_tokens,
            "draft_tokens": d.draft_tokens,
            "verification_steps": d.verification_steps,
        }

    out_tokens = sum(r.output_tokens for r in ok)
    return {
        "run_id": rid,
        "cell_id": cell_id(server),
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": "sweep",
        "env": env.model_dump(),
        "config": server.model_dump(),
        "load": {
            "workload": run.workload, "concurrency": run.concurrency,
            "seed": run.seed, "repeat": run.repeat,
            "n_requests": n_requests,
        },
        "sampling_params": sampling,
        "outcome": {
            "status": "ok" if len(ok) == len(results) else "partial",
            "requests_ok": len(ok), "requests_total": len(results),
        },
        "throughput": {
            "wall_clock_s": round(wall_s, 3),
            "output_tokens": out_tokens,
            "output_tokens_per_s": round(out_tokens / wall_s, 2) if wall_s else None,
            "requests_per_s": round(len(ok) / wall_s, 4) if wall_s else None,
        },
        "task": {
            "scored": scored, "correct": correct,
            "accuracy": round(correct / scored, 4) if scored else None,
        },
        "spec": spec,
        "requests_path": f"results/runs/{rid}.jsonl",
    }


def run_sweep(sweep_path: str, results_dir: str, force: bool = False) -> None:
    results_path = Path(results_dir)
    log_dir = Path("logs")

    print("Capturing environment...")
    env = capture_env()
    print(f"  GPU: {env.gpu} (SM{env.compute_cap.replace('.', '')}), "
          f"{env.vram_gb} GB, CUDA {env.cuda}, torch {env.torch}")

    run_cells = cells_mod.expand_run_cells(sweep_path)
    groups = group_by_server(run_cells)
    print(f"\n{sweep_path}: {len(run_cells)} RunCells over "
          f"{len(groups)} ServerCells")

    done = set()
    if not force:
        path = results_path / "sweep.jsonl"
        if path.exists():
            with open(path) as f:
                for line in f:
                    if line.strip():
                        done.add(json.loads(line).get("run_id"))
        print(f"  already complete: {len(done)}")

    session_gpu_s = 0.0
    for gi, (sid, runs) in enumerate(sorted(groups.items()), 1):
        server = runs[0].server
        pending = [
            r for r in runs
            if force or _run_id(r) not in done
        ]
        head = (f"[{gi}/{len(groups)}] {server.mechanism}|"
                f"{server.weight_precision}|{server.kv_cache_dtype}|"
                f"{server.attn_backend}")
        if not pending:
            print(f"\n{head} — all {len(runs)} measurements done, skipping boot")
            continue

        print(f"\n{'=' * 64}\n{head}\n  {len(pending)} measurements, one boot"
              f"\n{'=' * 64}")

        free = wait_for_gpu_free(MIN_FREE_GPU_MIB, timeout_s=180.0)
        if free is not None and free < MIN_FREE_GPU_MIB:
            print(f"  GPU only {free:.0f} MiB free — harness fault, skipping "
                  f"group (it will be retried, not recorded as a result)")
            continue

        launcher = get_launcher(server.engine)
        scraper = get_metrics_scraper(server.engine)
        group_start = time.monotonic()
        handle = launcher.start(server, sid, find_free_port(), log_dir)
        try:
            health = launcher.wait_healthy(handle, timeout_s=HEALTH_TIMEOUT_S)
            if not health.ok:
                print(f"  boot failed: {health.status} — skipping group")
                continue
            print(f"  healthy in {handle.boot_time_s:.0f}s | "
                  f"backend {launcher.selected_backend(handle)} | "
                  f"KV {launcher.kv_capacity(handle)}")

            for run in sorted(pending, key=lambda r: (r.workload, r.concurrency,
                                                      r.repeat)):
                rid = _run_id(run)
                print(f"  - {run.workload} c={run.concurrency} "
                      f"rep={run.repeat} ... ", end="", flush=True)
                rec = _run_one_measurement(
                    handle, server, run, rid, env, results_path, scraper)
                _append_sweep(rec, results_path)
                t = rec["throughput"]
                acc = rec["task"]["accuracy"]
                print(f"{t['output_tokens_per_s']} tok/s, "
                      f"{t['requests_per_s']} req/s"
                      + (f", acc {acc:.1%}" if acc is not None else "")
                      + (f", tau {rec['spec']['tau']}" if rec["spec"] else ""))
        finally:
            launcher.stop(handle)
            persist_log(sid, handle.log_path, results_path)
            session_gpu_s += time.monotonic() - group_start

    append_budget_log("sweep", session_gpu_s, session_gpu_s, results_path)
    print(f"\n{'=' * 64}\nSweep complete. GPU time this session: "
          f"{session_gpu_s:.0f}s ({session_gpu_s / 3600:.2f}h)\n{'=' * 64}")


def _run_id(run: RunCell) -> str:
    """Stable id for one measurement: server config plus load settings."""
    return cell_id(run)


def _append_sweep(record: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "sweep.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 — performance sweep")
    ap.add_argument("--sweep", default="sweeps/perf.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--force", action="store_true",
                    help="Re-run measurements even if already recorded.")
    args = ap.parse_args()
    run_sweep(args.sweep, args.results, force=args.force)


if __name__ == "__main__":
    main()
