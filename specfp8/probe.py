"""Phase 1 — Compatibility matrix probe.

For every ServerCell in the sweep YAML: attempt launch, run correctness
tests, scrape one short τ measurement. Output is the compatibility matrix
that decides headline A vs B.

Usage:
    python -m specfp8.probe --sweep sweeps/compat.yaml --results results/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from specfp8 import cells as cells_mod
from specfp8.cells import ServerCell, cell_id
from specfp8.client import probe_run
from specfp8.correctness import (
    get_all_prompts,
    get_gsm8k_answers,
    run_correctness,
)
from specfp8.env import EnvInfo, capture as capture_env
from specfp8.launchers.base import (
    HealthResult,
    ServerHandle,
    find_free_port,
)
from specfp8.launchers.sglang import SglangLauncher
from specfp8.launchers.vllm import VllmLauncher
from specfp8.metrics.base import scrape_spec_stats, delta as metrics_delta
from specfp8.metrics.sglang import SglangMetricsScraper
from specfp8.metrics.vllm import VllmMetricsScraper
from specfp8.store import append, append_budget_log, completed_ids


SAMPLING_PARAMS = {
    "temperature": 0,
    "max_tokens": 256,
}


def get_launcher(engine: str):
    if engine == "vllm":
        return VllmLauncher()
    elif engine == "sglang":
        return SglangLauncher()
    else:
        raise ValueError(f"Unknown engine: {engine}")


def get_metrics_scraper(engine: str):
    if engine == "vllm":
        return VllmMetricsScraper()
    elif engine == "sglang":
        return SglangMetricsScraper()
    else:
        raise ValueError(f"Unknown engine: {engine}")


def probe_one_cell(
    cell: ServerCell,
    cid: str,
    env: EnvInfo,
    results_dir: Path,
    log_dir: Path,
) -> dict:
    """Probe a single ServerCell: boot, correctness test, τ measurement."""
    launcher = get_launcher(cell.engine)
    scraper = get_metrics_scraper(cell.engine)
    port = find_free_port()
    prompts = get_all_prompts()

    print(f"\n{'='*60}")
    print(f"Probing: {cid}")
    print(f"  engine={cell.engine} mechanism={cell.mechanism}")
    print(f"  weight={cell.weight_precision} kv={cell.kv_cache_dtype}")
    print(f"{'='*60}")

    cell_start = time.monotonic()

    # Attempt launch
    try:
        handle = launcher.start(cell, cid, port, log_dir)
    except Exception as e:
        return _make_record(
            cid, cell, env,
            status="launch_failed",
            error=str(e),
            wallclock_s=time.monotonic() - cell_start,
        )

    try:
        # Wait for healthy
        health = launcher.wait_healthy(handle, timeout_s=300)

        if not health.ok:
            return _make_record(
                cid, cell, env,
                status=health.status,
                error=health.error_verbatim,
                log_path=health.log_path,
                wallclock_s=time.monotonic() - cell_start,
            )

        print(f"  Server healthy (boot: {handle.boot_time_s:.1f}s)")

        # KV capacity
        kv_cap = launcher.kv_capacity(handle)
        print(f"  KV capacity: {kv_cap}")

        # Scrape metrics BEFORE
        stats_before = scrape_spec_stats(handle.base_url, scraper)

        # Run correctness probe (concurrency 1)
        print(f"  Running correctness probe (32 prompts, warmup=3)...")
        probe_result = asyncio.run(
            probe_run(
                handle.base_url,
                cell.model,
                prompts,
                SAMPLING_PARAMS,
                n_warmup=3,
                timeout_s=120.0,
            )
        )

        # Check for request failures
        failed_count = sum(1 for r in probe_result.requests if not r.ok)
        if failed_count > 0:
            print(f"  WARNING: {failed_count}/{len(probe_result.requests)} requests failed")
            errors = [r.error for r in probe_result.requests if not r.ok and r.error]
            if errors:
                print(f"  First error: {errors[0][:200]}")

        # Scrape metrics AFTER
        stats_after = scrape_spec_stats(handle.base_url, scraper)

        # Compute τ
        tau = None
        spec_stats = None
        if stats_before is not None and stats_after is not None:
            spec_stats = metrics_delta(stats_before, stats_after)
            tau = spec_stats.tau
            print(f"  τ = {tau:.2f}" if tau else "  τ = null (no spec stats)")
        else:
            print(f"  τ = null (counters not available)")

        # Run correctness assessment
        # For probe, we compare against ourselves (spec vs spec) as a baseline.
        # The full Test 1 (spec vs non-spec at matched precision) requires booting
        # a non-spec server with the same precision — we handle this by collecting
        # non-spec baselines as separate cells (mechanism=none).
        #
        # For now, run Test 3 (garbage backstop) on this cell's outputs.
        gsm_answers = get_gsm8k_answers()
        from specfp8.correctness import test3_garbage, compute_verdict

        task_accuracy, degenerate_count, empty_count = test3_garbage(
            probe_result.outputs, gsm_answers
        )

        # Preliminary verdict from Test 3 only
        # Full Test 1 verdict requires the matched non-spec baseline
        verdict = compute_verdict(
            exact_match_rate=1.0,  # placeholder until Test 1 runs
            task_accuracy=task_accuracy,
            degenerate_count=degenerate_count,
            empty_count=empty_count,
        )

        print(f"  Task accuracy: {task_accuracy:.2%}" if task_accuracy is not None else "  Task accuracy: N/A")
        print(f"  Degenerate: {degenerate_count}, Empty: {empty_count}")
        print(f"  Preliminary verdict: {verdict}")

        wallclock_s = time.monotonic() - cell_start

        return _make_record(
            cid, cell, env,
            status="ok" if verdict == "correct" else verdict,
            wallclock_s=wallclock_s,
            boot_time_s=handle.boot_time_s,
            kv_capacity=kv_cap,
            tau=tau,
            spec_stats=spec_stats,
            task_accuracy=task_accuracy,
            degenerate_count=degenerate_count,
            empty_count=empty_count,
            verdict=verdict,
            log_path=handle.log_path,
            outputs=probe_result.outputs,
        )

    finally:
        launcher.stop(handle)


def _make_record(
    cid: str,
    cell: ServerCell,
    env: EnvInfo,
    status: str,
    wallclock_s: float,
    error: str | None = None,
    log_path: str = "",
    boot_time_s: float = 0.0,
    kv_capacity: int | None = None,
    tau: float | None = None,
    spec_stats=None,
    task_accuracy: float | None = None,
    degenerate_count: int = 0,
    empty_count: int = 0,
    verdict: str = "",
    outputs: list[str] | None = None,
) -> dict:
    """Build a result record for cells.jsonl."""
    record = {
        "cell_id": cid,
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": "probe",
        "env": env.model_dump(),
        "config": cell.model_dump(),
        "outcome": {
            "status": status,
            "error_verbatim": error,
            "log_path": log_path,
        },
        "boot_time_s": round(boot_time_s, 1),
        "wallclock_s": round(wallclock_s, 1),
    }

    if kv_capacity is not None:
        record["capacity"] = {"kv_tokens_max": kv_capacity}

    if tau is not None:
        record["spec"] = {"tau": round(tau, 3)}
        if spec_stats is not None:
            record["spec"].update({
                "accepted_tokens": spec_stats.accepted_tokens,
                "draft_tokens": spec_stats.draft_tokens,
                "verification_steps": spec_stats.verification_steps,
            })

    record["correctness"] = {
        "task_accuracy": round(task_accuracy, 4) if task_accuracy is not None else None,
        "degenerate_count": degenerate_count,
        "empty_count": empty_count,
        "verdict": verdict,
    }

    return record


def run_probe(sweep_path: str, results_dir: str) -> None:
    """Main probe entry point."""
    results_path = Path(results_dir)
    log_dir = Path("logs")

    print("Capturing environment...")
    env = capture_env()
    print(f"  GPU: {env.gpu}")
    print(f"  Compute capability: {env.compute_cap}")
    print(f"  VRAM: {env.vram_gb} GB")
    print(f"  CUDA: {env.cuda}, Torch: {env.torch}")

    print(f"\nExpanding sweep: {sweep_path}")
    all_cells = cells_mod.expand(sweep_path)
    print(f"  Total cells: {len(all_cells)}")

    done = completed_ids(results_path)
    remaining = [(c, cell_id(c)) for c in all_cells if cell_id(c) not in done]
    print(f"  Already completed: {len(done)}")
    print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("\nAll cells complete. Nothing to do.")
        return

    session_start = time.monotonic()
    cumulative_gpu_s = 0.0

    for i, (cell, cid) in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}]")

        record = probe_one_cell(cell, cid, env, results_path, log_dir)
        append(record, results_path)

        cell_time = record.get("wallclock_s", 0)
        cumulative_gpu_s += cell_time

        status = record["outcome"]["status"]
        print(f"  Status: {status}")
        print(f"  Cell time: {cell_time:.1f}s | Session total: {cumulative_gpu_s:.0f}s ({cumulative_gpu_s/3600:.1f}h)")

    # Budget log
    append_budget_log("probe", cumulative_gpu_s, cumulative_gpu_s, results_path)

    print(f"\n{'='*60}")
    print(f"Probe complete.")
    print(f"  Cells probed: {len(remaining)}")
    print(f"  Total GPU time: {cumulative_gpu_s:.0f}s ({cumulative_gpu_s/3600:.1f}h)")
    print(f"  Results: {results_path / 'cells.jsonl'}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Compatibility matrix probe")
    parser.add_argument(
        "--sweep", default="sweeps/compat.yaml",
        help="Path to sweep YAML (default: sweeps/compat.yaml)",
    )
    parser.add_argument(
        "--results", default="results",
        help="Results directory (default: results/)",
    )
    args = parser.parse_args()
    run_probe(args.sweep, args.results)


if __name__ == "__main__":
    main()
