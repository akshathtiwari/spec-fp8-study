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
    prompt_set_fingerprint,
    get_gsm8k_answers,
    run_correctness,
)
from specfp8.env import EnvInfo, capture as capture_env
from specfp8.launchers.base import (
    HealthResult,
    ServerHandle,
    find_free_port,
    gpu_free_mib,
    wait_for_gpu_free,
)
from specfp8.launchers.sglang import SglangLauncher
from specfp8.launchers.vllm import VllmLauncher
from specfp8.metrics.base import scrape_spec_stats, delta as metrics_delta
from specfp8.metrics.sglang import SglangMetricsScraper
from specfp8.metrics.vllm import VllmMetricsScraper
from specfp8.store import (
    append,
    append_budget_log,
    append_quality,
    append_requests,
    argv_fingerprint,
    completed_cells,
    persist_log,
)


SAMPLING_PARAMS = {
    "temperature": 0,
    "max_tokens": 256,
    # Qwen3 is a reasoning model: left in thinking mode it spends the whole
    # 256-token budget inside <think> and never states an answer, so GSM8K
    # extraction finds nothing and every cell scores 0% and reads "broken".
    # The non-thinking chat template makes the gate deterministic and keeps
    # answers extractable within the budget. Reasoning workloads are a
    # separate question for the performance sweep, not the correctness gate.
    "chat_template_kwargs": {"enable_thinking": False},
}


#: A cell needs roughly the model weights plus KV cache free before it can
#: even start. Set below the smallest workable headroom rather than tuned per
#: model: the point is to catch a leaked server holding the device, not to
#: predict each cell's exact footprint.
MIN_FREE_GPU_MIB = 8000.0

#: Booting a large model from a cold weight cache includes the download, which
#: can dominate. Sized for a 4B target pulled over the network; the probe is
#: still bounded overall by the runner's own timeout.
HEALTH_TIMEOUT_S = 900


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


def _quality_config_fingerprint() -> str:
    """Identity of the quality measurement's settings.

    Only the inputs that change the measured number belong here: the token
    budget (a truncated generation scores as no answer, see F014) and the
    prompt set. Concurrency is deliberately excluded — task accuracy is robust
    to batching (F009), so including it would invalidate good results on a
    harmless throughput tweak.
    """
    from specfp8.quality import QUALITY_MAX_TOKENS, quality_fingerprint
    from specfp8.store import quality_config_fingerprint
    return quality_config_fingerprint(QUALITY_MAX_TOKENS, quality_fingerprint())


def _cell_fingerprint(cell: ServerCell) -> str:
    """Fingerprint of the command this harness would launch for `cell`."""
    launcher = get_launcher(cell.engine)
    return argv_fingerprint(launcher.argv(cell, 0))


def probe_one_cell(
    cell: ServerCell,
    cid: str,
    env: EnvInfo,
    results_dir: Path,
    log_dir: Path,
    floors: dict[str, float] | None = None,
    run_quality_stage: bool = False,
    quality_concurrency: int = 32,
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

    # Pre-flight: the GPU must actually be free before this cell can be a
    # test of anything. If a previous server leaked, the engine aborts with
    # an out-of-memory error that is indistinguishable, in the record, from
    # the configuration being unsupported — which would quietly corrupt the
    # compatibility matrix. Fail as a harness fault instead, so the cell is
    # re-run rather than believed.
    free_mib = wait_for_gpu_free(MIN_FREE_GPU_MIB, timeout_s=180.0)
    if free_mib is not None and free_mib < MIN_FREE_GPU_MIB:
        print(f"  GPU only has {free_mib:.0f} MiB free "
              f"(need {MIN_FREE_GPU_MIB:.0f}) — harness fault, not a verdict")
        return _make_record(
            cid, cell, env,
            status="harness_error",
            error=(
                f"Only {free_mib:.0f} MiB GPU memory free before launch, "
                f"below the {MIN_FREE_GPU_MIB:.0f} MiB required. A previous "
                f"server most likely failed to release the device. This is a "
                f"harness/host fault and says nothing about whether this "
                f"configuration is supported."
            ),
            wallclock_s=time.monotonic() - cell_start,
        )

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
        health = launcher.wait_healthy(handle, timeout_s=HEALTH_TIMEOUT_S)

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

        # The engine version that actually answered. `engine_ref` in the sweep
        # is a declared intent nothing verifies; this is measured.
        engine_version = launcher.engine_version(handle)
        print(f"  Engine version: {engine_version}")

        # Verify the engine honoured the backend request.
        #
        # Three outcomes, kept distinct on purpose. An earlier version
        # collapsed "no backend requested" and "could not determine" into one
        # None and then printed "(as requested)" for both — asserting the flag
        # was honoured in exactly the case where that was unknown. That is the
        # F018 failure wearing a different hat, so the unverified case is now
        # named and is never treated as confirmation.
        selected = launcher.selected_backend(handle)
        requested = cell.attn_backend
        if requested == "auto":
            backend_check = "not_requested"
        elif selected is None:
            backend_check = "unverified"
        elif selected == requested:
            backend_check = "honoured"
        else:
            backend_check = "mismatch"

        if backend_check == "mismatch":
            print(f"  ** BACKEND MISMATCH: requested {requested}, engine "
                  f"selected {selected} — this cell does NOT test "
                  f"{requested} **")
        elif backend_check == "unverified":
            print(f"  ** BACKEND UNVERIFIED: requested {requested}, but no "
                  f"selection line found in the log — this cell does NOT "
                  f"confirm {requested} was used **")
        elif backend_check == "honoured":
            print(f"  Attention backend: {selected} (as requested)")
        else:
            print(f"  Attention backend: {selected or 'unknown'} (auto)")

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

        # Persist every generated output. A verdict derived from these is only
        # auditable if the text behind it is on disk.
        append_requests(
            cid,
            [
                {
                    "index": i,
                    "ok": r.ok,
                    "output_tokens": r.output_tokens,
                    "ttft_ms": round(r.ttft_ms, 3),
                    "e2e_ms": round(r.e2e_ms, 3),
                    "output_text": r.output_text,
                    "error": r.error,
                }
                for i, r in enumerate(probe_result.requests)
            ],
            results_dir,
        )

        # Optional powered quality measurement, on the same booted server so
        # it costs generation time rather than another boot.
        quality = None
        if run_quality_stage:
            from specfp8.quality import run_quality
            print(f"  Running quality measurement "
                  f"({quality_concurrency} concurrent)...")
            q = run_quality(
                handle.base_url, cell.model, SAMPLING_PARAMS,
                concurrency=quality_concurrency, timeout_s=180.0,
            )
            append_quality(cid, q.pop("records"), results_dir)
            # Recorded so a later change to the quality settings invalidates
            # this measurement instead of leaving it alongside newer ones.
            q["config_fingerprint"] = _quality_config_fingerprint()
            quality = q
            acc, hw = q["accuracy"], q["ci95_halfwidth"]
            print(f"  Quality: {acc:.1%} +/- {hw:.1%} "
                  f"(n={q['n']}, unparseable={q['unparseable']}, "
                  f"truncated={q['truncated']}, "
                  f"failures={q['request_failures']})")

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
            **(floors or {}),
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
            engine_version=engine_version,
            quality=quality,
            selected_backend=selected,
            backend_check=backend_check,
        )

    finally:
        # Stop first, then copy: the log is only complete once the server has
        # finished writing it, and this runs on every exit path so failed
        # cells — the ones whose logs matter most — are captured too.
        launcher.stop(handle)
        saved = persist_log(cid, handle.log_path, results_dir)
        if saved:
            print(f"  Log saved: {saved}")


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
    engine_version: str | None = None,
    quality: dict | None = None,
    selected_backend: str | None = None,
    backend_check: str | None = None,
) -> dict:
    """Build a result record for cells.jsonl."""
    argv = get_launcher(cell.engine).argv(cell, 0)
    record = {
        "cell_id": cid,
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": "probe",
        "env": env.model_dump(),
        "config": cell.model_dump(),
        # Verbatim launch command (port/cache-dir normalised) plus its hash:
        # provenance for a third party, and the resume staleness check.
        "argv": argv,
        "argv_fingerprint": argv_fingerprint(argv),
        # Sampling settings change both tau and task accuracy, so they are
        # part of the result, not an implicit constant of the harness.
        "sampling_params": SAMPLING_PARAMS,
        "prompt_set_fingerprint": prompt_set_fingerprint(),
        "engine_version": engine_version,
        # Requested is not selected: both are recorded so no analysis has to
        # trust the flag (findings/F018).
        "selected_backend": selected_backend,
        # honoured | mismatch | unverified | not_requested. "unverified" is
        # not a pass: it means the cell does not establish which backend ran.
        "backend_check": backend_check,
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
            rate = spec_stats.acceptance_rate
            record["spec"].update({
                "accepted_tokens": spec_stats.accepted_tokens,
                "draft_tokens": spec_stats.draft_tokens,
                "verification_steps": spec_stats.verification_steps,
                "acceptance_rate": round(rate, 4) if rate is not None else None,
            })

    if quality is not None:
        record["quality"] = quality
        record["quality_config_fingerprint"] = quality.get("config_fingerprint")

    record["correctness"] = {
        "task_accuracy": round(task_accuracy, 4) if task_accuracy is not None else None,
        "degenerate_count": degenerate_count,
        "empty_count": empty_count,
        "verdict": verdict,
    }

    return record


def run_probe(
    sweep_path: str, results_dir: str, retry_failed: bool = False,
    quality: bool = False, quality_concurrency: int = 32,
    force: bool = False,
) -> None:
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
    floors = cells_mod.load_correctness_floors(sweep_path)
    if floors:
        print(f"  Correctness floors (from sweep): {floors}")
    print(f"  Total cells: {len(all_cells)}")

    done = completed_cells(results_path)
    remaining: list[tuple[ServerCell, str]] = []
    stale = 0
    retried = 0
    for c in all_cells:
        cid = cell_id(c)
        prev = done.get(cid) if not force else None
        if prev is None:
            remaining.append((c, cid))
            continue
        # A recorded result only counts if the harness would produce it the
        # same way today. Otherwise it reflects an older (possibly broken)
        # harness and must not be read as a compatibility verdict. Sampling
        # settings count here alongside the command: they do not change argv
        # but they do change tau and task accuracy.
        if (prev["argv_fingerprint"] != _cell_fingerprint(c)
                or prev["sampling_params"] != SAMPLING_PARAMS
                or prev["prompt_set_fingerprint"] != prompt_set_fingerprint()):
            stale += 1
            remaining.append((c, cid))
            continue
        # A recorded cell that lacks the quality measurement, or carries one
        # taken under different settings, is not usable when quality is being
        # collected — otherwise a single sweep yields a dataset mixing
        # measurements that are not comparable.
        if quality:
            prev_q = prev.get("quality_config_fingerprint")
            if prev_q != _quality_config_fingerprint():
                stale += 1
                remaining.append((c, cid))
                continue

        # A harness fault is not a measurement of anything, so it is always
        # re-run rather than waiting for --retry-failed.
        if prev["status"] == "harness_error":
            retried += 1
            remaining.append((c, cid))
            continue
        # A failure recorded under a broken environment looks identical to a
        # genuine incompatibility, so retrying is opt-in.
        if retry_failed and prev["status"] != "ok":
            retried += 1
            remaining.append((c, cid))
            continue

    print(f"  Already completed: {len(done) - stale - retried}")
    if stale:
        print(f"  Stale (launch command changed, re-running): {stale}")
    if retried:
        print(f"  Previously failed (--retry-failed, re-running): {retried}")
    print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("\nAll cells complete. Nothing to do.")
        return

    session_start = time.monotonic()
    cumulative_gpu_s = 0.0

    for i, (cell, cid) in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}]")

        record = probe_one_cell(cell, cid, env, results_path, log_dir, floors,
                                run_quality_stage=quality,
                                quality_concurrency=quality_concurrency)
        append(record, results_path)

        cell_time = record.get("wallclock_s", 0)
        cumulative_gpu_s += cell_time

        status = record["outcome"]["status"]
        print(f"  Status: {status}")
        print(f"  Cell time: {cell_time:.1f}s | Session total: {cumulative_gpu_s:.0f}s ({cumulative_gpu_s/3600:.1f}h)")

    # Budget log
    append_budget_log("probe", cumulative_gpu_s, results_path)

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
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Re-run cells whose last recorded status was not 'ok'. Use after "
             "fixing an environment fault, which fails cells without changing "
             "the launch command and so is invisible to the staleness check.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run every cell regardless of staleness. cells.jsonl is "
             "append-only, so repeated forced runs leave one record per run "
             "per cell — which is how measurement reproducibility is tested.",
    )
    parser.add_argument(
        "--quality", action="store_true",
        help="Also run the powered task-accuracy measurement (256 GSM8K "
             "problems) against each booted server. Costs a few minutes per "
             "cell; the 32-prompt gate alone cannot resolve quality (F008).",
    )
    parser.add_argument(
        "--quality-concurrency", type=int, default=32,
        help="Concurrency for the quality measurement (default 16). Safe to "
             "raise: task accuracy, unlike exact match, does not require "
             "concurrency 1.",
    )
    args = parser.parse_args()
    run_probe(args.sweep, args.results, retry_failed=args.retry_failed,
              quality=args.quality,
              quality_concurrency=args.quality_concurrency,
              force=args.force)


def _setup_cuda_ld_path():
    """Ensure pip-installed CUDA libs are on LD_LIBRARY_PATH.

    On Colab, nvidia pip packages install .so files under
    site-packages/nvidia/*/lib/ which isn't on the default search path.
    This must run before any CUDA imports.
    """
    import glob
    import os

    nvidia_dirs = glob.glob("/usr/local/lib/python*/dist-packages/nvidia/*/lib")
    torch_dirs = glob.glob("/usr/local/lib/python*/dist-packages/torch/lib")
    cuda_dirs = ["/usr/local/cuda/lib64"]
    new_dirs = [d for d in nvidia_dirs + torch_dirs + cuda_dirs if os.path.isdir(d)]

    if new_dirs:
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        existing = set(ld_path.split(":")) if ld_path else set()
        to_add = [d for d in new_dirs if d not in existing]
        if to_add:
            os.environ["LD_LIBRARY_PATH"] = ":".join(to_add) + (":" + ld_path if ld_path else "")


if __name__ == "__main__":
    _setup_cuda_ld_path()
    main()
