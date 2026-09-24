"""Goodput under an SLO, computed offline from stored per-request timings.

Goodput is successful requests per second that *also* met a latency target.
Raw tokens/sec flatters a server that is fast on average while missing its
deadline, which is why R7 asks for goodput rather than throughput.

Everything here derives from `results/runs/<run_id>.jsonl`, so adding an SLO
after the fact costs no GPU. That is the whole reason Phase 2 stores every
request instead of aggregates.

    python analysis/goodput.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
OUT = ROOT / "analysis" / "out" / "tables"

#: Swept rather than fixed (R7), so the conclusion's sensitivity to the
#: threshold is visible instead of hidden inside one arbitrary choice.
ITL_SLOS_MS = [20, 35, 50, 75]
TTFT_SLOS_MS = [200, 500, 1000]


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile. No numpy dependency for one number."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q * (len(ordered) - 1)))))
    return ordered[k]


def load_runs() -> list[dict]:
    path = RESULTS / "sweep.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    # Later records supersede earlier ones for the same measurement.
    latest: dict[str, dict] = {}
    for r in out:
        latest[r["run_id"]] = r
    return list(latest.values())


def load_requests(run_id: str) -> list[dict]:
    path = RESULTS / "runs" / f"{run_id}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


#: Mirrors specfp8.client.MAX_PLAUSIBLE_ITL_MS. Duplicated rather than
#: imported so the analysis layer does not depend on the harness package.
MAX_PLAUSIBLE_ITL_MS = 60_000.0


def itl_is_valid(requests: list[dict]) -> bool:
    """False when a run's inter-token latencies predate the F025 fix.

    Runs recorded before 2026-09-23 carry itl_ms arrays that alternate sign
    and reach +/-Infinity, so every request in them fails any ITL target and
    contributes a spurious 0% to met_fraction. Averaging those together with
    valid runs produced a goodput table showing 40% met for a configuration
    whose raw records show 255/256 requests inside the target.

    Detected rather than dated: a timestamp cutoff would silently mislabel
    any future run that reintroduces the bug, and the condition is directly
    checkable.
    """
    import math
    for r in requests:
        for v in (r.get("itl_ms") or []):
            if not math.isfinite(v) or v < 0 or v > MAX_PLAUSIBLE_ITL_MS:
                return False
    return True


def goodput(requests: list[dict], wall_s: float,
            itl_slo_ms: float, ttft_slo_ms: float) -> dict:
    """Requests/sec that succeeded AND met both latency targets."""
    met = 0
    for r in requests:
        if not r.get("ok"):
            continue
        itls = r.get("itl_ms") or []
        p95 = percentile(itls, 0.95)
        # A request with no inter-token samples produced at most one token;
        # judge it on TTFT alone rather than discarding or failing it.
        if p95 is not None and p95 > itl_slo_ms:
            continue
        if r.get("ttft_ms", 0) > ttft_slo_ms:
            continue
        met += 1
    return {
        "met": met,
        "total": len(requests),
        "goodput_rps": round(met / wall_s, 4) if wall_s else None,
        "met_fraction": round(met / len(requests), 4) if requests else None,
    }


def selftest() -> int:
    """Check the SLO logic against hand-built cases.

    This exists because F025 left goodput.py in an awkward state: the code
    was almost certainly correct, but it had only ever been run on corrupt
    input, so every figure it had produced was 0.0 and nothing distinguished
    "the analysis is right and the data was broken" from "both are broken".
    Unverified is not the same as wrong, and it is not the same as right.

    Synthetic cases settle it without GPU time. Run with --selftest.
    """
    def req(ok=True, itls=(30.0,), ttft=100.0):
        return {"ok": ok, "itl_ms": list(itls), "ttft_ms": ttft}

    cases = [
        ("all within SLO", [req(), req(), req()], 3.0, 3),
        ("p95 ITL over target", [req(itls=(30, 30, 80, 90))], 1.0, 0),
        ("TTFT over target", [req(ttft=1500)], 1.0, 0),
        ("failed request excluded", [req(ok=False), req()], 1.0, 1),
        # A request that produced one token has no inter-token sample; it
        # is judged on TTFT alone rather than being discarded or failed.
        ("single-token judged on TTFT only", [req(itls=())], 1.0, 1),
        ("single-token, bad TTFT", [req(itls=(), ttft=2000)], 1.0, 0),
        ("exactly at threshold counts", [req(itls=(50.0,), ttft=1000.0)], 1.0, 1),
    ]

    failures = 0
    for name, reqs, wall, expect in cases:
        got = goodput(reqs, wall, 50, 1000)["met"]
        ok = got == expect
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<36} "
              f"met={got} expected={expect}")

    for name, vals, q, expect in [("p95 of 1..100", list(range(1, 101)), 0.95, 95),
                                  ("p95 of single value", [10], 0.95, 10)]:
        got = percentile(vals, q)
        ok = got == expect
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<36} "
              f"got={got} expected={expect}")

    print(f"\n{len(cases) + 2 - failures}/{len(cases) + 2} passed")
    return failures


def main() -> None:
    runs = load_runs()
    if not runs:
        print("No sweep results yet. Run: python -m specfp8.sweep "
              "--sweep sweeps/perf.yaml")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = 0
    for rec in runs:
        reqs = load_requests(rec["run_id"])
        # A run with corrupt ITL cannot contribute a meaningful SLO figure,
        # and averaging its spurious zeros into valid runs is worse than
        # dropping it (F025).
        if reqs and not itl_is_valid(reqs):
            skipped += 1
            continue
        wall = rec["throughput"]["wall_clock_s"]
        cfg, load = rec["config"], rec["load"]
        for itl in ITL_SLOS_MS:
            for ttft in TTFT_SLOS_MS:
                g = goodput(reqs, wall, itl, ttft)
                rows.append({
                    # cell_id first: it is the config's full identity, and
                    # the named columns below are a readable subset of it.
                    # Without it a row cannot be traced back to the cell
                    # that produced it, which is the provenance chain in
                    # docs/data-model.md section 5.
                    "cell_id": rec["cell_id"],
                    "mechanism": cfg["mechanism"],
                    "weight_precision": cfg["weight_precision"],
                    "kv_cache_dtype": cfg["kv_cache_dtype"],
                    "attn_backend": cfg.get("attn_backend", "auto"),
                    # Records written before this field existed ran with
                    # CUDA graphs on, which is the engine default, so a
                    # missing value reads as False rather than unknown.
                    "enforce_eager": bool(cfg.get("enforce_eager", False)),
                    "workload": load["workload"],
                    "concurrency": load["concurrency"],
                    # Repeats are separate measurements, not duplicates.
                    "repeat": load.get("repeat", 0),
                    "itl_slo_ms": itl,
                    "ttft_slo_ms": ttft,
                    "goodput_rps": g["goodput_rps"],
                    "met_fraction": g["met_fraction"],
                    "throughput_rps": rec["throughput"]["requests_per_s"],
                    "output_tokens_per_s": rec["throughput"]["output_tokens_per_s"],
                })

    header = list(rows[0].keys())
    csv = [",".join(header)]
    csv += [",".join("" if r[k] is None else str(r[k]) for k in header)
            for r in rows]
    (OUT / "goodput.csv").write_text("\n".join(csv) + "\n")
    print(f"  wrote analysis/out/tables/goodput.csv "
          f"({len(rows)} rows from {len(runs) - skipped} measurements)")
    if skipped:
        print(f"  skipped {skipped} measurement(s) with pre-F025 corrupt ITL")

    # A readable slice at one SLO; the CSV carries the full sweep.
    ref = [r for r in rows if r["itl_slo_ms"] == 50 and r["ttft_slo_ms"] == 1000]
    md = ["# Goodput at p95 ITL <= 50ms, TTFT <= 1000ms", "",
          "*Generated by `analysis/goodput.py` from `results/sweep.jsonl` and "
          "`results/runs/`. Full SLO sweep in `goodput.csv`.*", "",
          "| mech | weights | kv | graphs | conc | rep | goodput req/s | "
          "met | raw req/s | tok/s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(ref, key=lambda r: (r["mechanism"], r["weight_precision"],
                                        r["kv_cache_dtype"],
                                        r["enforce_eager"], r["concurrency"],
                                        r["repeat"])):
        md.append(
            f"| {r['mechanism']} | {r['weight_precision']} | "
            f"{r['kv_cache_dtype']} | "
            f"{'off' if r['enforce_eager'] else 'on'} | "
            f"{r['concurrency']} | {r['repeat']} | "
            f"{r['goodput_rps']} | {r['met_fraction']} | "
            f"{r['throughput_rps']} | {r['output_tokens_per_s']} |")
    (OUT / "goodput.md").write_text("\n".join(md) + "\n")
    print("  wrote analysis/out/tables/goodput.md")

    # SLO attainment across every ITL threshold, not just the reference one.
    #
    # External review noted that a zero-goodput result reported at a single
    # 50ms threshold could be an artifact of that threshold. It is not --
    # but the sweep proving so was computed and never reported, which is the
    # same defect in a different place. The threshold that matters is the
    # chunk cadence: any ITL target below it fails by construction under
    # speculation, and above it the result reverses.
    sweep: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["ttft_slo_ms"] != 1000 or r["met_fraction"] is None:
            continue
        sweep[(r["mechanism"], r["weight_precision"], r["concurrency"],
               r["itl_slo_ms"])].append(r["met_fraction"])
    thresholds = sorted({k[3] for k in sweep})
    S = ["# SLO attainment versus ITL threshold", "",
         "*Generated by `analysis/goodput.py`. Do not edit by hand.*", "",
         "Fraction of requests meeting p95 ITL $\\le$ threshold, with "
         "TTFT $\\le$ 1000 ms, FP8 weights.", "",
         "| conc | mech | " + " | ".join(f"{t} ms" for t in thresholds) + " |",
         "|---|---|" + "---|" * len(thresholds)]
    for conc in sorted({k[2] for k in sweep}):
        for mech in ("dflash", "none"):
            cells = []
            for t in thresholds:
                v = sweep.get((mech, "fp8", conc, t))
                cells.append(f"{sum(v)/len(v):.0%}" if v else "--")
            if any(c != "--" for c in cells):
                S.append(f"| {conc} | {mech} | " + " | ".join(cells) + " |")
    (OUT / "slo_sweep.md").write_text("\n".join(S) + "\n")
    print("  wrote analysis/out/tables/slo_sweep.md")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    main()
