"""Performance tables from the Phase 2 sweep.

Aggregates the three repeats per cell into a mean and a spread, because a
single measurement at low concurrency is unreliable: repeats draw different
prompts, and at 16 prompts the draw moves throughput by tens of percent.

Also checks that every measurement at a given concurrency used the same
number of requests. Request count is derived in code rather than being part
of a RunCell's identity, so a change to the formula cannot invalidate old
records through the staleness check -- it has to be caught here instead.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
OUT = ROOT / "analysis" / "out" / "tables"


def sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load() -> dict[str, dict]:
    path = RESULTS / "sweep.jsonl"
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                latest[r["run_id"]] = r   # later supersedes earlier
    return latest


#: Config fields shown as columns. Any field NOT here is invisible in the
#: table, which is safe only if it never varies -- see `display_collisions`.
DISPLAY_FIELDS = ("mechanism", "weight_precision", "kv_cache_dtype",
                  "attn_backend", "enforce_eager")


def key(rec: dict) -> tuple:
    """Group by the cell's content-addressed identity, not by a field list.

    This used to be a hand-written tuple of (mechanism, weight_precision,
    kv_cache_dtype), which meant any config axis not in that tuple was
    silently averaged over. attn_backend was already missing -- harmless
    only because perf.yaml pinned it -- and adding `enforce_eager` as an
    axis would have merged a 31 tok/s default-mode cell with a 74 tok/s
    eager cell into one "mean" (F021).

    A hand-listed key fails the same way every time: add an axis, forget
    the key, average across it. cell_id already IS the server config's
    identity, so keying on it cannot miss a field.
    """
    return (rec["cell_id"], rec["load"]["workload"],
            rec["load"]["concurrency"])


def label(cfg: dict) -> tuple:
    """The config values shown in the table for a cell."""
    return tuple(cfg.get(f) for f in DISPLAY_FIELDS)


#: Repeats of one cell run back-to-back against an already-booted server,
#: so a gap larger than this means the group spans more than one boot.
#: Generous: a slow repeat at c=64 takes minutes, a boot takes ~5.
SAME_BOOT_WINDOW_S = 30 * 60


def boot_spanning(groups: dict) -> dict[tuple, float]:
    """Groups whose repeats are too far apart in time to share a boot.

    The table used to assert that every cell's repeats came from a single
    boot, and then describe the resulting spread as prompt variance. That
    is a provenance claim the table never checked, and it was false: one
    cell's "three repeats" were two from 05:16 and one from a re-run four
    hours later, so its 1.66x spread was a cross-boot comparison mislabelled
    as prompt variance -- while F020/F021 put the boot term at up to 1.64x
    and the prompt term near 1.17x.

    Detected by timestamp span because no boot id is recorded per cell; that
    gap is the same one runs.json flags with `reconstructed: true`. Stamping
    a boot id live is the proper fix and would make this exact.
    """
    from datetime import datetime

    out: dict[tuple, float] = {}
    for k, rs in groups.items():
        stamps = []
        for r in rs:
            try:
                stamps.append(datetime.fromisoformat(
                    r["ts"].replace("Z", "+00:00")).timestamp())
            except (KeyError, ValueError):
                pass
        if len(stamps) > 1 and (span := max(stamps) - min(stamps)) > SAME_BOOT_WINDOW_S:
            out[k] = span
    return out


def display_collisions(groups: dict) -> dict[tuple, set[str]]:
    """Display labels that map to more than one distinct cell.

    Keying on cell_id makes the aggregation correct, but the table still
    shows a subset of fields. If two genuinely different configs render to
    the same row, the table reads as a duplicate and the reader cannot tell
    which is which. Catching it here means the next axis added shows up as
    a warning rather than as two mysterious identical rows.
    """
    seen: dict[tuple, set[str]] = defaultdict(set)
    for (cid, workload, conc), rs in groups.items():
        seen[(label(rs[0]["config"]), workload, conc)].add(cid)
    return {k: v for k, v in seen.items() if len(v) > 1}


def main() -> None:
    runs = load()
    if not runs:
        print("No sweep results. Run: python -m specfp8.sweep --sweep sweeps/perf.yaml")
        return

    # Consistency guard: same concurrency must mean same request count.
    by_conc: dict[int, set[int]] = defaultdict(set)
    for r in runs.values():
        by_conc[r["load"]["concurrency"]].add(r["load"]["n_requests"])
    inconsistent = {c: sorted(v) for c, v in by_conc.items() if len(v) > 1}

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs.values():
        groups[key(r)].append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    L = [f"# Throughput by configuration and concurrency", "",
         f"*Generated by `analysis/perf_tables.py` from `results/sweep.jsonl` "
         f"at code `{sha()}`. Do not edit by hand.*", ""]

    if inconsistent:
        L += ["> **Warning — inconsistent request counts.** These concurrency "
              "levels contain measurements taken with different numbers of "
              "requests, so their repeats are not comparable:", ""]
        for c, counts in sorted(inconsistent.items()):
            L.append(f"> - concurrency {c}: n in {counts}")
        L.append("")

    L += ["> **Speculative rows carry an error term this table does not show.** "
          "Throughput for a speculative configuration varies 2.16x across "
          "boots while acceptance does not (findings/F020, F021). Repeats "
          "run back-to-back against one booted server, so for cells not "
          "flagged below the spread shown is prompt variance within a single "
          "boot and is silent about the dominant source of error. "
          "Non-speculative rows are unaffected (0.6% across-boot "
          "reproducibility).", "",
          "Mean of 3 repeats, +/- sample standard deviation. Repeats draw "
          "**different** prompts (seed offset by repeat), so the spread "
          "includes prompt-sampling variance, not just measurement noise -- "
          "it answers \"what would a different draw from GSM8K give?\" rather "
          "than \"how repeatable is this run?\".", "",
          "| mech | weights | kv | backend | graphs | conc | n | tok/s | "
          "+/- | req/s | tau | acc |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def stat(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None
        return (statistics.mean(vals),
                statistics.stdev(vals) if len(vals) > 1 else 0.0)

    for k in sorted(groups, key=lambda k: (label(groups[k][0]["config"]), k[2])):
        rs = groups[k]
        cfg = rs[0]["config"]
        mech, wp, kv, backend, eager = label(cfg)
        tok, tok_sd = stat([r["throughput"]["output_tokens_per_s"] for r in rs])
        rps, _ = stat([r["throughput"]["requests_per_s"] for r in rs])
        tau, _ = stat([(r.get("spec") or {}).get("tau") for r in rs])
        acc, _ = stat([r["task"]["accuracy"] for r in rs])
        n = rs[0]["load"]["n_requests"]
        pct = f"{tok_sd / tok * 100:.1f}%" if tok else "-"
        # "graphs" is the reader-facing sense of the flag: enforce_eager
        # means CUDA graphs are OFF, and a column that inverts its meaning
        # between name and value is a column that gets misread.
        graphs = "off" if eager else "on"
        L.append(
            f"| {mech} | {wp} | {kv} | {backend or 'auto'} | {graphs} | "
            f"{k[2]} | {n} | {tok:.1f} | "
            f"{pct} | {rps:.3f} | {f'{tau:.2f}' if tau else '-'} | "
            f"{f'{acc:.1%}' if acc else '-'} |")

    # Scaling: what does concurrency actually buy, per configuration?
    conc_levels = sorted({k[2] for k in groups if k[2] != 1})
    L += ["", "## Throughput scaling with concurrency", "",
          "Ratio to that configuration's own concurrency-1 throughput. "
          "Perfectly linear scaling would give the concurrency itself.", "",
          "| mech | weights | kv | backend | graphs | c=1 tok/s | "
          + " | ".join(f"x{c}" for c in conc_levels) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(conc_levels)]
    cells = sorted({k[0] for k in groups},
                   key=lambda cid: label(next(groups[k][0]["config"]
                                              for k in groups if k[0] == cid)))
    for cid in cells:
        rows_for = {k[2]: groups[k] for k in groups if k[0] == cid}
        if 1 not in rows_for:
            continue
        base = stat([r["throughput"]["output_tokens_per_s"]
                     for r in rows_for[1]])[0]
        if not base:
            continue
        mech, wp, kv, backend, eager = label(rows_for[1][0]["config"])
        row = [f"| {mech} | {wp} | {kv} | {backend or 'auto'} | "
               f"{'off' if eager else 'on'} | {base:.1f} "]
        for c in conc_levels:
            v = stat([r["throughput"]["output_tokens_per_s"]
                      for r in rows_for.get(c, [])])[0]
            row.append(f"| {v / base:.1f}x " if v else "| - ")
        L.append("".join(row) + "|")

    spanning = boot_spanning(groups)
    if spanning:
        warn = ["", "> **Warning — these rows average across boots.** Their "
                "repeats are too far apart in time to come from one booted "
                "server, so the spread shown is NOT prompt variance: it "
                "mixes the boot-level term, which reaches 1.64x on "
                "speculative cells (F021). Treat the mean as "
                "uninterpretable rather than as a noisy estimate.", ""]
        for k, span in sorted(spanning.items(), key=lambda kv: -kv[1]):
            mech, wp, kv_, backend, eager = label(groups[k][0]["config"])
            warn.append(f"> - {mech}/{wp}/{kv_} c={k[2]}: repeats span "
                        f"{span / 3600:.1f}h ({len(groups[k])} measurements)")
        warn.append("")
        L += warn

    collisions = display_collisions(groups)
    if collisions:
        warn = ["", "> **Warning — rows that differ in a field this table "
                "does not show.** Each of these labels covers more than one "
                "distinct cell, so the rows are not duplicates and cannot be "
                "told apart here. Add the differing field to "
                "`DISPLAY_FIELDS`:", ""]
        for (lab, workload, conc), cids in sorted(
                collisions.items(), key=lambda kv: str(kv[0])):
            warn.append(f"> - {lab} @ {workload} c={conc}: "
                        f"{len(cids)} cells ({', '.join(sorted(cids))})")
        warn.append("")
        L = L[:4] + warn + L[4:]

    (OUT / "throughput.md").write_text("\n".join(L) + "\n")
    print(f"  wrote analysis/out/tables/throughput.md ({len(groups)} cells)")
    if collisions:
        print(f"  WARNING {len(collisions)} display label(s) cover multiple "
              f"cells -- a config axis is missing from DISPLAY_FIELDS")
    if spanning:
        print(f"  WARNING {len(spanning)} group(s) average across boots; "
              f"their spread is not prompt variance")
    if inconsistent:
        print(f"  WARNING inconsistent request counts at concurrency "
              f"{sorted(inconsistent)}")


if __name__ == "__main__":
    main()
