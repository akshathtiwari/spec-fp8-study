"""Figure 1: boot-to-boot throughput variance by mechanism and execution path.

One dot per boot of an identical configuration, drawn from the raw records
rather than transcribed, so the figure cannot drift from the data it
claims to show.

The point of the figure is the *shape*, which a table communicates badly:
DFlash's default arm is a wide, continuously filled distribution, while its
eager arm and both n-gram arms are tight groups.

An earlier version of this figure was captioned as showing two clusters. It
did not. At n=5 a gap looked like separation; at n=23 the distribution fills
in and BIC favours a single component (findings/F029). The figure now plots
every boot and lets the filling-in be visible, which is what refuted the
claim.

    python analysis/fig_bimodality.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"

#: Records whose token counts came from streamed chunks undercount
#: throughput by roughly tau under speculation and are not comparable.
def _valid(rec: dict) -> bool:
    return (rec.get("throughput") or {}).get("tokens_from_usage") is True


def dflash_boots() -> list[float]:
    """Every valid DFlash default-arm measurement at concurrency 1.

    Taken from sweep.jsonl across every session, because the whole claim is
    about variation between boots and restricting to one session would
    remove the effect being drawn.
    """
    out = []
    path = ROOT / "results" / "sweep.jsonl"
    # Deliberately NOT deduplicated by run_id. A run_id identifies a
    # (cell, workload, concurrency, seed, repeat) tuple, so the same id
    # recurs across sessions -- and those recurrences are different boots,
    # which is precisely the variation this figure exists to show.
    # Deduplicating collapsed 12 measurements to 5 and hid half the effect.
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = r["config"]
        if c["mechanism"] != "dflash" or c.get("enforce_eager"):
            continue
        if c["weight_precision"] != "bf16" or c["kv_cache_dtype"] != "auto":
            continue
        if r["load"]["concurrency"] != 1 or not _valid(r):
            continue
        out.append(r["throughput"]["output_tokens_per_s"])
    return out


def probe_boots(mechanism: str, arm: str) -> list[float]:
    """Boot-variance probe measurements at concurrency 1."""
    out = []
    path = ROOT / "results" / "probes.jsonl"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("phase") != "boot_variance" or r.get("arm") != arm:
            continue
        if (r.get("config") or {}).get("mechanism") != mechanism:
            continue
        if (r.get("load") or {}).get("concurrency") != 1:
            continue
        t = (r.get("throughput") or {}).get("output_tokens_per_s")
        if t:
            out.append(t)
    return out


def main() -> int:
    series = [
        ("DFlash\ndefault", dflash_boots() + probe_boots("dflash", "default"),
         "#c0392b"),
        ("DFlash\neager", probe_boots("dflash", "enforce_eager"), "#e67e22"),
        ("n-gram\ndefault", probe_boots("ngram", "default"), "#2c6fbb"),
        ("n-gram\neager", probe_boots("ngram", "enforce_eager"), "#5a9bd5"),
    ]
    series = [(n, v, c) for n, v, c in series if v]
    if not series:
        print("no data", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    for i, (name, vals, colour) in enumerate(series):
        # Deterministic horizontal offsets: no RNG, so the figure is
        # byte-reproducible from the same records.
        n = len(vals)
        offs = [((j % 5) - 2) * 0.045 for j in range(n)]
        ax.scatter([i + o for o in offs], vals, s=46, color=colour,
                   alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3)
        lo, hi = min(vals), max(vals)
        ax.plot([i - 0.28, i + 0.28], [lo, lo], color=colour, lw=0.8, alpha=0.4)
        ax.plot([i - 0.28, i + 0.28], [hi, hi], color=colour, lw=0.8, alpha=0.4)
        ax.annotate(f"{hi/lo:.2f}×", xy=(i + 0.33, (lo + hi) / 2),
                    fontsize=9, color=colour, va="center")

    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([n for n, _, _ in series], fontsize=9)
    ax.set_ylabel("throughput (tok/s), concurrency 1", fontsize=9)
    ax.set_xlim(-0.6, len(series) - 0.15)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"bimodality.{ext}", dpi=200, bbox_inches="tight")

    # The figure's own numbers must live in the derived layer, or a figure
    # citing "2.96x" has no record behind it and check_paper.py is right to
    # reject it. A plot is a claim like any other.
    tables = ROOT / "analysis" / "out" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    L = ["# Figure 1 data — boot-to-boot throughput by mechanism", "",
         "*Generated by `analysis/fig_bimodality.py` from `results/`. "
         "Do not edit by hand.*", "",
         "| series | boots | min | max | spread | values |",
         "|---|---|---|---|---|---|"]
    for name, vals, _ in series:
        flat = name.replace("\n", " ")
        L.append(f"| {flat} | {len(vals)} | {min(vals):.2f} | {max(vals):.2f} "
                 f"| **{max(vals)/min(vals):.2f}x** | "
                 f"{', '.join(f'{v:.2f}' for v in sorted(vals))} |")
    (tables / "figure1_data.md").write_text("\n".join(L) + "\n")

    for name, vals, _ in series:
        flat = name.replace("\n", " ")
        print(f"  {flat:<16} n={len(vals):<3} "
              f"{min(vals):6.1f}–{max(vals):6.1f}  spread {max(vals)/min(vals):.2f}x")
    print("  wrote paper/figures/bimodality.{pdf,png} and "
          "analysis/out/tables/figure1_data.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
