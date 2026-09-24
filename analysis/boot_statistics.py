"""Sample-size-robust dispersion statistics for boot-level measurements.

Written after external review pointed out that this study's headline
dispersion figures were **ranges**, and that a range is not a statistic you
can compare across unequal samples: it grows with n by construction.

Two concrete defects it found, both reproduced here:

1. The tau "noise floor" of 1.32% was the range of four boots. At n=12 the
   range grows to 1.38%. A quantity that increases with sampling is not a
   floor, not a confidence interval, and not a minimum detectable effect.
2. The headline throughput contrast compared a 23-boot range (2.96x)
   against an 8-boot range (1.04x). Subsampling the 23 boots to n=8 gives a
   mean range of about 2.38x, so roughly a fifth of that contrast was
   sample size rather than behaviour.

Reports coefficient of variation, median and IQR, a t-based confidence
interval on the mean, and a crude minimum detectable effect. CV is
dimensionless and does not grow with n, so arms measured with different
boot counts can be compared.

    python analysis/boot_statistics.py
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "out" / "tables"

#: Two-sided t critical values at 95%, indexed by degrees of freedom.
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
        8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18, 15: 2.13, 20: 2.09}


def tcrit(df: int) -> float:
    if df in _T95:
        return _T95[df]
    return 2.09 if df > 20 else 2.20


def boots(mech: str, eager: bool, field: str = "throughput",
          probes_only: bool = False):
    """Boots of one mechanism/arm at concurrency 1.

    `probes_only` restricts to boot_variance runs, which replay one fixed
    prompt set on every boot. That is the controlled measurement of
    BOOT dispersion.

    Pooling in grid records mixes in prompt variation, because sweep.py
    offsets the seed by repeat index so repeats draw different prompts by
    design. An earlier revision excluded them from the acceptance analysis
    for exactly that reason and included them for throughput -- inconsistent,
    and it inflated the headline DFlash CV from 13.9% to 29.9%. External
    review caught it.
    """
    out = []
    p = ROOT / "results" / "sweep.jsonl"
    if probes_only:
        p = None
    for line in (p.read_text().splitlines() if p else []):
        if not line.strip():
            continue
        r = json.loads(line)
        c = r["config"]
        if c["mechanism"] != mech or bool(c.get("enforce_eager")) != eager:
            continue
        if c.get("weight_precision") != "bf16" or c.get("kv_cache_dtype") != "auto":
            continue
        if r["load"]["concurrency"] != 1:
            continue
        # Chunk-counted records undercount throughput by roughly tau.
        if (r.get("throughput") or {}).get("tokens_from_usage") is not True:
            continue
        v = (r["throughput"]["output_tokens_per_s"] if field == "throughput"
             else (r.get("spec") or {}).get("tau"))
        if v:
            out.append(v)
    p = ROOT / "results" / "probes.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("phase") != "boot_variance":
                continue
            if (r.get("config") or {}).get("mechanism") != mech:
                continue
            if (r.get("arm") == "enforce_eager") != eager:
                continue
            if (r.get("load") or {}).get("concurrency") != 1:
                continue
            v = ((r.get("throughput") or {}).get("output_tokens_per_s")
                 if field == "throughput" else (r.get("spec") or {}).get("tau"))
            if v:
                out.append(v)
    return out


def _probe_taus() -> list[float]:
    """tau from boot_variance probes only, where prompts are held fixed."""
    out = []
    p = ROOT / "results" / "probes.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("phase") != "boot_variance" or r.get("arm") != "default":
            continue
        if (r.get("config") or {}).get("mechanism") != "dflash":
            continue
        if (r.get("load") or {}).get("concurrency") != 1:
            continue
        t = (r.get("spec") or {}).get("tau")
        if t:
            out.append(t)
    return out


def describe(v: list[float]) -> dict:
    n = len(v)
    m, s = st.mean(v), (st.stdev(v) if n > 1 else 0.0)
    sem = s / n ** 0.5 if n else 0.0
    q = st.quantiles(v, n=4) if n >= 4 else [m, m, m]
    t = tcrit(n - 1)
    return {
        "n": n, "mean": m, "sd": s, "cv": (s / m * 100) if m else 0.0,
        "median": st.median(v), "iqr": q[2] - q[0],
        "ci_lo": m - t * sem, "ci_hi": m + t * sem,
        "range_ratio": (max(v) / min(v)) if v and min(v) else 0.0,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# Boot-level dispersion, sample-size-robust", "",
         "*Generated by `analysis/boot_statistics.py`. Do not edit by hand.*", "",
         "Coefficient of variation is the comparable statistic here. A "
         "max/min **range grows with n by construction**, so ranges from "
         "arms with different boot counts cannot be compared; they are "
         "shown for continuity with earlier revisions and should not be "
         "used for inference.", "",
         "| series | n | median | IQR | CV | 95% CI on mean | max/min |",
         "|---|---|---|---|---|---|---|"]

    series = [("DFlash default", "dflash", False),
              ("DFlash eager", "dflash", True),
              ("n-gram default", "ngram", False),
              ("n-gram eager", "ngram", True)]
    for label, mech, eager in series:
        v = boots(mech, eager, probes_only=True)
        if len(v) < 2:
            continue
        d = describe(v)
        L.append(f"| {label} | {d['n']} | {d['median']:.1f} | {d['iqr']:.1f} "
                 f"| **{d['cv']:.2f}%** | [{d['ci_lo']:.1f}, {d['ci_hi']:.1f}] "
                 f"| {d['range_ratio']:.2f}x |")

    L += ["", "### For contrast: pooled across all observed runs", "",
          "Includes grid records, whose repeats draw different prompts, so "
          "these mix boot and prompt variation and are **not** a controlled "
          "boot-dispersion estimate. Shown because an earlier revision "
          "reported them as one.", "",
          "| series | n | CV (pooled) |", "|---|---|---|"]
    for label, mech, eager in series:
        v = boots(mech, eager)
        if len(v) > 1:
            L.append(f"| {label} | {len(v)} | {describe(v)['cv']:.2f}% |")

    # tau under REPEATED IDENTICAL MEASUREMENT.
    #
    # Only boot_variance probes qualify. They replay one fixed prompt set
    # (seed 1234) on every boot, so the only thing varying is the boot.
    # Grid records cannot be pooled in: sweep.py offsets the seed by repeat
    # index precisely so repeats draw DIFFERENT prompts, which is prompt
    # variance, a different quantity. Pooling them inflated the dispersion
    # and turned 2 distinct tau values into 12.
    L += ["", "## Acceptance ($\\tau$) under repeated identical boots", "",
          "Boot-variance probes only: one fixed prompt set replayed on "
          "every boot. Grid records are excluded because their repeats draw "
          "different prompts by design, which measures prompt variance "
          "rather than boot variance.", ""]
    tv = [t for t in _probe_taus()]
    if len(tv) > 1:
        d = describe(tv)
        distinct = sorted(set(round(x, 4) for x in tv))
        L += [f"- n = {d['n']}, mean {d['mean']:.4f}, sd {d['sd']:.4f}",
              f"- **CV = {d['cv']:.2f}%**",
              f"- 95% CI on the mean: [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]",
              f"- max/min range: {(max(tv)/min(tv)-1)*100:.2f}% "
              f"(was reported as a 1.32% \"noise floor\" at n=4; the range "
              f"**grew** with n, which is what ranges do)",
              "",
              f"- Distinct values observed: {distinct}. Acceptance is "
              f"near-deterministic given fixed prompts rather than "
              f"continuously noisy, which is a further reason \"noise "
              f"floor\" was the wrong description."]

    # demonstrate the range/n dependence directly
    v = boots("dflash", False)
    if len(v) >= 12:
        random.seed(0)
        sub = [max(s) / min(s) for s in
               (random.sample(v, 8) for _ in range(200))]
        L += ["", "## Why the range was not comparable", "",
              f"DFlash default at full n={len(v)} has max/min "
              f"{max(v)/min(v):.2f}x. Subsampling the same data to n=8, "
              f"200 draws, gives a mean max/min of **{st.mean(sub):.2f}x** "
              f"(observed {min(sub):.2f}x to {max(sub):.2f}x). The eager arm "
              f"was reported at n=8. Roughly a fifth of the headline "
              f"contrast was sample size, not behaviour."]

    (OUT / "boot_statistics.md").write_text("\n".join(L) + "\n")
    print(f"  wrote analysis/out/tables/boot_statistics.md")
    for label, mech, eager in series:
        v = boots(mech, eager, probes_only=True)
        if len(v) > 1:
            d = describe(boots(mech, eager, probes_only=True))
            print(f"    {label:<16} n={d['n']:<3} CV {d['cv']:5.2f}%  "
                  f"median {d['median']:6.1f}   (probe-only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
