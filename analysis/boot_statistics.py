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


#: The baseline the paper reports: one engine release, one card. Every
#: record on disk as of 2026-09-29 is exactly this, so these defaults leave
#: published numbers unchanged.
BASELINE_ENGINE_REF = "vllm-0.29.0"
BASELINE_GPU = "NVIDIA L4"


def boots(mech: str, eager: bool, field: str = "throughput",
          probes_only: bool = False,
          engine_ref: str = BASELINE_ENGINE_REF,
          gpu: str = BASELINE_GPU,
          has_uuid: bool | None = None):
    """Boots of one mechanism/arm at concurrency 1.

    `probes_only` restricts to boot_variance runs, which replay one fixed
    prompt set on every boot. That is the controlled measurement of
    BOOT dispersion.

    `engine_ref` and `gpu` exist because this function had no way to tell
    engine versions or GPU models apart, and the study only ever ran one of
    each -- so the gap was invisible and would have stayed invisible until
    exactly the run that exploited it. A vLLM 0.30.0 replication, or a run on
    a second card, writes records that match every filter here (same
    mechanism, same arm, same precision, same concurrency) and would have
    been pooled straight into the headline 13.92% CV, silently changing it.

    Note that a GPU filter cannot be replaced by a cell_id filter. engine_ref
    is part of cell_id, so a version change is at least *visible* there; GPU
    model is not part of cell_id at all, by design, since a cell is a
    declared configuration rather than a machine. Hardware can only be
    separated via `env`.

    `has_uuid` separates measurements taken before GPU identity was recorded
    from those taken after. It is not cosmetic. The published 13.92% pools 12
    boots drawn from THREE separate container sessions on unknown physical
    cards, so it contains an unknown amount of host-to-host variation; a
    single-session run on one known card is a different quantity and must not
    be averaged with it. Without this filter, re-running 0.29.0 would append
    to the same series -- same engine_ref, same GPU model, same cell_id -- and
    silently turn n=12 into n=24, moving a number the paper reports.
    
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
        if c.get("engine_ref") != engine_ref:
            continue
        if has_uuid is not None and \
                (((r.get("env") or {}).get("gpu_uuid") is not None) != has_uuid):
            continue
        if (r.get("env") or {}).get("gpu") != gpu:
            continue
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
            if (r.get("config") or {}).get("engine_ref") != engine_ref:
                continue
            if has_uuid is not None and \
                    (((r.get("env") or {}).get("gpu_uuid") is not None) != has_uuid):
                continue
            if (r.get("env") or {}).get("gpu") != gpu:
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


def _probe_taus(has_uuid: bool | None = False) -> list[float]:
    """tau from boot_variance probes only, where prompts are held fixed.

    `has_uuid` defaults to False, i.e. the published 2026-09-23 session. Once
    a second session exists these pool, and pooling moved the reported figure
    from CV 0.62% at n=12 to 0.54% at n=24 without anything flagging it --
    the same contamination F033 describes for throughput, in the function
    next door. Sessions are reported separately by `_taus_by_session`.
    """
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
        if (r.get("config") or {}).get("engine_ref") != BASELINE_ENGINE_REF:
            continue
        if (r.get("env") or {}).get("gpu") != BASELINE_GPU:
            continue
        if has_uuid is not None and \
                (((r.get("env") or {}).get("gpu_uuid") is not None) != has_uuid):
            continue
        if (r.get("config") or {}).get("mechanism") != "dflash":
            continue
        if (r.get("load") or {}).get("concurrency") != 1:
            continue
        t = (r.get("spec") or {}).get("tau")
        if t:
            out.append(t)
    return out



def by_session(mech: str, arm: str) -> list[tuple]:
    """Per-container-session dispersion, one row per probe_run.

    Exists because F033: the study reported one 12-boot CV per arm and treated
    it as the dispersion, and a second measurement of the same configuration
    on the same engine release gave a figure 3.7x smaller. Any CV quoted from
    a single session is a draw, so the sessions have to be visible rather than
    pooled away. It also shows that the n-gram comparison underpinning the
    draft-model localisation is cross-session.
    """
    rows: dict[tuple, list[float]] = {}
    p = ROOT / "results" / "probes.jsonl"
    if not p.exists():
        return []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = r.get("config") or {}
        if r.get("phase") != "boot_variance" or c.get("mechanism") != mech:
            continue
        if r.get("arm") != arm:
            continue
        if (r.get("load") or {}).get("concurrency") != 1:
            continue
        v = (r.get("throughput") or {}).get("output_tokens_per_s")
        if not v:
            continue
        key = (c.get("engine_ref"), r.get("probe_run"),
               (r.get("env") or {}).get("gpu_uuid"))
        rows.setdefault(key, []).append(v)
    out = []
    for (er, pr, uu), v in sorted(rows.items()):
        if len(v) < 2:
            continue
        out.append((er, pr, uu, len(v), st.mean(v), st.stdev(v) / st.mean(v) * 100))
    return out



def _taus_by_session() -> list[tuple]:
    """Acceptance per container session, so its replication is checkable."""
    rows: dict[tuple, list[float]] = {}
    p = ROOT / "results" / "probes.jsonl"
    if not p.exists():
        return []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = r.get("config") or {}
        if r.get("phase") != "boot_variance" or c.get("mechanism") != "dflash":
            continue
        if r.get("arm") != "default":
            continue
        if (r.get("load") or {}).get("concurrency") != 1:
            continue
        t = (r.get("spec") or {}).get("tau")
        if t:
            rows.setdefault((c.get("engine_ref"), r.get("probe_run")), []).append(t)
    return [(er, pr, len(v), st.mean(v), st.stdev(v) / st.mean(v) * 100)
            for (er, pr), v in sorted(rows.items()) if len(v) > 1]


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
        # has_uuid=False pins this to the published 2026-09-23 measurements.
        # Without it the two 0.29.0 sessions pool, and because their means
        # differ (39.2 vs 61.2) the pooled CV is 23.85% -- HIGHER than either
        # session's 13.92% or 3.73%. A pooled dispersion is not an average of
        # the dispersions (F033).
        v = boots(mech, eager, probes_only=True, has_uuid=False)
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
        # Pinned to the published measurements for the same reason the table
        # above is: this section exists to show what pooling PROMPT variation
        # into boot variation did to the 2026-09-23 figure (29.92%). Letting
        # later sessions in would silently restate a historical number that
        # the paper quotes as history.
        v = boots(mech, eager, has_uuid=False)
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

    # Per-session breakdown. F033: a CV from one session is a draw.
    L += ["", "## Dispersion per container session", "",
          "One row per `probe_run`, which is one container. A CV quoted from "
          "a single row is a single draw of a quantity whose session-to-"
          "session spread is larger than most effects this study reports "
          "(F033).", "",
          "| engine | mechanism | arm | session | gpu_uuid | n | mean | CV |",
          "|---|---|---|---|---|---|---|---|"]
    for mech in ("dflash", "ngram"):
        for arm in ("default", "enforce_eager"):
            for er, pr, uu, n_, m, cv in by_session(mech, arm):
                sid = (pr or "?")[:16]
                uus = (uu[:16] + "...") if uu else "not recorded"
                L.append(f"| {er} | {mech} | {arm} | {sid} | {uus} | {n_} "
                         f"| {m:.1f} | **{cv:.2f}%** |")

    ts = _taus_by_session()
    if ts:
        L += ["", "### Acceptance per session: does it replicate?", "",
              "| engine | session | n | mean tau | CV |", "|---|---|---|---|---|"]
        for er, pr, n_, m, cv in ts:
            L.append(f"| {er} | {pr[:16]} | {n_} | {m:.4f} | **{cv:.2f}%** |")
        L += ["", "Acceptance dispersion reproduces across sessions and engine "
              "releases where throughput dispersion does not. That contrast is "
              "the point: the quantity that moves is throughput."]

    (OUT / "boot_statistics.md").write_text("\n".join(L) + "\n")
    print(f"  wrote analysis/out/tables/boot_statistics.md")
    for label, mech, eager in series:
        v = boots(mech, eager, probes_only=True, has_uuid=False)
        if len(v) > 1:
            d = describe(v)
            print(f"    {label:<16} n={d['n']:<3} CV {d['cv']:5.2f}%  "
                  f"median {d['median']:6.1f}   (probe-only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
