"""Session-scoped throughput regimes for the speculative sweep.

Written after three further vLLM 0.29.0 sessions, run on 2026-10-31, all
reached parity with their eager control. Four revisions of the paper had
read the 0.487/0.768-against-1.014/1.017 split as an engine-release
regression fixed in 0.30.0. With 3 deficit sessions of 6 on 0.29.0 against
0 of 3 on later releases, a one-sided Fisher exact test gives p = 0.24 and
that reading does not survive.

What survives is structural and stronger than the version claim ever was.
Classifying each container session by its within-session default/eager
ratio produces two disjoint groups, and the SAME partition separates the
boot-to-boot dispersion of the graph arm. Deficit and dispersion are not
two findings; they are one session-scoped phenomenon.

The separation also holds boot by boot: pooled over every fixed-prompt
default-arm boot, the two groups do not overlap.

Both axes are within-session quantities, so neither can be produced by a
card-level or session-level offset affecting both arms equally -- which is
the confound that cost this study its first four headline numbers.

    python analysis/session_regimes.py
"""

from __future__ import annotations

import json
import statistics as st
import subprocess
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "out" / "tables"

#: A session is in the deficit regime if its graph arm runs below this
#: fraction of the eager arm measured in the same container. The observed
#: gap is 0.768 to 1.003, so any threshold in that interval gives the same
#: partition; this is not a tuned parameter.
DEFICIT_RATIO = 0.95

BASELINE_GPU = "NVIDIA L4"


def code_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


#: The speculative mechanism every regime figure in the paper is measured
#: on. It is a parameter rather than a constant because boot_variance can
#: now measure the non-speculative cell in the SAME container, but every
#: default in this module stays "dflash" so that F040's numbers regenerate
#: byte-identically.
BASELINE_MECH = "dflash"


def load(gpu: str | None = BASELINE_GPU,
         mech: str = BASELINE_MECH) -> dict[tuple, list[dict]]:
    """Fixed-prompt boot_variance records for one mechanism at c=1."""
    out: dict[tuple, list[dict]] = {}
    with (ROOT / "results" / "probes.jsonl").open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            cfg, env = r.get("config") or {}, r.get("env") or {}
            if r.get("phase") != "boot_variance":
                continue
            if cfg.get("mechanism") != mech:
                continue
            if (r.get("load") or {}).get("concurrency") != 1:
                continue
            if gpu is not None and env.get("gpu") != gpu:
                continue
            if not (r.get("throughput") or {}).get("output_tokens_per_s"):
                continue
            key = (cfg.get("engine_ref"), r["probe_run"], r.get("arm"))
            out.setdefault(key, []).append(r)
    return out


def _cv(v: list[float]) -> float:
    return st.stdev(v) / st.mean(v) * 100 if len(v) > 1 else 0.0


def sessions(gpu: str | None = BASELINE_GPU,
             mech: str = BASELINE_MECH) -> list[dict]:
    """One row per container session that measured both arms."""
    by_arm = load(gpu, mech)
    keys = {(e, p) for e, p, _ in by_arm}
    rows = []
    for engine, run in sorted(keys, key=lambda k: k[1]):
        d = by_arm.get((engine, run, "default"))
        e = by_arm.get((engine, run, "enforce_eager"))
        if not d or not e:
            continue
        tp = lambda rs: [x["throughput"]["output_tokens_per_s"] for x in rs]
        dm, em = st.mean(tp(d)), st.mean(tp(e))
        uu = {x.get("env", {}).get("gpu_uuid") for x in d} - {None}
        rows.append({
            "engine": engine, "session": run[5:16],
            "d_n": len(d), "d_mean": dm, "d_cv": _cv(tp(d)),
            "e_n": len(e), "e_mean": em, "e_cv": _cv(tp(e)),
            "ratio": dm / em,
            "regime": "deficit" if dm / em < DEFICIT_RATIO else "parity",
            "uuid": sorted(uu)[0] if uu else None,
        })
    return rows


def fisher_one_sided(rows: list[dict], baseline: str = "vllm-0.29.0") -> dict:
    """Is the deficit regime associated with the baseline engine release?"""
    n = len(rows)
    k = sum(1 for r in rows if r["regime"] == "deficit")
    grp = [r for r in rows if r["engine"] == baseline]
    a = sum(1 for r in grp if r["regime"] == "deficit")
    p = sum(comb(k, x) * comb(n - k, len(grp) - x) / comb(n, len(grp))
            for x in range(a, min(k, len(grp)) + 1))
    return {"n": n, "deficit": k, "grp_n": len(grp), "grp_deficit": a, "p": p}


def partition_agreement(rows: list[dict]) -> dict:
    """Do the ratio-defined and dispersion-defined partitions coincide?

    If they do, the probability of that happening by chance under random
    assignment is 1 / C(n, k): the k highest-dispersion sessions must be
    exactly the k lowest-ratio ones.
    """
    k = sum(1 for r in rows if r["regime"] == "deficit")
    by_cv = sorted(rows, key=lambda r: -r["d_cv"])[:k]
    agree = {r["session"] for r in by_cv} == {
        r["session"] for r in rows if r["regime"] == "deficit"}
    dcv = [r["d_cv"] for r in rows if r["regime"] == "deficit"]
    pcv = [r["d_cv"] for r in rows if r["regime"] == "parity"]
    dr = [r["ratio"] for r in rows if r["regime"] == "deficit"]
    pr = [r["ratio"] for r in rows if r["regime"] == "parity"]
    return {"agree": agree, "p": 1 / comb(len(rows), k) if agree else None,
            "comb": comb(len(rows), k), "n": len(rows), "k": k,
            "gap": min(pr) - max(dr),
            "cv_deficit": (min(dcv), max(dcv)), "cv_parity": (min(pcv), max(pcv)),
            "cv_disjoint": max(pcv) < min(dcv),
            "ratio_deficit": (min(dr), max(dr)), "ratio_parity": (min(pr), max(pr))}


def boot_separation(rows: list[dict]) -> dict:
    """Pool individual boots and ask whether the two groups overlap.

    Sessions whose graph arm ran without a paired eager arm have no ratio
    and so no regime from `sessions()`. They are classified here by level
    against the parity floor, and counted separately, because excluding a
    measured boot for lacking a control it never had would overstate the
    separation.
    """
    by_arm = load()
    reg = {r["session"]: r["regime"] for r in rows}
    d_boots, p_boots, unpaired = [], [], []
    for (engine, run, arm), recs in by_arm.items():
        if arm != "default":
            continue
        v = [x["throughput"]["output_tokens_per_s"] for x in recs]
        s = run[5:16]
        if s in reg:
            (d_boots if reg[s] == "deficit" else p_boots).extend(v)
        else:
            unpaired.append((engine, s, st.mean(v), len(v)))
    floor = min(p_boots)
    for engine, s, m, _n in unpaired:
        if m >= floor:
            continue
        for (_e, r2, a2), rs in by_arm.items():
            if a2 == "default" and r2[5:16] == s:
                d_boots.extend(x["throughput"]["output_tokens_per_s"] for x in rs)
    boundary = max(d_boots)
    allb = sorted(d_boots + p_boots)
    gaps = [(allb[i + 1] - allb[i], allb[i]) for i in range(len(allb) - 1)]
    inner = max(g for g, lo in gaps if lo != boundary)
    return {"n": len(allb), "n_deficit": len(d_boots), "n_parity": len(p_boots),
            "deficit_range": (min(d_boots), max(d_boots)),
            "parity_range": (min(p_boots), max(p_boots)),
            "gap": min(p_boots) - max(d_boots), "largest_inner_gap": inner,
            "overlap": max(d_boots) >= min(p_boots),
            "unpaired": [(e, s, m, n) for e, s, m, n in unpaired]}


def mechanism_contrast(gpu: str | None = BASELINE_GPU,
                       mechs: tuple[str, ...] = ("dflash", "none")) -> list[dict]:
    """Sessions that measured several mechanisms in ONE container.

    The question F040 left open is whether the regimes are a property of
    speculative decoding or of CUDA graphs generally. Only a within-container
    comparison can answer it: measured in separate containers, each session's
    regime is a free variable again, which is the confound the default/eager
    ratio exists to remove.

    A session appears here only if every mechanism in `mechs` has BOTH arms
    in that same container. A session that measured one mechanism is not a
    half-answer, it is silence, and listing it would invite reading the
    missing arm as a null.
    """
    per = {m: {r["session"]: r for r in sessions(gpu, m)} for m in mechs}
    shared = set.intersection(*(set(v) for v in per.values())) if per else set()
    out = []
    for sess in sorted(shared):
        row = {"session": sess,
               "engine": per[mechs[0]][sess]["engine"],
               "ratios": {m: per[m][sess]["ratio"] for m in mechs},
               "cvs": {m: per[m][sess]["d_cv"] for m in mechs}}
        spec = [m for m in mechs if m != "none"]
        if "none" in row["ratios"] and spec:
            dfc = any(row["ratios"][m] < DEFICIT_RATIO for m in spec)
            non = row["ratios"]["none"] < DEFICIT_RATIO
            row["reading"] = (
                "regime needs speculation" if dfc and not non else
                "CUDA-graph property" if dfc and non else
                "parity session, separates nothing" if not dfc and not non else
                "non-speculative deficient only (unexpected)")
        out.append(row)
    return out


def telemetry(rows: list[dict]) -> list[dict]:
    """Achieved clock, power and utilisation per session and arm.

    F037 refuted power throttling by comparing the two arms inside one
    deficit session. This compares the graph arm ACROSS regimes, which is
    the stronger form: same arm, same configuration, different regime.
    """
    by_arm = load()
    reg = {r["session"]: r["regime"] for r in rows}
    out = []
    for (engine, run, arm), recs in sorted(by_arm.items(), key=lambda kv: kv[0][1]):
        g = [x["gpu_during"] for x in recs
             if (x.get("gpu_during") or {}).get("gpu_samples")]
        if not g:
            continue
        f = lambda n: [x[n] for x in g if x.get(n) is not None]
        mean = lambda n: st.mean(f(n)) if f(n) else None
        out.append({
            "engine": engine, "session": run[5:16], "arm": arm,
            "regime": reg.get(run[5:16], "unpaired"), "n": len(g),
            "tok_s": st.mean([x["throughput"]["output_tokens_per_s"]
                              for x in recs if (x.get("gpu_during") or {}).get("gpu_samples")]),
            "clock": mean("sm_clock_mean_mhz"), "capped": mean("frac_power_capped"),
            "power": mean("power_draw_mean_w"), "thermal": mean("frac_thermal_throttled"),
            "util": mean("util_gpu_mean_pct"),
        })
    return out


def taus() -> list[tuple]:
    by_arm = load()
    out = []
    for (engine, run, arm), recs in sorted(by_arm.items(), key=lambda kv: kv[0][1]):
        if arm != "default":
            continue
        v = [(x.get("spec") or {}).get("tau") for x in recs]
        v = [x for x in v if x]
        if len(v) > 1:
            out.append((engine, run[5:16], len(v), st.mean(v), _cv(v)))
    return out


def main() -> int:
    rows = sessions()
    if not rows:
        print("no paired sessions found", file=sys.stderr)
        return 1
    fi = fisher_one_sided(rows)
    pa = partition_agreement(rows)
    bs = boot_separation(rows)
    tl = telemetry(rows)
    ts = taus()

    L = [f"# Session-scoped throughput regimes\n",
         f"*Generated by `analysis/session_regimes.py` from "
         f"`results/probes.jsonl` at code `{code_sha()}`. Do not edit by hand.*\n",
         f"Sweep: dflash, concurrency 1, fixed prompts, {BASELINE_GPU}. A session "
         f"is `deficit` if its within-container default/eager ratio is below "
         f"{DEFICIT_RATIO}; the observed gap is "
         f"{pa['ratio_deficit'][1]:.3f} to {pa['ratio_parity'][0]:.3f}, so any "
         f"threshold inside it gives this same partition.\n",
         "| engine | session | default | CV | eager | CV | ratio | regime | gpu_uuid |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        uu = (r["uuid"][:16] + "...") if r["uuid"] else "not recorded"
        L.append(f"| {r['engine']} | {r['session']} | {r['d_mean']:.1f} "
                 f"(n={r['d_n']}) | **{r['d_cv']:.2f}%** | {r['e_mean']:.1f} "
                 f"(n={r['e_n']}) | {r['e_cv']:.2f}% | **{r['ratio']:.3f}** "
                 f"| {r['regime']} | {uu} |")

    L += ["", "## The two partitions coincide", "",
          f"- ratio: deficit {pa['ratio_deficit'][0]:.3f}--{pa['ratio_deficit'][1]:.3f}, "
          f"parity {pa['ratio_parity'][0]:.3f}--{pa['ratio_parity'][1]:.3f}",
          f"- graph-arm CV: deficit {pa['cv_deficit'][0]:.2f}--{pa['cv_deficit'][1]:.2f}%, "
          f"parity {pa['cv_parity'][0]:.2f}--{pa['cv_parity'][1]:.2f}% "
          f"(disjoint: {pa['cv_disjoint']})",
          f"- gap between the two ratio groups: **{pa['gap']:.3f}**",
          f"- same partition: **{pa['agree']}**"
          + (f", p = **{pa['p']:.3f}** under random assignment "
             f"(exactly 1/{pa['comb']} = {pa['p']:.4f}, the chance that the "
             f"{pa['k']} highest-dispersion sessions of {pa['n']} are exactly "
             f"the {pa['k']} lowest-ratio ones)" if pa["p"] else ""),
          "",
          "Both axes are within-session, so neither can be produced by a "
          "card-level offset that moves both arms together (F034).", "",
          "## Is the deficit associated with the engine release?", "",
          f"{fi['grp_deficit']} of {fi['grp_n']} sessions on vllm-0.29.0 against "
          f"{fi['deficit'] - fi['grp_deficit']} of {fi['n'] - fi['grp_n']} on later "
          f"releases. One-sided Fisher exact **p = {fi['p']:.3f}**.", "",
          "At this sample size the releases are indistinguishable. The earlier "
          "engine-release reading (F032, F035) is withdrawn.", "",
          "## Boot-level separation", "",
          f"Pooling every fixed-prompt default-arm boot gives n = **{bs['n']}**: "
          f"{bs['n_deficit']} deficit boots spanning "
          f"{bs['deficit_range'][0]:.1f}--{bs['deficit_range'][1]:.1f} tok/s and "
          f"{bs['n_parity']} parity boots spanning "
          f"{bs['parity_range'][0]:.1f}--{bs['parity_range'][1]:.1f} tok/s.", "",
          f"- groups overlap: **{bs['overlap']}**",
          f"- gap between groups: **{bs['gap']:.1f}** tok/s",
          f"- largest gap within either group: {bs['largest_inner_gap']:.1f} tok/s"]
    if bs["unpaired"]:
        L += ["", "Sessions with no paired eager arm, classified by level only:"]
        for e, s, m, n in bs["unpaired"]:
            L.append(f"- {e} {s}: {m:.1f} tok/s (n={n})")

    L += ["", "## Achieved clock and power, by regime", "",
          "F037 refuted power throttling across the two arms of one deficit "
          "session. This is the stronger comparison: the same arm, the same "
          "configuration, across regimes.", "",
          "| engine | session | regime | arm | n | tok/s | SM clock | capped | power | thermal | util |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in tl:
        u = f"{t['util']:.1f}%" if t["util"] is not None else "not sampled"
        L.append(f"| {t['engine']} | {t['session']} | {t['regime']} | {t['arm']} "
                 f"| {t['n']} | {t['tok_s']:.1f} | {t['clock']:.0f} MHz "
                 f"| {t['capped']:.2f} | {t['power']:.1f} W | {t['thermal']:.2f} | {u} |")
    dg = [t for t in tl if t["regime"] in ("deficit", "unpaired") and t["arm"] == "default"]
    pg = [t for t in tl if t["regime"] == "parity" and t["arm"] == "default"]
    if dg and pg:
        L += ["", f"Deficit graph arms clock "
              f"**{min(t['clock'] for t in dg) - max(t['clock'] for t in pg):.0f}"
              f"--{max(t['clock'] for t in dg) - min(t['clock'] for t in pg):.0f} MHz "
              f"higher** than any parity graph arm while producing "
              f"{min(t['tok_s'] for t in pg) - max(t['tok_s'] for t in dg):.1f}"
              f"--{max(t['tok_s'] for t in pg) - min(t['tok_s'] for t in dg):.1f} tok/s "
              f"fewer, drawing less power and being capped less often. A card "
              f"that boosts higher and delivers less work is not at a power "
              f"ceiling. Utilisation sampling was added after the last deficit "
              f"session, so the stall reading this implies is still untested."]

    k = sum(1 for r in rows if r["regime"] == "deficit")
    q = (len(rows) - k) / len(rows)
    mc = mechanism_contrast()
    L += ["", "## Is the regime speculative, or a CUDA-graph property?", "",
          "Sessions that measured both mechanisms inside ONE container. A "
          "session measuring only one mechanism is absent rather than listed "
          "blank: a missing arm is silence, not a null result.", ""]
    if not mc:
        L += ["None yet. `boot_variance` gained the comma-separated mechanism "
              "list on 2026-11-01; run it as",
              "",
              "```",
              "modal run cloud/modal_probe.py --engine bootvar \\",
              "    --sweep dflash,none --repeats 12",
              "```",
              "",
              "Until a session lands in the deficit regime with both "
              "mechanisms measured, this question stays open: a parity "
              "session separates the two readings not at all."]
    else:
        L += ["| session | engine | dflash ratio | none ratio | dflash CV | "
              "none CV | reading |", "|---|---|---|---|---|---|---|"]
        for r in mc:
            L.append(f"| {r['session']} | {r['engine']} "
                     f"| **{r['ratios'].get('dflash', float('nan')):.3f}** "
                     f"| **{r['ratios'].get('none', float('nan')):.3f}** "
                     f"| {r['cvs'].get('dflash', float('nan')):.2f}% "
                     f"| {r['cvs'].get('none', float('nan')):.2f}% "
                     f"| {r.get('reading', '')} |")
        decisive = [r for r in mc
                    if r.get("reading") in ("regime needs speculation",
                                            "CUDA-graph property")]
        L += ["", f"{len(decisive)} of {len(mc)} sessions are decisive; the "
              f"rest are parity sessions, which separate the two readings "
              f"not at all."]

    L += ["", "## What the n-gram comparison can still support", "",
          f"The draft-model localisation (F018) rests on two n-gram sessions "
          f"being quiet. Both sat in the parity regime. At the observed deficit "
          f"frequency of {k} in {len(rows)}, two sessions both land in parity "
          f"with probability ({len(rows)-k}/{len(rows)})^2 = **{q*q:.2f}** even "
          f"if n-gram is affected exactly as dflash is. Two quiet n-gram "
          f"sessions are therefore an unremarkable outcome under the null, and "
          f"the localisation needs n-gram measured inside a deficit session.", ""]

    L += ["", "## Acceptance per session", "",
          "| engine | session | n | mean tau | CV |", "|---|---|---|---|---|"]
    for er, s, n, m, cv in ts:
        L.append(f"| {er} | {s} | {n} | {m:.4f} | **{cv:.2f}%** |")
    tv = [t[4] for t in ts]
    mv = [t[3] for t in ts]
    L += ["", f"{len(ts)} sessions, mean tau {min(mv):.4f}--{max(mv):.4f} "
          f"(a {(max(mv)/min(mv)-1)*100:.1f}% range), CV at most **{max(tv):.2f}%** "
          f"in any session. Acceptance is quiet in BOTH regimes; throughput is "
          f"not. That contrast is what makes the regime split worth reporting."]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "session_regimes.md").write_text("\n".join(L) + "\n")
    print(f"  wrote analysis/out/tables/session_regimes.md ({len(rows)} sessions)")
    for r in rows:
        print(f"    {r['engine']:<13}{r['session']:<13}ratio {r['ratio']:.3f}  "
              f"CV {r['d_cv']:5.2f}%  {r['regime']}")
    print(f"    partition agreement {pa['agree']}  "
          f"engine-release Fisher p={fi['p']:.3f}  "
          f"boot overlap {bs['overlap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
