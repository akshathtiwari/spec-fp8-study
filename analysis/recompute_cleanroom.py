"""Clean-room recomputation of the paper's headline numbers from raw records.

An external audit (3 Nov 2026) named this the most important remaining
task: recompute the published statistics from the raw files rather than
audit the scripts that produced them. This file imports nothing from
analysis/ or specfp8/ -- only the JSONL and the standard library -- so a bug
in the repo's analysis code shows up as a disagreement here.

Limit, stated: the same author wrote both implementations. This is a second
implementation, not an independent replication.

    python analysis/recompute_cleanroom.py
"""
from __future__ import annotations
import os
import json, math, statistics as st
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "out" / "tables"


def records():
    with (ROOT / "results" / "probes.jsonl").open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sessions(platform="modal", gpu="NVIDIA L4", mech="dflash"):
    arms = defaultdict(lambda: defaultdict(list))
    for r in records():
        c = r.get("config") or {}
        if r.get("phase") != "boot_variance" or c.get("mechanism") != mech:
            continue
        if (r.get("load") or {}).get("concurrency") != 1:
            continue
        if (r.get("env") or {}).get("gpu") != gpu or (r.get("platform") or "modal") != platform:
            continue
        v = (r.get("throughput") or {}).get("output_tokens_per_s")
        if v:
            arms[r["probe_run"]][r["arm"]].append(v)
    return arms


def cv(v):
    return st.stdev(v) / st.mean(v) * 100


def ci_ratio(d, e):
    t = {12: 2.201, 11: 2.228, 8: 2.365, 10: 2.262}
    def b(a):
        m = st.mean(a); h = t.get(len(a), 2.2) * st.stdev(a) / math.sqrt(len(a)); return m - h, m + h
    (dl, dh), (el, eh) = b(d), b(e)
    return dl / eh, dh / el


def main() -> int:
    A = sessions()
    paired = {run: a for run, a in A.items() if "default" in a and "enforce_eager" in a}
    rows = sorted(((run, st.mean(a["default"]) / st.mean(a["enforce_eager"]), cv(a["default"]), cv(a["enforce_eager"]))
                   for run, a in paired.items()), key=lambda x: x[0])
    deficit = [r for r in rows if r[1] < 0.95]; parity = [r for r in rows if r[1] >= 0.95]
    n, k = len(rows), len(deficit)
    by_cv = {r[0] for r in sorted(rows, key=lambda x: -x[2])[:k]}
    agree = by_cv == {r[0] for r in deficit}

    # boot pool: every default-arm boot from every Modal L4 dflash session, paired or not
    d_boots, p_boots = [], []
    for run, a in A.items():
        if "default" not in a:
            continue
        lvl = st.mean(a["default"])
        is_def = (run in paired and st.mean(a["default"]) / st.mean(a["enforce_eager"]) < 0.95) \
                 or (run not in paired and lvl < 70)
        (d_boots if is_def else p_boots).extend(a["default"])
    allb = sorted(d_boots + p_boots)
    gaps = [(allb[i + 1] - allb[i], allb[i]) for i in range(len(allb) - 1)]
    inner = max(g for g, lo in gaps if lo != max(d_boots))

    taus = []
    for r in records():
        c = r.get("config") or {}
        if (r.get("phase") == "boot_variance" and c.get("mechanism") == "dflash"
                and (r.get("load") or {}).get("concurrency") == 1 and r.get("arm") == "default"
                and (r.get("env") or {}).get("gpu") == "NVIDIA L4" and (r.get("platform") or "modal") == "modal"):
            t = (r.get("spec") or {}).get("tau")
            if t:
                taus.append((r["probe_run"], t))
    tau_by = defaultdict(list)
    for run, t in taus:
        tau_by[run].append(t)
    tau_cvs = [cv(v) for v in tau_by.values() if len(v) > 1]
    tau_means = [st.mean(v) for v in tau_by.values() if len(v) > 1]

    G = sessions(platform="gcp")
    gcp = []
    for run, a in sorted(G.items()):
        if "default" in a and "enforce_eager" in a:
            lo, hi = ci_ratio(a["default"], a["enforce_eager"])
            gcp.append((run[5:16], st.mean(a["default"]) / st.mean(a["enforce_eager"]), lo, hi, cv(a["default"])))

    checks = []  # (claim, paper value, recomputed, scope note)
    checks.append(("Modal L4 paired DFlash sessions", "nine", str(n), ""))
    checks.append(("deficit ratio band", "0.487-0.768", f"{min(r[1] for r in deficit):.3f}-{max(r[1] for r in deficit):.3f}", ""))
    checks.append(("parity ratio band", "1.003-1.039", f"{min(r[1] for r in parity):.3f}-{max(r[1] for r in parity):.3f}", ""))
    checks.append(("deficit graph-arm CV band", "3.73-10.38%", f"{min(r[2] for r in deficit):.2f}-{max(r[2] for r in deficit):.2f}%", "paper's table includes 13.92 too"))
    checks.append(("parity graph-arm CV band", "0.86-1.95%", f"{min(r[2] for r in parity):.2f}-{max(r[2] for r in parity):.2f}%", ""))
    checks.append(("partitions coincide", "yes", "yes" if agree else "NO", ""))
    checks.append(("partition p = 1/C(n,k)", "0.012 (1/84)", f"{1 / comb(n, k):.4f} (1/{comb(n, k)})", ""))
    checks.append(("pooled boots", "116", str(len(allb)), ""))
    checks.append(("deficit boot range", "29.5-65.6", f"{min(d_boots):.1f}-{max(d_boots):.1f}", ""))
    checks.append(("parity boot range", "75.5-83.5", f"{min(p_boots):.1f}-{max(p_boots):.1f}", ""))
    checks.append(("gap between groups", "9.9", f"{min(p_boots) - max(d_boots):.1f}", ""))
    checks.append(("largest inner gap", "5.7", f"{inner:.1f}", ""))
    checks.append(("tau CV, max over sessions", "below 0.88%", f"{max(tau_cvs):.2f}%", ""))
    checks.append(("tau mean range", "4.0702-4.1594", f"{min(tau_means):.4f}-{max(tau_means):.4f}", ""))
    for s, r, lo, hi, c in gcp:
        checks.append((f"GCP {s} ratio [95% CI]", "1.001 [0.989, 1.013] / 1.028 [1.012, 1.044]",
                       f"{r:.3f} [{lo:.3f}, {hi:.3f}]", f"graph CV {c:.2f}%"))

    # --- derived figures the paper also states ----------------------------
    fac = {"deficit": [], "parity": []}
    for run, a in paired.items():
        ratio = st.mean(a["default"]) / st.mean(a["enforce_eager"])
        tv = tau_by.get(run, [])
        tcv = cv(tv) if len(tv) > 1 else 0.0
        if tcv > 0:
            fac["deficit" if ratio < 0.95 else "parity"].append(cv(a["default"]) / tcv)
    pf_lo, pf_hi = min(fac["parity"]), max(fac["parity"])
    span = (max(tau_means) / min(tau_means) - 1) * 100
    q = (n - k) / n

    # --- claim-level gate -------------------------------------------------
    # Each claim is the exact string the paper must contain, formatted from
    # the recomputed value. check_paper only asks whether a number exists
    # somewhere in the evidence; v1.5.0 shipped "3.73--10.38%" for a band
    # whose true upper end is 13.92, and passed, because 10.38 is a real CV
    # of another session. A wrong range cannot contain the true string.
    # SPECFP8_PAPER points the gate at another text, e.g. an older release,
    # to show it fails there.
    tex = Path(os.environ.get("SPECFP8_PAPER", ROOT / "paper" / "paper.tex")).read_text()
    words = {3: "three", 7: "seven", 9: "nine", 10: "ten"}
    must = [
        ("deficit ratio band", f"{min(r[1] for r in deficit):.3f}--{max(r[1] for r in deficit):.3f}"),
        ("parity ratio band", f"{min(r[1] for r in parity):.3f}--{max(r[1] for r in parity):.3f}"),
        ("deficit CV band", f"{min(r[2] for r in deficit):.2f}--{max(r[2] for r in deficit):.2f}\\%"),
        ("parity CV band", f"{min(r[2] for r in parity):.2f}--{max(r[2] for r in parity):.2f}\\%"),
        ("parity session count", f"{words.get(len(parity), len(parity))} at"),
        ("partition p", f"\\binom{{{n}}}{{{k}}} = {1 / comb(n, k):.3f}"),
        ("pooled boots", f"$n = {len(allb)}$"),
        ("deficit boot range", f"{min(d_boots):.1f}--{max(d_boots):.1f}"),
        ("parity boot range", f"{min(p_boots):.1f}--{max(p_boots):.1f}"),
        ("boot gap", f"{min(p_boots) - max(d_boots):.1f} tok/s"),
        ("tau mean range", f"{min(tau_means):.4f}--{max(tau_means):.4f}"),
        ("tau span", f"range of {span:.1f}\\%"),
        ("tau CV ceiling", f"more than {math.ceil(max(tau_cvs) * 100) / 100:.2f}\\%"),
        ("parity CV/tau factor", f"{pf_lo:.1f}--{pf_hi:.1f}"),
        ("n-gram both-parity", f"({n - k}/{n})^2 = {q * q:.2f}"),
    ]
    # v1.5.0 also said "only 1 of the 6 parity sessions" disperse more on the
    # graph arm than on eager; its own table gave 2 of 6. Count, do not read.
    pw = [r for r in parity if r[2] > r[3]]
    others = [r for r in rows if r[0] != min(rows)[0]]   # all but the first session
    must += [
        ("parity sessions, graph CV > eager CV", f"only {len(pw)} of the {len(parity)} parity"),
        ("parity, both arms CV range", f"between {min(min(r[2], r[3]) for r in parity):.2f}\\% and "
                                       f"{max(max(r[2], r[3]) for r in parity):.2f}\\%"),
        ("figure caption, other sessions", f"{words.get(len(others), len(others)).capitalize()} further sessions"),
        ("figure caption, CV range", f"from {min(r[2] for r in others):.2f}\\% to {max(r[2] for r in others):.2f}\\% CV"),
        ("figure caption, parity count", f"{words.get(len(parity), len(parity))} of them at parity"),
        ("L4 session count", f"{words.get(n, n)} on the L4"),
        ("default CV span", f"from {min(r[2] for r in rows):.2f}\\% to {max(r[2] for r in rows):.2f}\\%"),
    ]
    for s_, r, lo, hi, c in gcp:
        must.append((f"GCP {s_} ratio", f"{r:.3f}"))
        must.append((f"GCP {s_} CI", f"[{lo:.3f}, {hi:.3f}]"))
    failed = [(name, want) for name, want in must if want.replace("\\\\", "\\") not in tex]

    L = ["# Clean-room recomputation of the paper's headline numbers", "",
         "*Generated by `analysis/recompute_cleanroom.py` from `results/probes.jsonl`, "
         "importing nothing from `analysis/` or `specfp8/`. Do not edit by hand.*", "",
         "| claim | paper (v1.5.0) | recomputed now | note |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in checks]
    L += ["", "## Derived figures", "",
          f"- thrCV/tauCV in parity sessions where tau varies: {pf_lo:.1f}--{pf_hi:.1f}",
          f"- thrCV/tauCV in deficit sessions where tau varies: {min(fac['deficit']):.1f}--{max(fac['deficit']):.1f}",
          f"- tau mean span: {span:.1f}%; max tau CV {max(tau_cvs):.4f}%",
          f"- n-gram both-parity probability: ({n - k}/{n})^2 = {q * q:.2f}",
          f"- partition p: 1/C({n},{k}) = {1 / comb(n, k):.4f}",
          "", "## Claim-level gate: the paper must contain each recomputed value", "",
          "| claim | required string | in paper |", "|---|---|---|"]
    L += [f"| {nm} | `{w}` | {'yes' if (nm, w) not in failed else '**NO**'} |" for nm, w in must]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recompute_cleanroom.md").write_text("\n".join(L) + "\n")
    print(f"  claim-level gate: {len(must) - len(failed)}/{len(must)} recomputed values found in paper.tex")
    for nm, w in failed:
        print(f"    MISSING  {nm}: paper does not contain {w!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
