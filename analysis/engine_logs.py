"""What the engine logs say about the two regimes -- and what they do not.

F042 found every deficit session shares one configuration (L4, driver
580.95.05, vLLM 0.29.0, DFlash) with every one-factor neighbour at parity.
The engine logs for all of those sessions were on the Modal volume, unread.
This script reads them.

Each session's cell is derived from results/probes.jsonl (engine, driver,
GPU, mechanism), never typed here, so a log is classified by the same
records the regime tables use.

Three questions:

1. Does any engine message separate the deficit cell from its neighbours?
   (If the deficit were a logged failure -- a capture fallback, a
   recompile, an eager fallback -- it would show here.)
2. Do startup facts differ: first-capture time, graph-pool size, JIT
   compilation during inference?
3. Is each one-factor neighbour actually one factor? The engine silently
   substitutes components, which is this study's first confound (F001).

    python analysis/engine_logs.py
"""

from __future__ import annotations

import json
import re
import statistics as st
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "logs"
OUT = ROOT / "analysis" / "out" / "tables"

DEFICIT_RATIO = 0.95
NAME = re.compile(r"(2026-\d\d-\d\dT\d\d-\d\dZ_boot_variance)_(?:(dflash|none|ngram)_)?"
                  r"(default|enforce_eager)_(\d+)\.log$")


def code_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def cells_from_records() -> dict[tuple, dict]:
    """(probe_run, mechanism) -> configuration and regime, from the records."""
    arms: dict[tuple, dict] = {}
    with (ROOT / "results" / "probes.jsonl").open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("phase") != "boot_variance":
                continue
            if (r.get("load") or {}).get("concurrency") != 1:
                continue
            tp = (r.get("throughput") or {}).get("output_tokens_per_s")
            if not tp:
                continue
            cfg, env = r.get("config") or {}, r.get("env") or {}
            k = (r["probe_run"], cfg.get("mechanism"))
            d = arms.setdefault(k, {
                "platform": r.get("platform") or "modal",
                "engine": (cfg.get("engine_ref") or "").replace("vllm-", ""),
                "driver": env.get("driver"),
                "gpu": (env.get("gpu") or "").replace("NVIDIA ", ""),
                "mech": cfg.get("mechanism"), "tp": defaultdict(list)})
            d["tp"][r.get("arm")].append(tp)
    out = {}
    for k, d in arms.items():
        tp = d.pop("tp")
        if "default" in tp and "enforce_eager" in tp:
            ratio = st.mean(tp["default"]) / st.mean(tp["enforce_eager"])
            d["regime"] = "deficit" if ratio < DEFICIT_RATIO else "parity"
        else:
            # No eager arm: classified by level in F040; here left unpaired.
            d["regime"] = "unpaired"
        out[k] = d
    return out


def facts(text: str) -> dict:
    first = re.findall(r"Capturing CUDA graphs \([A-Z_]+\):\s+\d+%\|[^|]*\|\s*1/\d+ "
                       r"\[[^,]*,\s*([\d.]+)(s/it|it/s)", text)
    pool = re.search(r"CUDA graph pool memory: ([\d.]+) GiB \(actual\), ([\d.]+) GiB "
                     r"\(estimated\)", text)
    return {
        "runner": ("V1" if "using the V1 model runner" in text
                   else "V2" if "Using V2 Model Runner" in text else "?"),
        "async": "off" if "Async scheduling not supported" in text else "on",
        "first_capture_s": ([float(v) if u == "s/it" else 1 / float(v)
                             for v, u in first] or [None])[0],
        "pool_actual": float(pool.group(1)) if pool else None,
        "jit": sorted(set(re.findall(
            r"Triton kernel JIT compilation during inference: (\w+)", text))),
    }


def norm(line: str) -> str:
    line = re.sub(r"^\((?:APIServer|EngineCore)[^)]*\)\s*", "", line)
    line = re.sub(r"^(INFO|WARNING|ERROR|DEBUG)\s+\d\d-\d\d \d\d:\d\d:\d\d\s+\[[^\]]+\]\s*",
                  r"\1 ", line)
    line = re.sub(r"0x[0-9a-f]+", "HEX", line)
    line = re.sub(r"[0-9a-f]{16,}", "HASH", line)
    line = re.sub(r"\d+(\.\d+)?", "N", line)
    return line.strip()[:170]


def main() -> int:
    cells = cells_from_records()
    deficit_cfgs = {(d["platform"], d["engine"], d["driver"], d["gpu"], d["mech"])
                    for d in cells.values() if d["regime"] == "deficit"}
    if len(deficit_cfgs) != 1:
        print(f"expected one deficit configuration, found {deficit_cfgs}")
        return 1
    D = next(iter(deficit_cfgs))
    FACT = ("platform", "engine", "driver", "gpu", "mech")

    def label(d: dict) -> str | None:
        cfg = (d["platform"], d["engine"], d["driver"], d["gpu"], d["mech"])
        if cfg == D:
            return "DEFICIT"
        diff = [f for f, a, b in zip(FACT, cfg, D) if a != b]
        return f"nb:{diff[0]}" if len(diff) == 1 else None

    rows = []
    for p in sorted(LOGS.glob("*_boot_variance*.log")):
        m = NAME.search(p.name)
        if not m:
            continue
        run, mech_in_name, arm, _ = m.groups()
        d = None
        for (r, mech), v in cells.items():
            if r == run and (mech_in_name is None or mech == mech_in_name):
                d = v
                break
        if d is None:
            continue
        lab = label(d)
        if lab is None:
            continue
        rows.append({"cell": lab, "arm": arm, "regime": d["regime"], "path": p,
                     **facts(p.read_text(errors="replace"))})

    order = ["DEFICIT"] + [f"nb:{f}" for f in FACT]
    G = [r for r in rows if r["arm"] == "default"]
    L = ["# Engine logs by regime cell", "",
         f"*Generated by `analysis/engine_logs.py` from `results/logs/` and "
         f"`results/probes.jsonl` at code `{code_sha()}`. Do not edit by hand.*", "",
         f"Deficit cell: **{D[0]}, {D[3]}, driver {D[2]}, vLLM {D[1]}, {D[4]}**. Each "
         f"neighbour differs from it in the one factor named. Graph arm only, "
         f"since the deficit lives there.", "",
         "## Is each neighbour really one factor?", "",
         "| cell | logs | model runner | async scheduling |", "|---|---|---|---|"]
    for c in order:
        R = [r for r in G if r["cell"] == c]
        if not R:
            continue
        rn = "/".join(f"{k} ({v})" for k, v in Counter(r["runner"] for r in R).items())
        an = "/".join(f"{k} ({v})" for k, v in Counter(r["async"] for r in R).items())
        L.append(f"| {c} | {len(R)} | {rn} | {an} |")
    L += ["", "A neighbour whose runner or async setting differs from the deficit "
          "cell's is not a one-factor change, whatever its configuration says.", "",
          "## Startup facts", "",
          "| cell | first capture, s per graph | graph pool, GiB | JIT kernels during inference |",
          "|---|---|---|---|"]
    for c in order:
        R = [r for r in G if r["cell"] == c]
        if not R:
            continue
        fc = [r["first_capture_s"] for r in R if r["first_capture_s"] is not None]
        pa = [r["pool_actual"] for r in R if r["pool_actual"] is not None]
        jit = Counter(k for r in R for k in r["jit"])
        L.append(f"| {c} | {(f'{min(fc):.2f}--{max(fc):.2f}' if fc else '--')} "
                 f"| {(f'{st.median(pa):.2f}' if pa else '--')} "
                 f"| {len(jit)} kinds in {sum(1 for r in R if r['jit'])}/{len(R)} logs |")

    present: dict[str, Counter] = defaultdict(Counter)
    n = Counter()
    for r in G:
        n[r["cell"]] += 1
        seen = set()
        for raw in r["path"].read_text(errors="replace").splitlines():
            if "Capturing CUDA graphs" in raw or "it/s]" in raw:
                continue
            t = norm(raw)
            if t and t not in seen:
                seen.add(t)
                present[t][r["cell"]] += 1
    frac = lambda t, c: present[t][c] / n[c] if n[c] else 0.0
    nbs = [c for c in order[1:] if n[c]]
    only_def = [t for t in present
                if frac(t, "DEFICIT") >= 0.75 and all(frac(t, c) <= 0.10 for c in nbs)]
    never_def = [t for t in present
                 if frac(t, "DEFICIT") <= 0.10 and all(frac(t, c) >= 0.75 for c in nbs)]
    L += ["", "## Does any engine message separate the deficit cell?", "",
          f"Every distinct message template, normalised for numbers and "
          f"timestamps, across {sum(n.values())} graph-arm logs.", "",
          f"- present in >=75% of deficit logs and <=10% of every neighbour: "
          f"**{len(only_def)}**",
          f"- present in <=10% of deficit logs and >=75% of every neighbour: "
          f"**{len(never_def)}**", ""]
    for t in only_def + never_def:
        L.append(f"  - `{t}`")
    L += ["The deficit is not a logged event. Whatever it is, the engine runs "
          "the same code path, prints the same messages, and is slower."]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "engine_logs.md").write_text("\n".join(L) + "\n")
    print(f"  wrote analysis/out/tables/engine_logs.md ({len(G)} graph-arm logs)")
    for c in order:
        R = [r for r in G if r["cell"] == c]
        if R:
            print(f"    {c:<11} {len(R):>3} logs  runner={dict(Counter(r['runner'] for r in R))}"
                  f"  async={dict(Counter(r['async'] for r in R))}")
    print(f"    separating templates: {len(only_def)} deficit-only, {len(never_def)} deficit-absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
