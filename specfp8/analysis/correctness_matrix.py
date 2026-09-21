"""Matched-pair correctness analysis — Tests 1 and 2, computed offline.

`probe.py` scores each cell in isolation, so it can only run Test 3 (the
garbage backstop) and passes a placeholder for Test 1. Tests 1 and 2 are
*comparisons between cells*, which means they cannot be computed while a
single cell is in flight — but they can be computed afterwards from the
stored outputs, with no GPU.

Test 1 (design §4) — speculation equivalence. Speculative decoding is
lossless with respect to the target it verifies against, whatever precision
that target is in. So for each precision setting, compare each speculative
mechanism against `mechanism=none` at the *same* precision. Divergence is
precision-neutral evidence of a broken speculative path, which is exactly
the silent-failure mode H1 predicts and a crash-only matrix would miss.

Test 2 (design §4) — quantization damage. Non-speculative FP8 KV against
non-speculative BF16 KV, same weights. Divergence here is expected and is
the thing being measured, not a failure.

Reading the numbers: Test 1 is only interpretable against its own BF16
baseline. Greedy decoding is not bit-exact across differing batch shapes,
so some divergence is present even at BF16, and the question is always
whether FP8 makes it materially worse — never whether it reaches 100%.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_cells(results_dir: str | Path) -> dict[str, dict]:
    """Latest record per cell_id from cells.jsonl."""
    path = Path(results_dir) / "cells.jsonl"
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "cell_id" in rec:
                latest[rec["cell_id"]] = rec
    return latest


def load_outputs(results_dir: str | Path, cell_id: str) -> list[str]:
    """Generated texts for a cell, ordered by prompt index.

    The prompt set is fixed and identical across cells, so index i is the
    same prompt everywhere and outputs are directly comparable.
    """
    path = Path(results_dir) / "requests" / f"{cell_id}.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: r.get("index", 0))
    return [r.get("output_text", "") for r in rows]


def prefix_agreement(a: str, b: str) -> float:
    """Fraction of whitespace tokens agreeing before the first divergence.

    Whitespace tokens, not model tokens: the harness only stores text, and
    re-tokenising would need the engine's tokeniser, which the HTTP-only
    boundary keeps out of this process. Reported as a shape indicator —
    "diverges immediately" vs "diverges near the end" — not as an exact
    token count.
    """
    ta, tb = a.split(), b.split()
    if not ta and not tb:
        return 1.0
    n = min(len(ta), len(tb))
    i = 0
    while i < n and ta[i] == tb[i]:
        i += 1
    longest = max(len(ta), len(tb))
    return i / longest if longest else 1.0


def compare(baseline: list[str], candidate: list[str]) -> dict:
    """Exact-match rate and mean prefix agreement over aligned outputs."""
    n = min(len(baseline), len(candidate))
    if n == 0:
        return {"n": 0, "exact_match_rate": None, "mean_prefix_agreement": None}
    exact = sum(1 for i in range(n) if baseline[i] == candidate[i])
    prefix = sum(prefix_agreement(baseline[i], candidate[i]) for i in range(n))
    return {
        "n": n,
        "exact_match_rate": exact / n,
        "mean_prefix_agreement": prefix / n,
    }


def _precision_key(cfg: dict) -> tuple[str, str]:
    return (cfg["weight_precision"], cfg["kv_cache_dtype"])


def analyse(results_dir: str | Path, model: str | None = None) -> dict:
    """Run Tests 1 and 2 over every matched pair present in the results."""
    cells = load_cells(results_dir)
    if model:
        cells = {k: v for k, v in cells.items() if v["config"]["model"] == model}

    # index: (weight, kv) -> mechanism -> cell_id
    by_precision: dict[tuple[str, str], dict[str, str]] = {}
    for cid, rec in cells.items():
        cfg = rec["config"]
        by_precision.setdefault(_precision_key(cfg), {})[cfg["mechanism"]] = cid

    outputs = {cid: load_outputs(results_dir, cid) for cid in cells}

    test1 = []
    for precision, mechs in sorted(by_precision.items()):
        base_id = mechs.get("none")
        if base_id is None or not outputs.get(base_id):
            continue
        for mech, cid in sorted(mechs.items()):
            if mech == "none" or not outputs.get(cid):
                continue
            row = compare(outputs[base_id], outputs[cid])
            row.update({
                "weight_precision": precision[0],
                "kv_cache_dtype": precision[1],
                "mechanism": mech,
            })
            test1.append(row)

    # Test 2: same weights, FP8 KV vs BF16 KV, non-speculative only.
    test2 = []
    for weight in sorted({p[0] for p in by_precision}):
        base_id = by_precision.get((weight, "auto"), {}).get("none")
        if base_id is None or not outputs.get(base_id):
            continue
        for precision, mechs in sorted(by_precision.items()):
            if precision[0] != weight or precision[1] == "auto":
                continue
            cid = mechs.get("none")
            if cid is None or not outputs.get(cid):
                continue
            row = compare(outputs[base_id], outputs[cid])
            row.update({
                "weight_precision": weight,
                "kv_cache_dtype": precision[1],
            })
            test2.append(row)

    return {"test1": test1, "test2": test2}


def format_report(result: dict) -> str:
    """Human-readable report, with the BF16 baseline called out explicitly."""
    lines: list[str] = []

    lines.append("TEST 1 — speculation equivalence (spec vs none, matched precision)")
    lines.append("  Lossless speculation would be 1.000. Read each row against the")
    lines.append("  bf16/auto row for the same mechanism, not against 1.000.")
    lines.append("")
    lines.append(f"  {'mechanism':10}{'weight':8}{'kv':10}{'n':>4}{'exact':>9}{'prefix':>9}")
    lines.append("  " + "-" * 50)
    for r in result["test1"]:
        lines.append(
            f"  {r['mechanism']:10}{r['weight_precision']:8}{r['kv_cache_dtype']:10}"
            f"{r['n']:>4}{r['exact_match_rate']:>9.3f}{r['mean_prefix_agreement']:>9.3f}"
        )

    # The comparison that matters: does FP8 KV degrade equivalence relative
    # to the BF16 baseline for the same mechanism?
    base = {
        r["mechanism"]: r
        for r in result["test1"]
        if r["weight_precision"] == "bf16" and r["kv_cache_dtype"] == "auto"
    }
    deltas = []
    for r in result["test1"]:
        b = base.get(r["mechanism"])
        if not b or (r["weight_precision"], r["kv_cache_dtype"]) == ("bf16", "auto"):
            continue
        deltas.append(
            f"  {r['mechanism']:10}{r['weight_precision']}/{r['kv_cache_dtype']:10}"
            f" exact {b['exact_match_rate']:.3f} -> {r['exact_match_rate']:.3f}"
            f"  (delta {r['exact_match_rate'] - b['exact_match_rate']:+.3f})"
        )
    if deltas:
        lines.append("")
        lines.append("  vs the mechanism's own bf16/auto baseline:")
        lines.extend(deltas)

    lines.append("")
    lines.append("TEST 2 — quantization damage (non-spec FP8 KV vs non-spec BF16 KV)")
    lines.append("  Divergence here is expected and is the measurement, not a failure.")
    lines.append("")
    lines.append(f"  {'weight':8}{'kv':10}{'n':>4}{'exact':>9}{'prefix':>9}")
    lines.append("  " + "-" * 40)
    for r in result["test2"]:
        lines.append(
            f"  {r['weight_precision']:8}{r['kv_cache_dtype']:10}"
            f"{r['n']:>4}{r['exact_match_rate']:>9.3f}{r['mean_prefix_agreement']:>9.3f}"
        )

    return "\n".join(lines)
