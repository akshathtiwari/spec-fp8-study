"""Paired comparison of task accuracy across configurations (McNemar).

Every cell is put the same pinned problems in the same order, so comparing two
configurations is a paired problem, not two independent samples. That matters
practically: detecting a 10-point drop needs ~356 problems per arm treating the
arms as independent, but only ~234 pairs by McNemar at a 20% disagreement rate
(findings/F008). Differencing two accuracy rates throws that away.

McNemar looks only at the problems where the two configurations disagree. If a
configuration is harmless, disagreements should fall roughly evenly in both
directions; systematic damage shows up as an imbalance. Problems both get right,
or both get wrong, carry no information about which is better and are excluded —
which is exactly why the test is more powerful than comparing rates.

The exact binomial test is used rather than the chi-squared approximation
because the discordant count is often small, and chi-squared is unreliable
there.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path


def load_quality(results_dir: str | Path) -> dict[str, list[bool | None]]:
    """cell_id -> per-problem correctness, ordered by problem index."""
    qdir = Path(results_dir) / "quality"
    out: dict[str, list[bool | None]] = {}
    if not qdir.is_dir():
        return out
    for path in sorted(qdir.glob("*.jsonl")):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows.sort(key=lambda r: r.get("index", 0))
        out[path.stem] = [r.get("correct") for r in rows]
    return out


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value, by summing tail probabilities."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(n + 1)]
    observed = probs[k]
    # Two-sided: everything at least as extreme as what was seen.
    return min(1.0, sum(pr for pr in probs if pr <= observed * (1 + 1e-9)))


def mcnemar(a: list[bool | None], b: list[bool | None]) -> dict:
    """Paired comparison of `a` (baseline) against `b` (candidate).

    Items unparseable in either arm are dropped rather than scored wrong: an
    unextractable answer is a different failure from a wrong one, and counting
    it as wrong would attribute a formatting change to accuracy.
    """
    n = min(len(a), len(b))
    both_right = both_wrong = only_a = only_b = dropped = 0
    for i in range(n):
        x, y = a[i], b[i]
        if x is None or y is None:
            dropped += 1
            continue
        if x and y:
            both_right += 1
        elif x and not y:
            only_a += 1          # baseline right, candidate wrong
        elif y and not x:
            only_b += 1          # candidate right, baseline wrong
        else:
            both_wrong += 1

    discordant = only_a + only_b
    p = binom_two_sided(min(only_a, only_b), discordant) if discordant else 1.0
    scored = both_right + both_wrong + discordant
    return {
        "n_compared": scored,
        "dropped_unparseable": dropped,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "baseline_only": only_a,
        "candidate_only": only_b,
        "discordant": discordant,
        "acc_baseline": both_right + only_a,
        "acc_candidate": both_right + only_b,
        "delta": (only_b - only_a) / scored if scored else 0.0,
        "p_value": p,
        "significant_at_05": p < 0.05,
    }


def analyse(results_dir: str | Path, model: str | None = None) -> list[dict]:
    """Compare every cell against the matched BF16/auto baseline."""
    results_dir = Path(results_dir)
    per_cell = load_quality(results_dir)
    if not per_cell:
        return []

    cfg: dict[str, dict] = {}
    with open(results_dir / "cells.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                cfg[rec["cell_id"]] = rec["config"]

    rows = []
    for cid_a, cid_b in combinations(sorted(per_cell), 2):
        ca, cb = cfg.get(cid_a), cfg.get(cid_b)
        if not ca or not cb:
            continue
        if model and ca["model"] != model:
            continue
        # Only compare cells that differ in precision, holding everything
        # else fixed — otherwise the comparison confounds several changes.
        fixed = ("model", "mechanism", "attn_backend", "engine")
        if any(ca.get(k) != cb.get(k) for k in fixed):
            continue
        if (ca["weight_precision"], ca["kv_cache_dtype"]) == \
           (cb["weight_precision"], cb["kv_cache_dtype"]):
            continue
        res = mcnemar(per_cell[cid_a], per_cell[cid_b])
        res.update({
            "baseline": f"{ca['mechanism']}|{ca['weight_precision']}|"
                        f"{ca['kv_cache_dtype']}|{ca.get('attn_backend','auto')}",
            "candidate": f"{cb['mechanism']}|{cb['weight_precision']}|"
                         f"{cb['kv_cache_dtype']}|{cb.get('attn_backend','auto')}",
        })
        rows.append(res)
    return rows


def format_report(rows: list[dict]) -> str:
    if not rows:
        return ("No quality measurements found. Run the probe with --quality "
                "to populate results/quality/.")
    L = [f"{'baseline':34}{'candidate':34}{'base':>6}{'cand':>6}"
         f"{'disc':>6}{'p':>8}  verdict", "-" * 104]
    for r in rows:
        verdict = ("DIFFERS" if r["significant_at_05"]
                   else "no detected difference")
        L.append(f"{r['baseline']:34}{r['candidate']:34}"
                 f"{r['acc_baseline']:>6}{r['acc_candidate']:>6}"
                 f"{r['discordant']:>6}{r['p_value']:>8.3f}  {verdict}")
    L.append("\n`base`/`cand` are counts correct out of `n_compared`. "
             "`disc` is the number of problems the two disagree on — only "
             "those carry information.")
    L.append("A non-significant result is not evidence of equivalence; with "
             "few discordant pairs the test simply lacks power to say.")
    return "\n".join(L)
