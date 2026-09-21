"""Fetch and pin the GSM8K problems used for the quality measurement.

Committed as a script rather than run once by hand so the prompt set is
regenerable and its provenance is checkable: anyone can re-run this and diff
the result against what is in the repo.

Source: openai/gsm8k, config `main`, split `test` (1319 problems), taken in
dataset order from offset 0. Order is the dataset's own, not shuffled and not
sampled, so "the first N" is unambiguous and stable without needing a seed.

    python -m specfp8.workloads.fetch_gsm8k --n 256
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

API = ("https://datasets-server.huggingface.co/rows"
       "?dataset=openai/gsm8k&config=main&split=test&offset={off}&length={n}")

OUT = Path(__file__).parent / "quality_prompts.json"
GATE = Path(__file__).parent / "correctness_prompts.json"


def parse_answer(answer: str) -> float | None:
    """GSM8K reference answers end with `#### <number>`."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    return int(value) if value == int(value) else value


def fetch(n: int) -> list[dict]:
    rows: list[dict] = []
    off = 0
    while len(rows) < n:
        take = min(100, n - len(rows))  # API caps length at 100
        with urllib.request.urlopen(API.format(off=off, n=take), timeout=60) as r:
            payload = json.load(r)
        batch = payload["rows"]
        if not batch:
            break
        for item in batch:
            row = item["row"]
            ans = parse_answer(row["answer"])
            if ans is None:
                continue  # a problem with no extractable reference is unusable
            rows.append({
                "id": f"gsm8k_test_{item['row_idx']}",
                "source_index": item["row_idx"],
                "prompt": row["question"],
                "answer": ans,
            })
        off += take
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()

    rows = fetch(args.n)

    # Cross-check against the already-pinned gate set. The gate's 16 GSM8K
    # problems were taken from the head of the same split, so a mismatch means
    # either the upstream dataset moved or the gate set was not what it claims.
    gate = json.load(open(GATE))["gsm8k"]
    head = rows[:len(gate)]
    mismatched = [
        (i, g["answer"], h["answer"])
        for i, (g, h) in enumerate(zip(gate, head)) if g["answer"] != h["answer"]
    ]

    payload = {
        "_comment": (
            f"{len(rows)} GSM8K problems for the quality measurement, pinned by "
            "source index. Distinct from correctness_prompts.json, which is the "
            "cheap per-cell gate: this set exists to give the accuracy arm "
            "enough statistical power to detect FP8 damage (see findings/F008). "
            "Never reorder or modify — regenerate with "
            "`python -m specfp8.workloads.fetch_gsm8k`."
        ),
        "source": {
            "dataset": "openai/gsm8k", "config": "main", "split": "test",
            "total_available": 1319,
            "selection": "dataset order from offset 0, no shuffle, no sampling",
            "api": "https://datasets-server.huggingface.co/rows",
        },
        "gate_head_matches": not mismatched,
        "n": len(rows),
        "gsm8k": rows,
    }
    OUT.write_text(json.dumps(payload, indent=1))

    print(f"wrote {OUT.relative_to(Path.cwd())}: {len(rows)} problems")
    print(f"  first 16 match the pinned gate set: {not mismatched}")
    if mismatched:
        for i, a, b in mismatched[:5]:
            print(f"    index {i}: gate={a} fetched={b}")
    answers = [r["answer"] for r in rows]
    print(f"  answer range: {min(answers)} .. {max(answers)}")


if __name__ == "__main__":
    main()
