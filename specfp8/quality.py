"""Powered task-accuracy measurement.

Separate from the correctness gate in `correctness.py`, and deliberately so.
The gate runs 32 prompts at concurrency 1 against every cell: it is cheap, and
concurrency 1 is required because it feeds exact-match comparisons, which are
sensitive to batching (see findings/F009).

This measures answer quality, which is a different job with different
requirements. It needs statistical power the gate cannot afford — at n=16 the
95% interval is +/-24 points, wide enough to hide any realistic FP8 effect
(findings/F008) — and it does *not* need concurrency 1, because whether a model
reaches the right answer is robust to the small numeric differences batching
introduces. So it runs many more problems at high concurrency, which costs
minutes rather than the hours concurrency 1 would.

Per-item correctness is stored, not just the aggregate, because the comparison
that matters is paired: the same problems are put to every configuration, so
FP8 damage is detected by McNemar on the disagreements rather than by
differencing two independent rates.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path

from specfp8.client import load_run
from specfp8.correctness import extract_gsm8k_answer

PROMPTS = Path(__file__).parent / "workloads" / "quality_prompts.json"

#: The quality measurement is pinned to the FIRST 256 problems, even though
#: the file now holds the full 1319-problem test split. Phase 2 needed a
#: bigger pool so its repeats could draw disjoint prompt sets, and growing the
#: file would otherwise have silently changed what findings/F016 measured --
#: the same class of drift the prompt-set fingerprint exists to catch. The
#: file is in dataset order from offset 0, so the first 256 are byte-identical
#: to the set F016 used.
QUALITY_N = 256

#: The quality stage overrides the gate's token budget rather than inheriting
#: it. Measured on Qwen3-4B with thinking disabled, 27% of the gate's GSM8K
#: generations hit a 256-token cap mid-reasoning, and the rate is higher on the
#: harder problems in this set. A truncated generation is scored as no answer,
#: so at that budget the measurement reports how often the model runs out of
#: tokens rather than how often it is right — and if a precision change alters
#: verbosity, that confound lands directly on the comparison it is meant to
#: make. 768 leaves headroom for the verbose markdown reasoning this model
#: produces. Raise if `unparseable` stays high; it is a measurement artifact,
#: never a result.
QUALITY_MAX_TOKENS = 768


def load_quality_prompts(limit: int | None = QUALITY_N
                         ) -> tuple[list[str], list[float]]:
    """Return (prompts, reference answers) in pinned order.

    Defaults to the first `QUALITY_N`, which is what the quality measurement
    scores. Pass `limit=None` for the whole pool, which is what Phase 2
    workloads draw from.
    """
    data = json.load(open(PROMPTS))
    rows = data["gsm8k"]
    if limit is not None:
        rows = rows[:limit]
    return [r["prompt"] for r in rows], [r["answer"] for r in rows]


def quality_fingerprint() -> str:
    """Hash of the quality prompt set, so a changed set invalidates results."""
    prompts, answers = load_quality_prompts()
    payload = json.dumps(list(zip(prompts, answers)), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sensibly at extreme proportions, where accuracy measurements
    plausibly land if a configuration is badly damaged.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def score(outputs: list[str], answers: list[float]) -> list[bool | None]:
    """Per-item correctness. None where no answer could be extracted.

    None is distinct from False on purpose: a truncated or unparseable
    generation is a different failure from a confidently wrong answer, and
    collapsing them would hide a model that has stopped producing answers at
    all behind an ordinary-looking accuracy drop.
    """
    out: list[bool | None] = []
    for text, want in zip(outputs, answers):
        got = extract_gsm8k_answer(text)
        out.append(None if got is None else abs(got - want) < 0.01)
    return out


def run_quality(
    base_url: str,
    model: str,
    sampling_params: dict,
    concurrency: int = 32,
    timeout_s: float = 300.0,
    limit: int | None = None,
) -> dict:
    """Run the quality set against an already-booted server."""
    # Own token budget: see QUALITY_MAX_TOKENS.
    sampling_params = {**sampling_params, "max_tokens": QUALITY_MAX_TOKENS}

    prompts, answers = load_quality_prompts()
    if limit:
        prompts, answers = prompts[:limit], answers[:limit]

    results = asyncio.run(
        load_run(base_url, model, prompts, sampling_params,
                 concurrency=concurrency, n_warmup=3, timeout_s=timeout_s)
    )
    outputs = [r.output_text if r.ok else "" for r in results]
    per_item = score(outputs, answers)

    n = len(per_item)
    correct = sum(1 for v in per_item if v is True)
    unparseable = sum(1 for v in per_item if v is None)
    failed = sum(1 for r in results if not r.ok)
    lo, hi = wilson_ci(correct, n)

    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "ci95": [round(lo, 4), round(hi, 4)],
        "ci95_halfwidth": round((hi - lo) / 2, 4),
        "unparseable": unparseable,
        "request_failures": failed,
        "concurrency": concurrency,
        "max_tokens": QUALITY_MAX_TOKENS,
        "truncated": sum(
            1 for r in results if r.output_tokens >= QUALITY_MAX_TOKENS - 1
        ),
        "prompt_set_fingerprint": quality_fingerprint(),
        "per_item": [None if v is None else bool(v) for v in per_item],
        "records": [
            {
                "index": i,
                "ok": r.ok,
                "correct": per_item[i],
                "output_tokens": r.output_tokens,
                "e2e_ms": round(r.e2e_ms, 3),
                "output_text": r.output_text,
            }
            for i, r in enumerate(results)
        ],
    }
