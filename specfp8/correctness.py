"""Correctness testing — three layered tests per design §4.

Test 1 — Speculation equivalence: spec vs non-spec at matched precision.
         The sharp instrument for detecting H1's missing dequant.
Test 2 — Quantization damage: FP8 vs BF16 non-spec (expected, measured).
Test 3 — Garbage backstop: catches the case where both paths are broken
         identically and Test 1 passes on matching nonsense.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


# Thresholds
EXACT_MATCH_THRESHOLD = 0.95    # Test 1: ≥95% of 32 prompts must match
GSM8K_FLOOR_CORRECT = 0.70     # Test 3: ≥70% = correct for a 4B model
GSM8K_FLOOR_DEGRADED = 0.50    # Test 3: ≥50% = degraded; <50% = broken
REPETITION_NGRAM = 4            # Test 3: 4-gram
REPETITION_THRESHOLD = 5        # Test 3: repeating ≥5 times = degenerate


class CorrectnessResult(BaseModel):
    """Aggregated correctness verdict for a cell."""

    # Test 1: speculation equivalence
    exact_match_rate: float         # fraction of prompts with identical output
    mean_prefix_agreement: float    # mean tokens agreeing before first divergence

    # Test 2: quantization damage
    damage_vs_bf16: float           # token-level disagreement fraction vs BF16

    # Test 3: garbage backstop
    task_accuracy: float | None     # GSM8K accuracy; None if no GSM8K prompts
    degenerate_count: int           # number of outputs flagged as repetitive
    empty_count: int                # number of empty outputs

    verdict: Literal["correct", "degraded", "broken"]


def load_correctness_prompts() -> tuple[list[dict], list[dict]]:
    """Load the fixed 32 prompts: 16 MT-Bench + 16 GSM8K."""
    prompts_path = Path(__file__).parent / "workloads" / "correctness_prompts.json"
    with open(prompts_path) as f:
        data = json.load(f)
    return data["mtbench"], data["gsm8k"]


def get_all_prompts() -> list[str]:
    """Return the 32 prompt strings in fixed order (MT-Bench then GSM8K)."""
    mt, gsm = load_correctness_prompts()
    return [p["prompt"] for p in mt] + [p["prompt"] for p in gsm]


def get_gsm8k_answers() -> list[int | float]:
    """Return GSM8K reference answers in order."""
    _, gsm = load_correctness_prompts()
    return [p["answer"] for p in gsm]


def test1_equivalence(
    spec_outputs: list[str],
    nonspec_outputs: list[str],
) -> tuple[float, float]:
    """Test 1: compare speculative vs non-speculative outputs at matched precision.

    Returns (exact_match_rate, mean_prefix_agreement).
    """
    assert len(spec_outputs) == len(nonspec_outputs), (
        f"Output count mismatch: {len(spec_outputs)} vs {len(nonspec_outputs)}"
    )

    n = len(spec_outputs)
    exact_matches = 0
    prefix_agreements = []

    for spec, nonspec in zip(spec_outputs, nonspec_outputs):
        if spec == nonspec:
            exact_matches += 1
            # Full match — prefix agreement is 100% of tokens
            prefix_agreements.append(1.0)
        else:
            # Count agreeing tokens before first divergence
            spec_tokens = spec.split()
            nonspec_tokens = nonspec.split()
            agreeing = 0
            for s, ns in zip(spec_tokens, nonspec_tokens):
                if s == ns:
                    agreeing += 1
                else:
                    break
            total = max(len(spec_tokens), len(nonspec_tokens), 1)
            prefix_agreements.append(agreeing / total)

    exact_match_rate = exact_matches / max(n, 1)
    mean_prefix = sum(prefix_agreements) / max(len(prefix_agreements), 1)

    return exact_match_rate, mean_prefix


def test2_damage(
    fp8_outputs: list[str],
    bf16_outputs: list[str],
) -> float:
    """Test 2: quantization damage — token-level disagreement fraction.

    Compares FP8 non-spec vs BF16 non-spec outputs. Returns the fraction
    of tokens that disagree (divergence is expected and measured, not asserted).
    """
    total_tokens = 0
    disagreeing_tokens = 0

    for fp8, bf16 in zip(fp8_outputs, bf16_outputs):
        fp8_tokens = fp8.split()
        bf16_tokens = bf16.split()
        max_len = max(len(fp8_tokens), len(bf16_tokens))

        for i in range(max_len):
            total_tokens += 1
            t1 = fp8_tokens[i] if i < len(fp8_tokens) else ""
            t2 = bf16_tokens[i] if i < len(bf16_tokens) else ""
            if t1 != t2:
                disagreeing_tokens += 1

    return disagreeing_tokens / max(total_tokens, 1)


def test3_garbage(
    outputs: list[str],
    gsm8k_answers: list[int | float] | None = None,
) -> tuple[float | None, int, int]:
    """Test 3: garbage backstop.

    Returns (task_accuracy_or_none, degenerate_count, empty_count).
    """
    empty_count = sum(1 for o in outputs if not o.strip())
    degenerate_count = sum(1 for o in outputs if _is_degenerate(o))

    task_accuracy = None
    if gsm8k_answers is not None:
        # GSM8K outputs are the last 16 of the 32 outputs
        gsm_outputs = outputs[16:]  # last 16 are GSM8K
        correct = 0
        for output, answer in zip(gsm_outputs, gsm8k_answers):
            extracted = extract_gsm8k_answer(output)
            if extracted is not None and abs(extracted - answer) < 0.01:
                correct += 1
        task_accuracy = correct / max(len(gsm8k_answers), 1)

    return task_accuracy, degenerate_count, empty_count


def compute_verdict(
    exact_match_rate: float,
    task_accuracy: float | None,
    degenerate_count: int,
    empty_count: int,
    n_prompts: int = 32,
) -> Literal["correct", "degraded", "broken"]:
    """Determine the correctness verdict from test results."""

    # Broken conditions: any strong signal of failure
    if empty_count > n_prompts * 0.25:
        return "broken"
    if degenerate_count > n_prompts * 0.25:
        return "broken"
    if task_accuracy is not None and task_accuracy < GSM8K_FLOOR_DEGRADED:
        return "broken"

    # Degraded: Test 1 fails threshold or GSM8K below expected
    if exact_match_rate < EXACT_MATCH_THRESHOLD:
        return "degraded"
    if task_accuracy is not None and task_accuracy < GSM8K_FLOOR_CORRECT:
        return "degraded"

    return "correct"


def run_correctness(
    spec_outputs: list[str],
    nonspec_outputs: list[str],
    bf16_nonspec_outputs: list[str] | None = None,
) -> CorrectnessResult:
    """Run all three tests and return the aggregated result.

    Args:
        spec_outputs: outputs from speculative decoding at test precision
        nonspec_outputs: outputs from non-speculative at SAME precision
        bf16_nonspec_outputs: outputs from non-speculative at BF16 (for Test 2)
    """
    # Test 1
    exact_match_rate, mean_prefix = test1_equivalence(spec_outputs, nonspec_outputs)

    # Test 2
    if bf16_nonspec_outputs is not None:
        damage = test2_damage(nonspec_outputs, bf16_nonspec_outputs)
    else:
        damage = -1.0  # not measured

    # Test 3
    gsm_answers = get_gsm8k_answers()
    task_accuracy, degenerate_count, empty_count = test3_garbage(
        spec_outputs, gsm_answers
    )

    verdict = compute_verdict(
        exact_match_rate, task_accuracy, degenerate_count, empty_count
    )

    return CorrectnessResult(
        exact_match_rate=exact_match_rate,
        mean_prefix_agreement=mean_prefix,
        damage_vs_bf16=damage,
        task_accuracy=task_accuracy,
        degenerate_count=degenerate_count,
        empty_count=empty_count,
        verdict=verdict,
    )


def _is_degenerate(text: str) -> bool:
    """Check if text contains degenerate repetition (any 4-gram repeating ≥5×)."""
    if not text.strip():
        return False

    words = text.split()
    if len(words) < REPETITION_NGRAM:
        return False

    ngram_counts: dict[tuple, int] = {}
    for i in range(len(words) - REPETITION_NGRAM + 1):
        ngram = tuple(words[i : i + REPETITION_NGRAM])
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        if ngram_counts[ngram] >= REPETITION_THRESHOLD:
            return True

    return False


def strip_reasoning(text: str) -> str:
    """Drop reasoning-model <think> blocks, leaving only the final answer.

    Without this the "last number in text" fallback happily reads a number
    out of the middle of a reasoning trace and scores it as the answer.

    An unterminated <think> means the model was still reasoning when it hit
    the token limit and never stated an answer, so this returns empty rather
    than guessing — the honest outcome is "no answer", not a number lifted
    from the trace.
    """
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    if "<think>" in text:
        return ""
    return text


def extract_gsm8k_answer(text: str) -> float | None:
    """Extract the final numeric answer from a GSM8K-style response.

    Looks for patterns like "#### 42", "The answer is 42", or the last
    number in the text. Reasoning blocks are removed first.
    """
    text = strip_reasoning(text)
    if not text.strip():
        return None

    # Pattern 1: "#### <number>"
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return _parse_number(match.group(1))

    # Pattern 2: "the answer is <number>"
    match = re.search(
        r"(?:the\s+answer\s+is|therefore|thus|so)\s*:?\s*\$?(-?[\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if match:
        return _parse_number(match.group(1))

    # Pattern 3: last number in text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return _parse_number(numbers[-1])

    return None


def _parse_number(s: str) -> float | None:
    """Parse a number string, handling commas."""
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None
