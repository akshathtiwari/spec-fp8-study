"""Verify every measured number in the paper traces to a finding or a table.

The paper's argument is that benchmark numbers are routinely reported
without their error terms. It would be embarrassing, and fatal to the
argument, for one of its own figures to be a transcription error or a
number remembered rather than looked up.

So every distinctive numeric token in `paper/` must appear somewhere in
`findings/` or `analysis/out/`. Distinctive means it has a decimal point or
three-plus significant digits -- enough to be a measurement rather than
"three confounds" or "two of four backends", which are prose and are
checked by reading.

This catches the realistic failure: a number that was correct when written,
in a finding that has since been amended. The 1.32% noise floor changed
F006's conclusions after the paper text was drafted; a stale copy would
have gone unnoticed by every other check in this repo.

**What it does not catch, stated precisely.** This is a membership test
against the whole evidence corpus, so a wrong figure that happens to match
an unrelated number elsewhere passes. Tested: replacing the 1.32% noise
floor with a fabricated 1.47% is NOT caught, because 1.47 occurs in F006
and in two generated tables for unrelated reasons. A transcription slip to
4.1699 IS caught, because that token occurs nowhere.

So the honest claim is: **no figure in the paper is invented out of
nothing.** It is not "every figure is correct", and it is not "every figure
comes from the finding it cites". The stronger check -- resolving each
figure against the specific finding cited in its section -- needs per-claim
provenance markers in the paper text, which do not exist yet. Until they
do, this catches the careless error and not the plausible one.

Exit code is the number of untraceable figures.

    python analysis/check_paper.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
SOURCES = [ROOT / "findings", ROOT / "analysis" / "out", ROOT / "docs"]

#: Structural numbers that are not measurements: section refs, years,
#: arXiv ids, version pins, hardware names, and the finding ids themselves.
_SKIP_CONTEXT = re.compile(
    r"(?:§\s*\d+|F\d{3}|\b20\d{2}\b|\b\d{4}\.\d{4,5}\b|vLLM\s*0\.\d+\.\d+"
    r"|CUDA\s*\d+|SM\d+|rev\s*\d+|Qwen3-4B|#\d+)", re.I)

#: A measurement: has a decimal point, or is a long integer.
_NUMBER = re.compile(r"(?<![\w.])(\d+\.\d+|\d{3,})(?![\w.])")


def corpus() -> set[str]:
    """Every numeric TOKEN in the evidence, as a set.

    Tokens, not a concatenated string. Substring matching looked fine and
    silently passed wrong figures: "1.47" is a substring of "21.470" and of
    "1.472", both of which occur in the goodput CSV, so a fabricated noise
    floor of 1.47% verified against numbers that have nothing to do with it.
    A negative control caught this; reading the code did not.
    """
    tokens: set[str] = set()
    for root in SOURCES:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not (p.is_file() and p.suffix in (".md", ".csv", ".json", ".py")):
                continue
            try:
                text = p.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            # Trailing unit characters are allowed: findings write ratios as
            # "1.64x" with an ASCII x, which IS a word character, so a
            # (?![\w.]) lookahead harvested none of them and every ratio in
            # the paper read as untraceable. The paper writes "1.64×" with
            # U+00D7, which is not a word character -- so the two sides
            # tokenised differently and the mismatch looked like 21 errors.
            # Reject only a following digit or dot, which is what actually
            # signals a longer number.
            for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\d.])", text):
                tok = m.group(1)
                tokens.add(tok)
                # Stored as 2.380 but written 2.38, or vice versa.
                if "." in tok:
                    tokens.add(tok.rstrip("0").rstrip("."))
    return tokens


def main() -> int:
    if not PAPER.exists():
        print("no paper/ directory yet", file=sys.stderr)
        return 0

    haystack = corpus()
    missing: list[tuple[str, int, str, str]] = []
    checked = 0

    for doc in sorted(PAPER.rglob("*.md")):
        for lineno, line in enumerate(doc.read_text().splitlines(), 1):
            # Blank out structural tokens so their digits are not harvested.
            scrubbed = _SKIP_CONTEXT.sub(" ", line)
            for m in _NUMBER.finditer(scrubbed):
                tok = m.group(1)
                checked += 1
                stem = tok.rstrip("0").rstrip(".") if "." in tok else tok
                if tok in haystack or stem in haystack:
                    continue
                missing.append((doc.name, lineno, tok, line.strip()[:72]))

    print(f"paper files : {len(list(PAPER.rglob('*.md')))}")
    print(f"figures     : {checked}")
    print(f"untraceable : {len(missing)}")
    for name, lineno, tok, ctx in missing:
        print(f"  - {name}:{lineno}  {tok!r}  in: {ctx}")
    if missing:
        print("\nEach must appear in findings/ or analysis/out/. If a figure "
              "is right but absent, the finding behind it was never written.")
    return len(missing)


if __name__ == "__main__":
    sys.exit(main())
