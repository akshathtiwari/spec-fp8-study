"""Fail if the committed PDF is older than the sources it is built from.

The PDF is a build artifact that is also committed, so it can silently fall
behind `paper.tex`. That happened: source corrections were committed and the
PDF rebuilt afterwards, so the commit carried a paper whose text differed
from its own source. External review caught it by diffing the two.

This is the same shape as the checked-in arXiv tarball that drifted behind
`paper.tex`, and as `analysis/out` before it was regenerated on every build.
A derived file that can go stale will, unless something checks.

    python analysis/check_pdf_fresh.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "paper" / "paper.pdf"
SOURCES = [ROOT / "paper" / "paper.tex",
           ROOT / "paper" / "figures" / "bimodality.pdf"]


def main() -> int:
    if not PDF.exists():
        print("paper/paper.pdf missing; run paper/build.sh", file=sys.stderr)
        return 1
    pdf_mtime = PDF.stat().st_mtime
    stale = [s for s in SOURCES if s.exists() and s.stat().st_mtime > pdf_mtime]
    if stale:
        print("paper/paper.pdf is STALE. Newer than the PDF:", file=sys.stderr)
        for s in stale:
            print(f"  - {s.relative_to(ROOT)}", file=sys.stderr)
        print("Run ./paper/build.sh before committing.", file=sys.stderr)
        return len(stale)
    print(f"paper.pdf is current against {len(SOURCES)} source(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
