"""Prepare a review-copy .tex whose citations survive conversion to .docx.

pandoc drops `\\cite{...}` when no bibliography processor is running, and
this paper uses an inline `thebibliography` rather than a `.bib`. The result
is prose with the citations silently removed --- "expensive verification ."
--- which makes exactly the claims a reviewer most needs to check
uncheckable.

Three transformations, all reversible and none touching `paper.tex`:

1. `\\cite{a,b}` becomes a literal `[1, 2]`, numbered in `\\bibitem` order
   so the markers match the reference list.
2. A `References` heading is inserted, because `thebibliography` emits one
   in LaTeX and pandoc does not.
3. The figure is swapped to PNG and the title's forced break removed.

The LaTeX stays canonical; this is a derived view, regenerated on every
build for the same reason `analysis/out` is.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main(src: Path, dst: Path) -> int:
    tex = src.read_text()

    # Bibliography order defines the numbering.
    keys = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    number = {k: i + 1 for i, k in enumerate(keys)}
    if not keys:
        print("warning: no \\bibitem found; citations will be dropped",
              file=sys.stderr)

    def cite(m: re.Match) -> str:
        refs = [k.strip() for k in m.group(1).split(",")]
        nums = [str(number[k]) for k in refs if k in number]
        unknown = [k for k in refs if k not in number]
        for k in unknown:
            print(f"warning: \\cite{{{k}}} has no matching \\bibitem",
                  file=sys.stderr)
        return f"[{', '.join(nums)}]" if nums else ""

    tex, n_cites = re.subn(r"\\cite\{([^}]+)\}", cite, tex)

    # thebibliography prints its own heading in LaTeX; pandoc does not.
    tex = tex.replace(r"\begin{thebibliography}",
                      "\\section*{References}\n\\begin{thebibliography}", 1)

    # Word cannot render an embedded PDF; it shows a blank object.
    tex = tex.replace("figures/bimodality.pdf", "figures/bimodality.png")

    # pandoc joins the title's two lines with no space.
    tex = tex.replace(r"When You\\Benchmark", "When You Benchmark")

    dst.write_text(tex)
    print(f"  {n_cites} citations numbered against {len(keys)} references")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
