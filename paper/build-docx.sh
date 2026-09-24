#!/usr/bin/env bash
# Build a review copy of the paper as .docx.
#
# This is for reading and commenting, NOT for submission -- arXiv wants the
# TeX source (see SUBMISSION.md). The LaTeX remains canonical; this is a
# derived view of it, regenerated rather than maintained, for the same
# reason analysis/out is regenerated: two hand-maintained copies of one
# document drift.
#
# Two transformations are needed on the way:
#   - the figure is swapped from PDF to PNG, because Word embeds a PDF as an
#     unrenderable object and the reviewer sees a blank box
#   - the title's forced line break is removed, because pandoc joins the two
#     lines without a space ("...When YouBenchmark...")
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== regenerating figure (PNG needed for Word) =="
.venv/bin/python analysis/fig_bimodality.py >/dev/null

echo "== preparing review source =="
mkdir -p paper/build
.venv/bin/python paper/docx_prep.py paper/paper.tex paper/build/paper-docx.tex

echo "== converting =="
.venv/bin/python - <<'PY'
import pypandoc, os
os.chdir("paper")
pypandoc.convert_file(
    "build/paper-docx.tex", "docx", format="latex",
    outputfile="paper.docx",
    extra_args=["--resource-path=.:build", "--standalone",
                "--toc", "--toc-depth=2"])
print("  wrote paper/paper.docx")
PY

ls -la paper/paper.docx
