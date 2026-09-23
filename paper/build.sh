#!/usr/bin/env bash
# Build the paper, refusing to build one whose figures do not check out.
#
# Order matters. The figure is generated from raw records, then the number
# checker runs against the regenerated tables, and only then does LaTeX run.
# Building first and checking afterwards would produce a PDF that looks
# finished while failing its own audit, which is the failure mode the paper
# is about.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== regenerating figure and its data table =="
.venv/bin/python analysis/fig_bimodality.py

echo "== provenance =="
.venv/bin/python analysis/check_provenance.py || {
  echo "provenance defects above; see docs/data-model.md section 6" >&2
}

echo "== every figure in the paper traces to a record =="
.venv/bin/python analysis/check_paper.py

echo "== typesetting =="
cd paper
./build/tectonic paper.tex
echo "== built paper/paper.pdf =="
ls -la paper.pdf
