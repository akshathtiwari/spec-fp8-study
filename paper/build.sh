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
# This must BLOCK. An earlier version piped the failure into an echo and
# carried on, so the build advertised "a paper that fails its own audit is
# not typeset" while typesetting one that failed. External review caught it.
#
# ALLOWED_PROVENANCE_DEFECTS is the two findings (F009, F014) that rest on
# probe output predating per-measurement recording. They are disclosed in
# the paper's threats section. Any additional defect stops the build.
ALLOWED_PROVENANCE_DEFECTS=${ALLOWED_PROVENANCE_DEFECTS:-2}
set +e
.venv/bin/python analysis/check_provenance.py
defects=$?
set -e
if [ "$defects" -gt "$ALLOWED_PROVENANCE_DEFECTS" ]; then
  echo "BUILD STOPPED: $defects provenance defects, $ALLOWED_PROVENANCE_DEFECTS disclosed." >&2
  echo "Resolve them or update the disclosure in the paper; see docs/data-model.md section 6." >&2
  exit 1
fi

echo "== every numeric claim in the paper traces to a record =="
# check_paper exits with the count of untraceable claims; set -e stops here
# on any nonzero, which is the intended behaviour: zero is the only
# acceptable value.
.venv/bin/python analysis/check_paper.py

echo "== typesetting =="
cd paper
./build/tectonic paper.tex
echo "== built paper/paper.pdf =="
ls -la paper.pdf

echo "== packaging arXiv source =="
# Built fresh from the canonical files every time, never kept as a checked-in
# copy. A previous revision kept paper/arxiv-submission/paper.tex on disk; it
# drifted behind paper.tex and then failed the numeric audit as though the
# paper itself were stale. A copy that can go stale will.
rm -rf build/arxiv && mkdir -p build/arxiv/figures
cp paper.tex build/arxiv/
cp figures/bimodality.pdf build/arxiv/figures/
tar czf arxiv-submission.tar.gz -C build/arxiv paper.tex figures
ls -la arxiv-submission.tar.gz
