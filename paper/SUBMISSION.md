# arXiv submission checklist

**Status: ready to upload; blocked on endorsement.**

## What to upload

`paper/arxiv-submission.tar.gz` — contains `paper.tex` and
`figures/bimodality.pdf`.

arXiv requires **TeX source**, not a PDF: "If your submission is written in
TeX you must upload your source. PDFs produced from TeX may be declined."
The tarball is the source. Rebuild it after any edit:

```bash
./paper/build.sh          # regenerates figure, runs checkers, typesets
cd paper/arxiv-submission && tar czf ../arxiv-submission.tar.gz paper.tex figures/
```

The bibliography is inline (`thebibliography`), so there is no `.bib` and no
`.bbl` to ship.

## Form fields

| Field | Value |
|---|---|
| Primary category | **cs.LG** |
| Cross-list | **cs.PF** (Performance) |
| Title | What You Are Actually Measuring When You Benchmark Speculative Decoding Under FP8 |
| Authors | Akshath Tiwari |
| Comments | Suggested: "Harness, raw records and findings at github.com/akshathtiwari/spec-fp8-study" |
| License | arXiv non-exclusive is fine; CC BY 4.0 matches `LICENSE-DATA` and is preferred |

The abstract must be pasted as **plain text**. Strip the LaTeX: `\emph{}`,
`$\tau$` becomes "tau", `$2.96\times$` becomes "2.96x".

## Order of operations

1. Click **START NEW SUBMISSION**. arXiv will state that an endorsement is
   required and issue a **six-character code** tied to cs.LG.
2. Send that code to an endorser (see `docs/endorsement.md` for candidates
   and a draft email). The code is archive-specific — requesting it for
   cs.LG then switching to cs.PF means getting a new one.
3. Once endorsed, complete the upload.

**Announcement is permanent.** arXiv's own preflight: "All announced content
is archival and cannot be removed." Read the PDF end to end before this
step, not after.

## Known state at time of packaging

- `check_paper.py`: 199 figures, 0 untraceable.
- `check_provenance.py`: 2 defects (F009, F014 rest on probe output predating
  per-measurement recording). Both are stated in the paper's §10; neither is
  cited in it.
- The paper carries one retraction of its own headline (F029, bimodality),
  made four days before packaging.
