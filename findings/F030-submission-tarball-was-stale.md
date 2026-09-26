---
id: F030
title: The arXiv submission tarball silently froze two commits behind the paper, because a `cd` inside build.sh made the freshness check die before the packaging step ever ran
kind: method
status: established
confidence: high
date: 2026-09-26
evidence:
  analysis: [paper/build.sh, analysis/check_pdf_fresh.py]
  code_sha: pending
supersedes: []
superseded_by: []
---

## Claim

`paper/build.sh` ended with three steps: typeset, verify the PDF is current
against its sources, then package `arxiv-submission.tar.gz` from the canonical
files. The typesetting step began with a bare `cd paper`. Every path after it
was still written relative to the repository root.

So `.venv/bin/python analysis/check_pdf_fresh.py` resolved to
`paper/.venv/bin/python`, which does not exist. Under `set -euo pipefail` that
exited 127 and **the packaging step never executed**, on any build, for as
long as the bug existed. The script had already printed
`== built paper/paper.pdf ==` and a plausible `ls -la` by then, so a build
that failed two of its own final checks looked like a build that had worked.

The consequence was not cosmetic. `paper/arxiv-submission.tar.gz` is the file
intended for upload to arXiv, and it froze at commit `840c389` while
`paper.tex` advanced to `13284ad`:

```
committed tarball's paper.tex   md5 2fe9a540...
committed paper.tex             md5 83191e8e...
```

The divergence was the HTTP-boundary correction. The paper had been changed,
at the author's explicit instruction, to remove the claim that the harness
measures "what a serving deployment runs" and replace it with wording that
concedes the harness does not reproduce every production environment. That
correction is present in `paper.tex` and in `paper.pdf`. It is **absent from
the tarball**, which still contains the original sentence:

```
$ tar xzOf arxiv-submission.tar.gz paper.tex | grep -c "what a serving deployment runs"
1
```

Had the submission been completed from the committed tarball, the posted
version of the paper would have carried the overclaim that had already been
identified and withdrawn.

## Evidence

| Artifact | State before fix |
|---|---|
| `paper/paper.tex` | current, correction present |
| `paper/paper.pdf` | current, rebuilt each time |
| `paper/arxiv-submission.tar.gz` | **two commits stale, correction absent** |
| `check_pdf_fresh.py` | never ran after the `cd` was introduced |
| packaging step | never ran after the `cd` was introduced |

The fix runs tectonic in a subshell so the working directory stays at the
repository root, and rewrites the packaging paths accordingly.

## Why it happened

This is the third instance in this study of the same shape, and the shape is
now specific enough to name: **a check that reports its own success before it
has finished doing its work.**

1. `build.sh` piped the provenance failure into an `echo` and continued, so
   the build advertised "a paper that fails its own audit is not typeset"
   while typesetting one that failed (external review caught it).
2. `paper/arxiv-submission/paper.tex` was kept as a checked-in copy, drifted
   behind `paper.tex`, and was then audited as though it were the paper.
3. This one: the build printed `built paper/paper.pdf`, which was true, and
   exited before packaging, which was not visible.

Item 2 was fixed by generating the tarball fresh on every build rather than
storing it. That fix was correct and it is the reason this defect is
recoverable — but generating it fresh is worthless if the generating step
does not run, and nothing checked that it had.

The deeper cause is that `set -e` converts a path bug into a silent early
exit. The script's last successful output is indistinguishable from its
intended last output unless the reader knows what should have followed.

## What this does NOT establish

- **That anything was actually submitted incorrectly.** Nothing was uploaded.
  The stale tarball was caught before the arXiv submission was completed,
  because the endorsement gate had not yet cleared. That is luck, not process.
- **That the PDF was ever wrong.** `paper.pdf` was rebuilt on every run and
  was current throughout. Only the tarball was stale.
- **That `check_pdf_fresh.py` is sound.** It was never exercised after the
  `cd` was introduced, so its first real run is the one in this commit. It
  passes, but a checker that has run once has not been tested.
- **That no other script has the same bug.** Only `build.sh` was inspected.
  Any script mixing `cd` with root-relative paths could fail the same way.

## Consequence

Two practices, both narrow:

> **A build step whose output is an artifact must verify the artifact it
> produced, not report that it produced one.**

The packaging step now runs to completion or the build fails, and the next
revision should compare the tarball's `paper.tex` against `paper/paper.tex`
by hash rather than trusting that `cp` ran.

> **Never `cd` inside a script whose later paths are root-relative.** Use a
> subshell, or make every path absolute.

It also argues for extending `check_pdf_fresh.py` to cover the tarball. The
freshness checker exists precisely because a derived artifact had gone stale
once before; it checks the PDF and not the file that actually gets uploaded.
The instrument was pointed at the artifact that had failed, and not at the
artifact class — which is F028's finding, recurring.

## How to reproduce

The bug is fixed, so reproduction requires reintroducing it:

```bash
# in paper/build.sh, replace the subshell with a bare `cd paper`
./paper/build.sh; echo "exit=$?"        # exit=127, no packaging output
```

To verify the current state is sound:

```bash
./paper/build.sh
tar xzOf paper/arxiv-submission.tar.gz paper.tex | md5sum
md5sum < paper/paper.tex                # must match
```
