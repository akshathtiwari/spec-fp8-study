"""Verify the provenance chain described in docs/data-model.md section 5.

That document states that a paper claim must resolve down to bytes:

    findings/F0xx -> analysis/out/tables/* -> results/cells.jsonl -> results/logs/*

and that "any break in that chain is a defect". Nothing enforced it. A
finding could cite a table that a later refactor renamed, or a cell_id that
no run ever produced, and the break would surface only when a reviewer
followed the link and found nothing -- which is the worst possible time.

Checks, in the order a reader would follow them:

1. Every `analysis/out/...` path cited by a finding exists.
2. Every cell_id cited by a finding appears in results/cells.jsonl.
3. Every cell that a finding leans on has its engine log persisted, since
   backend selection and verbatim failure text live only there (F012, F017).
4. Frontmatter is well-formed and uses declared vocabulary, so the corpus
   can be inventoried without hand-reading 21 files.
5. supersedes/superseded_by point at findings that exist and agree with
   each other -- a one-way retraction link is how a retracted claim keeps
   getting cited.

Exit code is the number of defects, so this can gate a commit.

    python analysis/check_provenance.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINDINGS = ROOT / "findings"
CELLS = ROOT / "results" / "cells.jsonl"
LOGS = ROOT / "results" / "logs"

KINDS = {"result", "method", "retraction", "gap"}
STATUSES = {"established", "provisional", "retracted", "superseded"}
CONFIDENCES = {"high", "medium", "low"}

_CELL_ID = re.compile(r"\b[0-9a-f]{16}\b")
_ANALYSIS_PATH = re.compile(r"analysis/out/[\w./-]+\.(?:md|csv|json)")


def _frontmatter(text: str) -> str:
    parts = text.split("---")
    return parts[1] if len(parts) >= 3 else ""


def _field(fm: str, key: str) -> str | None:
    m = re.search(rf"^{key}:\s*(.+)$", fm, re.M)
    return m.group(1).strip() if m else None


def _list_field(fm: str, key: str) -> list[str]:
    """Read a `key: [a, b]` list, tolerating line wrapping."""
    m = re.search(rf"^\s*{key}:\s*\[(.*?)\]", fm, re.M | re.S)
    if not m:
        return []
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


def known_cell_ids() -> set[str]:
    ids: set[str] = set()
    if not CELLS.exists():
        return ids
    for line in CELLS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(json.loads(line)["cell_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def main() -> int:
    defects: list[str] = []
    cells = known_cell_ids()
    finding_ids: dict[str, Path] = {}
    supersedes: dict[str, list[str]] = {}
    superseded_by: dict[str, list[str]] = {}

    files = sorted(FINDINGS.glob("F*.md"))
    if not files:
        print("no findings found", file=sys.stderr)
        return 1

    for path in files:
        text = path.read_text()
        fm = _frontmatter(text)
        name = path.name

        fid = _field(fm, "id")
        if not fid:
            defects.append(f"{name}: no id in frontmatter")
            continue
        if fid in finding_ids:
            defects.append(f"{name}: duplicate id {fid} "
                           f"(also {finding_ids[fid].name})")
        finding_ids[fid] = path

        if not name.startswith(fid + "-"):
            defects.append(f"{name}: filename does not start with id {fid}")

        for key, vocab in (("kind", KINDS), ("status", STATUSES),
                           ("confidence", CONFIDENCES)):
            val = _field(fm, key)
            if val is None:
                defects.append(f"{fid}: missing {key}")
            elif val not in vocab:
                defects.append(f"{fid}: {key}={val!r} not in {sorted(vocab)}")

        # data-model.md rule 2: the boundary is part of the claim.
        #
        # Retractions are exempt. They withdraw a claim rather than make
        # one, so the section has nothing to say; what they owe the reader
        # instead is what was believed and what caught it, which is the
        # body of a retraction rather than a named section. Applying the
        # rule to them would train the habit of writing the heading to
        # satisfy a checker, which is worse than not checking.
        if _field(fm, "kind") != "retraction" \
                and "## What this does NOT establish" not in text:
            defects.append(f"{fid}: missing 'What this does NOT establish'")

        # 1. cited derived outputs exist
        for rel in sorted(set(_ANALYSIS_PATH.findall(text))):
            if not (ROOT / rel).exists():
                defects.append(f"{fid}: cites missing {rel}")

        # 2 + 3. cited cells exist, and their logs were persisted
        for cid in _list_field(fm, "cells"):
            cid = cid.strip("'\"")
            if not _CELL_ID.fullmatch(cid):
                defects.append(f"{fid}: malformed cell id {cid!r}")
                continue
            if cid not in cells:
                defects.append(f"{fid}: cites cell {cid} absent from cells.jsonl")
            elif not (LOGS / f"{cid}.log").exists():
                defects.append(f"{fid}: cell {cid} has no persisted engine log")

        supersedes[fid] = [x.strip("'\"") for x in _list_field(fm, "supersedes")]
        superseded_by[fid] = [x.strip("'\"")
                              for x in _list_field(fm, "superseded_by")]

    # 5. retraction links are symmetric and point somewhere real
    for fid, targets in supersedes.items():
        for t in targets:
            if t not in finding_ids:
                defects.append(f"{fid}: supersedes unknown finding {t}")
            elif fid not in superseded_by.get(t, []):
                defects.append(
                    f"{fid} supersedes {t}, but {t} does not list "
                    f"superseded_by: [{fid}] -- a one-way link lets a "
                    f"retracted claim keep being cited")

    print(f"findings checked : {len(finding_ids)}")
    print(f"cell_ids on file : {len(cells)}")
    print(f"defects          : {len(defects)}")
    for d in defects:
        print(f"  - {d}")
    return len(defects)


if __name__ == "__main__":
    sys.exit(main())
