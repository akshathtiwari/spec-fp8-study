"""Verify the provenance chain described in docs/data-model.md section 5.

That document states that a paper claim must resolve down to bytes:

    findings/F0xx -> analysis/out/tables/* -> results/cells.jsonl -> results/logs/*

and that "any break in that chain is a defect". Nothing enforced it. A
finding could cite a table that a later refactor renamed, or a cell_id that
no run ever produced, and the break would surface only when a reviewer
followed the link and found nothing -- which is the worst possible time.

Checks, in the order a reader would follow them:

1. Every `analysis/out/...` path cited by a finding exists.
2. Every cell_id cited by a finding appears in some record file in
   results/ (cells, sweep, probes, or boot_failures).
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
    """Cell ids appearing in ANY raw record file, not just cells.jsonl.

    Phase 1 compatibility records live in cells.jsonl, but a cell can also
    be evidenced by a Phase 2 measurement (sweep.jsonl), a bespoke probe
    (probes.jsonl), or a boot that never became healthy
    (boot_failures.jsonl). F023's entire claim is that a cell does NOT
    boot, so its only possible evidence is a failure record; scanning
    cells.jsonl alone would reject the finding for citing the one file that
    could substantiate it.
    """
    ids: set[str] = set()
    for path in sorted((ROOT / "results").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cid = json.loads(line).get("cell_id")
            except json.JSONDecodeError:
                continue
            if cid:
                ids.add(cid)
    return ids


def _id_stability_defects() -> list[str]:
    """Recompute every stored cell_id from its stored config.

    A mismatch means the id rule changed under the existing records. The
    fix is never to rewrite results/ -- it is append-only -- but to make the
    new field post-v1 so cells that do not set it keep their ids. See
    `_ID_SCHEMA_V1` in specfp8/cells.py.
    """
    sys.path.insert(0, str(ROOT))
    try:
        from specfp8.cells import ServerCell, cell_id
    except ImportError as e:
        return [f"cannot import specfp8.cells to verify ids: {e}"]

    if not CELLS.exists():
        return []

    out: list[str] = []
    checked: set[str] = set()
    for line in CELLS.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            stored, config = rec["cell_id"], rec["config"]
        except (json.JSONDecodeError, KeyError):
            continue
        if stored in checked:
            continue
        checked.add(stored)
        try:
            recomputed = cell_id(ServerCell(**config))
        except Exception as e:
            out.append(f"cell {stored}: config no longer parses as "
                       f"ServerCell ({type(e).__name__}: {e})")
            continue
        if recomputed != stored:
            out.append(f"cell {stored}: id rule changed, config now hashes "
                       f"to {recomputed} -- {len(checked)} stored results "
                       f"would be orphaned")
    return out


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
                defects.append(f"{fid}: cites cell {cid}, absent from every "
                               f"record file in results/")
            elif not (LOGS / f"{cid}.log").exists():
                defects.append(f"{fid}: cell {cid} has no persisted engine log")

        # 7. data-model.md section 4, rule 4: findings never cite terminal
        # output. A result or method claim has to point at something in
        # results/ or analysis/out/, or it cannot be checked by a reader --
        # and the earlier rules are silent about it, because they validate
        # citations that exist rather than noticing their absence.
        #
        # Retractions and gaps are exempt: a withdrawal's evidence is the
        # finding it withdraws, and a gap records something not measured.
        if _field(fm, "kind") in ("result", "method"):
            if not _list_field(fm, "cells") and not _list_field(fm, "analysis"):
                defects.append(
                    f"{fid}: no evidence in results/ or analysis/out/ -- "
                    f"cites neither a cell nor an analysis output, so the "
                    f"claim rests on terminal output only")

        supersedes[fid] = [x.strip("'\"") for x in _list_field(fm, "supersedes")]
        superseded_by[fid] = [x.strip("'\"")
                              for x in _list_field(fm, "superseded_by")]

    # 6. stored ids still recompute from their stored configs
    #
    # cell_id is content-addressed, so the schema and the hashing rule are
    # part of the on-disk format. Adding a field, renaming one, or changing
    # a post-v1 default silently re-partitions every id: old results stop
    # being findable, findings cite cells that no longer resolve, and a
    # sweep re-runs work it already paid for. Nothing else catches it,
    # because every individual piece still looks well-formed.
    defects.extend(_id_stability_defects())

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
