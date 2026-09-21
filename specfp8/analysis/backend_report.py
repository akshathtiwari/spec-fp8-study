"""Which attention backend actually served each cell.

H1 is a claim about FlashInfer specifically, so a cell that ran with
`attn_backend: auto` only bears on H1 if vLLM happened to select FlashInfer.
The engine announces its choice during startup, but that line lives in the
server log — which is why `store.persist_log` exists. This reads those logs
back and reports the selected backend per cell.

The exact wording of the announcement is not stable across engine versions,
so this matches a family of patterns and, when none hit, falls back to
returning the candidate lines verbatim rather than silently reporting
"unknown". A wrong backend attribution would misassign the central result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

#: Ordered most-specific first. Each must capture the backend name in group 1.
_PATTERNS = [
    re.compile(r"Using\s+([A-Za-z0-9_]+)\s+backend", re.IGNORECASE),
    re.compile(r"Using\s+backend\s+([A-Za-z0-9_]+)", re.IGNORECASE),
    re.compile(r"attention[_ ]backend[\"'\s:=]+([A-Za-z0-9_]+)", re.IGNORECASE),
    re.compile(r"AttentionBackendEnum\.([A-Za-z0-9_]+)"),
    re.compile(r"backend\s*=\s*['\"]?([A-Za-z0-9_]+)", re.IGNORECASE),
]

#: Lines worth showing a human when nothing matches.
_CANDIDATE = re.compile(r"backend|flashinfer|attention", re.IGNORECASE)

#: Valid attention backend names, from AttentionBackendEnum on vLLM 0.29.0.
#: A capture is only trusted if it is one of these. Without this guard the
#: patterns happily matched `AG_RS` — an all-gather/reduce-scatter comms
#: setting, not an attention backend — and reported it as the backend that
#: served, which is worse than reporting nothing: a confident wrong
#: attribution would misassign the study's central result.
_VALID = {
    "AMX_MLA", "CPU_ATTN", "CPU_MLA", "CUSTOM", "CUTLASS_MLA", "CUTLASS_MSA",
    "FLASHINFER", "FLASHINFER_MLA", "FLASHINFER_MLA_SPARSE",
    "FLASHINFER_MLA_SPARSE_DSV4", "FLASHINFER_MLA_SPARSE_SM120", "FLASHMLA",
    "FLASHMLA_SPARSE", "FLASHMLA_SPARSE_DSV4", "FLASH_ATTN",
    "FLASH_ATTN_DIFFKV", "FLASH_ATTN_MLA", "FLASH_ATTN_MLA_SPARSE",
    "FLEX_ATTENTION", "HPC_ATTN", "MINIMAX_M3_SPARSE", "NO_ATTENTION",
    "ROCM_AITER_FA", "ROCM_AITER_MLA", "ROCM_AITER_MLA_SPARSE",
    "ROCM_AITER_TRITON_MLA", "ROCM_AITER_UNIFIED_ATTN", "ROCM_ATTN",
    "ROCM_FLASHMLA_SPARSE_DSV4", "TOKENSPEED_MLA", "TORCH_SDPA",
    "TRITON_ATTN", "TRITON_ATTN_DIFFKV", "TRITON_MLA", "TRITON_MSA",
    "TURBOQUANT", "XPU_MLA_SPARSE",
}


def backend_from_log(log_path: str | Path) -> tuple[str | None, list[str]]:
    """Return (backend_name_or_None, candidate_lines)."""
    path = Path(log_path)
    if not path.is_file():
        return None, []

    candidates: list[str] = []
    found: str | None = None
    with open(path, errors="replace") as f:
        for line in f:
            if not _CANDIDATE.search(line):
                continue
            stripped = line.rstrip()
            if len(candidates) < 60:
                candidates.append(stripped[:240])
            if found is None:
                for pat in _PATTERNS:
                    m = pat.search(stripped)
                    if m:
                        name = m.group(1).upper()
                        # Only accept a real backend name. Anything else is a
                        # coincidental match on an unrelated setting.
                        if name in _VALID:
                            found = name
                            break
    return found, candidates


def analyse(results_dir: str | Path) -> list[dict]:
    """One row per cell: config, status, and the backend that served it."""
    results_dir = Path(results_dir)
    cells_path = results_dir / "cells.jsonl"
    latest: dict[str, dict] = {}
    if cells_path.exists():
        with open(cells_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "cell_id" in rec:
                    latest[rec["cell_id"]] = rec

    rows = []
    for cid, rec in latest.items():
        cfg = rec["config"]
        backend, candidates = backend_from_log(results_dir / "logs" / f"{cid}.log")
        rows.append({
            "cell_id": cid,
            "mechanism": cfg["mechanism"],
            "weight_precision": cfg["weight_precision"],
            "kv_cache_dtype": cfg["kv_cache_dtype"],
            "requested_backend": cfg.get("attn_backend", "auto"),
            "actual_backend": backend,
            "status": rec["outcome"]["status"],
            "tau": (rec.get("spec") or {}).get("tau"),
            "kv_tokens": (rec.get("capacity") or {}).get("kv_tokens_max"),
            "error": (rec["outcome"].get("error_verbatim") or "")[-400:],
            "candidates": candidates,
        })
    rows.sort(key=lambda r: (r["mechanism"], r["kv_cache_dtype"],
                             r["requested_backend"]))
    return rows


def format_report(rows: list[dict], model_filter: str | None = None) -> str:
    lines = [
        f"{'mech':7}{'kv':10}{'requested':13}{'actual':14}{'status':14}"
        f"{'tau':>7}{'KVtok':>9}",
        "-" * 74,
    ]
    for r in rows:
        lines.append(
            f"{r['mechanism']:7}{r['kv_cache_dtype']:10}{r['requested_backend']:13}"
            f"{str(r['actual_backend']):14}{r['status']:14}"
            f"{r['tau'] if r['tau'] else '-':>7}"
            f"{r['kv_tokens'] if r['kv_tokens'] else '-':>9}"
        )

    failed = [r for r in rows if r["status"] not in ("ok", "broken", "degraded")]
    if failed:
        lines.append("\n### errors on non-serving cells")
        for r in failed:
            lines.append(f"\n-- {r['mechanism']}|{r['kv_cache_dtype']}|"
                         f"{r['requested_backend']} -> {r['status']}")
            for ln in r["error"].split("\n")[-8:]:
                if ln.strip():
                    lines.append(f"   {ln.strip()[:200]}")

    unknown = [r for r in rows if r["actual_backend"] is None]
    if unknown:
        lines.append("\n### backend not matched — candidate log lines")
        for r in unknown[:3]:
            lines.append(f"\n-- {r['mechanism']}|{r['kv_cache_dtype']}|"
                         f"{r['requested_backend']}")
            for ln in r["candidates"][:12]:
                lines.append(f"   {ln}")
    return "\n".join(lines)
