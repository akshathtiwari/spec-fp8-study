"""Cross-engine comparison (R2).

Takes results/cells.jsonl, groups cells by config (minus engine field),
pairs SGLang vs vLLM verdicts. Flags disagreements — agreement is evidence
the gap is real; disagreement is itself reportable.

Usage:
    python -m specfp8.analysis.cross_engine --results results/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def config_key(config: dict) -> str:
    """Create a comparison key from config, excluding engine-specific fields."""
    # Remove engine and engine_ref — we're comparing across these
    key_fields = {
        k: v for k, v in sorted(config.items())
        if k not in ("engine", "engine_ref", "attn_backend")
    }
    return json.dumps(key_fields, sort_keys=True, separators=(",", ":"))


def load_results(results_dir: str | Path) -> list[dict]:
    """Load all records from cells.jsonl."""
    cells_path = Path(results_dir) / "cells.jsonl"
    records = []
    with open(cells_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compare(results_dir: str | Path) -> list[dict]:
    """Compare verdicts across engines for matching configurations.

    Returns a list of comparison records, each containing:
    - config_key: the shared configuration
    - vllm_status: vLLM's verdict (or "missing")
    - sglang_status: SGLang's verdict (or "missing")
    - agreement: True if both match
    - tau_vllm: τ from vLLM (if available)
    - tau_sglang: τ from SGLang (if available)
    """
    records = load_results(results_dir)

    # Group by config key and engine
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in records:
        config = rec.get("config", {})
        engine = config.get("engine", "unknown")
        key = config_key(config)
        groups[key][engine] = rec

    comparisons = []
    for key, engines in sorted(groups.items()):
        vllm_rec = engines.get("vllm")
        sglang_rec = engines.get("sglang")

        vllm_status = vllm_rec["outcome"]["status"] if vllm_rec else "missing"
        sglang_status = sglang_rec["outcome"]["status"] if sglang_rec else "missing"

        vllm_tau = None
        sglang_tau = None
        if vllm_rec and "spec" in vllm_rec:
            vllm_tau = vllm_rec["spec"].get("tau")
        if sglang_rec and "spec" in sglang_rec:
            sglang_tau = sglang_rec["spec"].get("tau")

        # Parse config for display
        config_dict = json.loads(key)

        comparisons.append({
            "mechanism": config_dict.get("mechanism", "?"),
            "weight_precision": config_dict.get("weight_precision", "?"),
            "kv_cache_dtype": config_dict.get("kv_cache_dtype", "?"),
            "vllm_status": vllm_status,
            "sglang_status": sglang_status,
            "agreement": vllm_status == sglang_status,
            "tau_vllm": vllm_tau,
            "tau_sglang": sglang_tau,
        })

    return comparisons


def print_comparison_table(comparisons: list[dict]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'Mechanism':<10} {'Weight':<6} {'KV':<12} {'vLLM':<20} {'SGLang':<20} {'Match'}")
    print("-" * 80)

    disagreements = 0
    for c in comparisons:
        match_str = "✓" if c["agreement"] else "✗ DISAGREE"
        if not c["agreement"]:
            disagreements += 1

        vllm_str = c["vllm_status"]
        if c["tau_vllm"] is not None:
            vllm_str += f" (τ={c['tau_vllm']:.2f})"

        sglang_str = c["sglang_status"]
        if c["tau_sglang"] is not None:
            sglang_str += f" (τ={c['tau_sglang']:.2f})"

        print(
            f"{c['mechanism']:<10} {c['weight_precision']:<6} "
            f"{c['kv_cache_dtype']:<12} {vllm_str:<20} {sglang_str:<20} {match_str}"
        )

    print(f"\nTotal configs: {len(comparisons)}, Disagreements: {disagreements}")


def main():
    parser = argparse.ArgumentParser(description="Cross-engine comparison (R2)")
    parser.add_argument("--results", default="results", help="Results directory")
    args = parser.parse_args()

    comparisons = compare(args.results)
    print_comparison_table(comparisons)

    # Write machine-readable output
    out_path = Path(args.results) / "cross_engine.json"
    with open(out_path, "w") as f:
        json.dump(comparisons, f, indent=2)
    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
