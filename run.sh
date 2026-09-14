#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# Model cache — set once, used by launchers
export SPECFP8_MODEL_CACHE="${SPECFP8_MODEL_CACHE:-${HOME}/.cache/huggingface/hub}"

usage() {
    echo "Usage: ./run.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  probe    Run Phase 1 compatibility matrix"
    echo "  sweep    Run Phase 2 performance sweep"
    echo "  figures  Regenerate all figures from results/"
    echo ""
    echo "Options:"
    echo "  --sweep <path>   Path to sweep YAML (default: sweeps/compat.yaml or sweeps/perf.yaml)"
    echo "  --results <dir>  Results directory (default: results/)"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

CMD="$1"
shift

case "$CMD" in
    probe)
        python -m specfp8.probe "$@"
        ;;
    sweep)
        python -m specfp8.sweep "$@"
        ;;
    figures)
        python -m analysis.figures "$@"
        ;;
    *)
        echo "Error: unknown command '$CMD'"
        usage
        ;;
esac
