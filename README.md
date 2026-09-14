# Speculative Decoding × FP8 Compatibility Study

**Does FP8 KV cache actually work with speculative decoding on the GPUs people rent?**

This repo contains the measurement harness for an independent study of how FP8
quantization interacts with speculative decoding across draft mechanisms,
precision configurations, and serving engines.

## The question

Production LLM serving stacks speculative decoding and FP8 quantization
together. The published literature evaluates them separately. We measure the
interaction directly: compatibility, acceptance length (τ), and goodput under
SLO across a concurrency sweep.

The specific hypothesis (H1): FP8 KV cache combined with non-causal speculative
drafting (DFlash) may be silently broken on SM89 (Ada: RTX 4090 / L4 / L40S)
because the dequant path is gated to SM100 (datacenter Blackwell).

## Quick start

```bash
# Clone and install
git clone https://github.com/<user>/spec-fp8-study.git
cd spec-fp8-study
pip install -e .

# Set model cache (optional)
export SPECFP8_MODEL_CACHE=~/.cache/huggingface/hub

# Phase 1: compatibility matrix
./run.sh probe --sweep sweeps/compat.yaml

# Phase 2: performance sweep (runs only cells Phase 1 marked ok)
./run.sh sweep --sweep sweeps/perf.yaml

# Regenerate figures from results
./run.sh figures
```

## Hardware requirements

- One CUDA GPU with FP8 support (compute capability ≥ 8.9)
- Target: RTX 4090 / L4 / L40S (Ada, SM89)
- 24+ GB VRAM
- vLLM and/or SGLang installed (in separate environments if needed)

## Project structure

```
spec-fp8-study/
├── docs/                    # SDD: requirements, design, tasks
├── specfp8/                 # Measurement harness (pure Python, HTTP-only)
│   ├── cells.py             # Cell model, stable ID, sweep expansion
│   ├── env.py               # GPU/driver/CUDA version capture
│   ├── store.py             # Atomic JSONL append + resume
│   ├── client.py            # Async OpenAI-compatible load generator
│   ├── correctness.py       # Three-layer correctness testing
│   ├── probe.py             # Phase 1 — compatibility matrix
│   ├── sweep.py             # Phase 2 — performance sweep
│   ├── launchers/           # Engine-specific subprocess launchers
│   ├── metrics/             # Prometheus counter scraping
│   ├── workloads/           # Prompt sources and validators
│   └── analysis/            # Offline analysis (goodput, figures)
├── sweeps/                  # Sweep configuration YAMLs
├── results/                 # Output (gitignored)
└── run.sh                   # Single entrypoint (N6)
```

## Design principles

- **HTTP-only boundary**: the harness never imports vLLM or SGLang. Engines run
  as subprocesses, communication is via the OpenAI-compatible API.
- **Atomic resume**: results are written per-cell with fsync + os.replace. An
  interrupted sweep resumes without re-running completed cells.
- **Store raw, derive later**: per-request timings are saved so goodput at any
  SLO is computed offline at zero GPU cost.

## Documentation

- [Requirements](docs/requirements.md) — what and why
- [Design](docs/design.md) — how
- [Tasks](docs/tasks.md) — build order

## License

MIT
