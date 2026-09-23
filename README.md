# What You Are Actually Measuring When You Benchmark Speculative Decoding Under FP8

An independent measurement study of speculative decoding and FP8 quantization
on vLLM 0.29.0 / NVIDIA L4 (SM89), with the harness, the raw records, and the
full history of its own corrections.

**Paper:** [`paper/paper.pdf`](paper/paper.pdf) (source: `paper/paper.tex`)

---

## What this found

We set out to measure how FP8 quantization interacts with speculative
decoding. We could not, four times, until the measurement apparatus itself
was fixed. Those four obstacles turned out to be larger than the effect they
obscured:

1. **Attention-backend selection is conditioned on KV-cache dtype.** At BF16
   KV vLLM selects FlashAttention-2; at `fp8_e4m3` it selects FlashInfer. A
   precision A/B under default settings is a kernel A/B as well. Target and
   draft models select *independently*, so pinning one does not pin the other.
2. **Throughput of an identical speculative configuration varies up to 2.96x
   across boots** while acceptance does not. The variation is associated with
   the CUDA-graph execution path and is fixed at boot. It is specific to
   *draft-model* speculation: n-gram speculation shows 1.10x over twelve boots.
3. **Acceptance (tau) has a boot-to-boot noise floor of 1.32%** that we could
   not find reported anywhere. Effects smaller than that are unfalsifiable;
   we withdrew one of our own published claims that sat at 1.7x the floor.
4. **"Inter-token latency" is inter-chunk latency.** A speculative step emits
   every accepted token in one stream chunk, so a p95-ITL objective ranks the
   highest-throughput configuration in our grid as fully non-compliant. This
   one does not add error -- it inverts the decision.

The original hypothesis that motivated the study --- that FP8 KV cache plus
non-causal drafting is silently broken on Ada (SM89) --- was **refuted**
(`findings/F002`). The combination runs. That refutation is why the study
became a measurement-methodology paper.

## The record

```
findings/        27 numbered findings: results, methods, retractions, gaps.
                 Each carries a mandatory "What this does NOT establish".
                 Wrong claims are superseded, never deleted.
results/         Raw records, append-only. ~40 MB, committed on purpose:
                 every claim is one click from its evidence.
analysis/        Scripts that turn raw records into the tables in the paper.
                 Derived output is regenerated, never hand-edited.
docs/            Requirements, design, data model, paper outline.
paper/           LaTeX source, figures, and build script.
```

Seven of the 27 findings are retractions. They are kept deliberately. The
study's argument is that benchmark numbers are routinely reported without
their error terms, and the most honest evidence for that is the list of times
we did it ourselves and caught it.

## Verifying it rather than trusting it

```bash
python analysis/check_provenance.py
```

Walks the chain a reader would walk: every finding's citations resolve to
stored records, stored ids still recompute from their stored configs,
retraction links are symmetric, and no result rests on terminal output. Exits
with the defect count.

```bash
python analysis/check_paper.py
```

Every distinctive numeric figure in the paper must appear in the underlying
records. It states its own limits in its docstring: it proves no figure is
invented, not that every figure is correct.

```bash
./paper/build.sh
```

Regenerates the figure from raw records, runs both checkers, and only then
typesets. A paper that fails its own audit does not get built.

## Reproducing the measurements

Every result is re-derivable by a third party on rented hardware. The study
cost tens of dollars of per-second L4 time in total; `analysis/budget_report.py`
prints the ledger.

```bash
git clone https://github.com/akshathtiwari/spec-fp8-study.git
cd spec-fp8-study && pip install -e .

# Phase 2 performance grid: 16 boots, both CUDA-graph arms
modal run cloud/modal_probe.py --engine sweep --sweep perf_v2 --force

# Boot-to-boot variance for one mechanism
modal run cloud/modal_probe.py --engine bootvar --sweep dflash --repeats 12
```

Requires a CUDA GPU with compute capability >= 8.9 and ~22 GiB of VRAM. The
`cloud/` runner uses [Modal](https://modal.com); the harness itself is
engine-agnostic and speaks only HTTP.

## Design principles that earned their keep

- **HTTP-only boundary.** The harness never imports vLLM. Engines run as
  subprocesses and are addressed over the OpenAI-compatible API, so what is
  measured is what a serving deployment runs.
- **Content-addressed cells.** A configuration hashes to a `cell_id`;
  identical configuration yields an identical id on any machine. The id
  schema is versioned so new axes do not orphan old results.
- **Append-only records.** A re-run supersedes rather than overwrites, and
  the superseded record stays readable. Most of our corrections were caught
  by re-reading stored records against a hypothesis formed later.
- **Store raw, derive later.** Per-request timings are saved, so goodput at
  any SLO is computed offline at zero GPU cost.

## Status

The paper is a preprint draft. It has not been peer reviewed, and
`check_provenance.py` currently reports two known defects (findings F009 and
F014 rest on probe output that predates per-measurement recording; neither is
cited in the paper). Both are stated in the paper rather than suppressed.

## License

Code is MIT (`LICENSE`). Measurements, findings and the paper are CC BY 4.0
(`LICENSE-DATA`). The split is deliberate: the harness should be reusable
with minimal friction, and the scientific record should carry attribution.
