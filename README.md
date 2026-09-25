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
2. **Throughput of an identical speculative configuration is
   unreproducible across boots** while acceptance is not: CV 13.92% over 12
   fixed-prompt boots, collapsing to 1.44% under `--enforce-eager`. The
   dispersion is *associated with* the CUDA-graph path; the mechanism is
   unidentified, and boot and host effects were not separated. Specific to
   *draft-model* speculation: n-gram gives CV 2.67% at the same n.
3. **Acceptance (tau) has a boot-to-boot CV of 0.62%** under repeated
   identical boots, which we could not find reported. We do *not* convert
   this into a detection threshold: our FP8 precision comparison is
   unreplicated and establishes neither a difference nor an equivalence.
4. **"Inter-token latency" is inter-chunk latency** when computed from
   stream-arrival timestamps. A speculative step emits every accepted token
   in one chunk, so at concurrency 64 a p95-ITL objective scores the
   highest-throughput configuration at zero compliant goodput, at every
   threshold from 20 to 75 ms. This one does not add error -- it inverts the
   decision.

The original hypothesis that motivated the study --- that FP8 KV cache plus
non-causal drafting is silently broken on Ada (SM89) --- was **refuted**
(`findings/F002`). The combination runs. That refutation is why the study
became a measurement-methodology paper.

## The record

```
findings/        29 numbered findings: results, methods, retractions, gaps.
                 Each carries a mandatory "What this does NOT establish".
                 Wrong claims are superseded, never deleted.
results/         Raw records, append-only. ~40 MB, committed on purpose:
                 every claim is one click from its evidence.
analysis/        Scripts that turn raw records into the tables in the paper.
                 Derived output is regenerated, never hand-edited.
docs/            Requirements, design, data model, paper outline.
paper/           LaTeX source, figures, and build script.
```

8 of the 29 findings are retractions. They are kept deliberately. The
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
  subprocesses and are addressed over the OpenAI-compatible API, so the
  experiment cannot use an in-process path unavailable to deployments, and
  server-side request handling, scheduling, batching and streaming are
  included in what is measured. It does *not* reproduce every production
  environment, and it does not eliminate client or network overhead.
- **Content-addressed cells.** A configuration hashes to a `cell_id`;
  identical configuration yields an identical id on any machine. The id
  schema is versioned so new axes do not orphan old results.
- **Append-only records.** A re-run supersedes rather than overwrites, and
  the superseded record stays readable. Most of our corrections were caught
  by re-reading stored records against a hypothesis formed later.
- **Store raw, derive later.** Per-request timings are saved, so goodput at
  any SLO is computed offline at zero GPU cost.

## Status

The paper is a preprint draft. It has not been peer reviewed.

Counts in this file go stale; run the tools rather than trusting them:

```bash
python analysis/check_provenance.py   # 2 disclosed defects (F009, F014)
python analysis/check_paper.py        # must report 0 untraceable
./paper/build.sh                      # blocks if either regresses
```

F009 and F014 rest on probe output predating per-measurement recording;
neither is cited in the paper, and both are disclosed in its threats
section. `paper/build.sh` fails the build on any *additional* defect --- an
earlier version printed the failure and continued, so the audit was
advertised and not enforced.

## License

Code is MIT (`LICENSE`). Measurements, findings and the paper are CC BY 4.0
(`LICENSE-DATA`). The split is deliberate: the harness should be reusable
with minimal friction, and the scientific record should carry attribution.
