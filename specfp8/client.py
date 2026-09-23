"""Async OpenAI-compatible load generator.

Talks to any engine exposing the /v1/chat/completions endpoint. Records
per-request timings: TTFT, ITL, E2E, output token count. Streaming SSE
parsing for token-level timing.

For probe (Phase 1): concurrency 1, warmup + measure.
For sweep (Phase 2): configurable concurrency (N in flight at all times).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx


#: Above this, an inter-token gap is worth flagging. It is NOT worth
#: failing on: under concurrency vLLM preempts and recomputes requests
#: under KV pressure, and a preempted request genuinely waits. A 253-second
#: gap was observed at concurrency 64 with 256 queued requests, and it is a
#: real measurement of a real stall.
LARGE_ITL_MS = 60_000.0


def itl_impossible(itls: list[float]) -> int:
    """Count inter-token latencies that cannot be measurements at all.

    Negative or non-finite only. Time does not run backwards and a gap is
    not infinite, so these are always harness defects — and they are the
    F025 signature: that bug produced alternating signs within three tokens
    and +/-Infinity within a hundred.

    Deliberately NOT a magnitude test. The first version of this guard
    raised on anything above 60s, on the reasoning that "an inter-token
    latency above a second on a local server is a bug". That reasoning was
    an assumption about the system stated without measuring it, which is
    the error this entire study is about, and it killed a paid run on a
    legitimate 253s preemption stall. The distinction that survives is
    *impossible* versus *merely extreme*.
    """
    import math
    return sum(1 for v in itls if not math.isfinite(v) or v < 0)


def itl_extreme(itls: list[float]) -> int:
    """Count gaps large enough to be worth reporting but not to reject."""
    import math
    return sum(1 for v in itls if math.isfinite(v) and v > LARGE_ITL_MS)


@dataclass
class RequestResult:
    """Per-request timing record."""

    ttft_ms: float              # time to first token
    e2e_ms: float               # end-to-end latency
    output_tokens: int
    itl_ms: list[float]         # inter-token latencies
    ok: bool
    #: True when output_tokens came from the server's usage field. False means
    #: it is a chunk count, which undercounts by ~tau under speculation.
    output_tokens_from_usage: bool = False
    #: Streamed chunks received. Under speculation a chunk can carry several
    #: tokens, so chunks/token is an observable proxy for acceptance.
    n_stream_chunks: int = 0
    output_text: str = ""
    sampling_params: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class ProbeResult:
    """Result of a probe run (concurrency 1, for correctness)."""

    requests: list[RequestResult]
    outputs: list[str]          # generated texts, aligned with requests


async def generate_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    sampling_params: dict,
    timeout_s: float = 120.0,
) -> RequestResult:
    """Send one streaming chat completion request, record timings."""
    messages = [{"role": "user", "content": prompt}]

    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        # Ask the server for real token counts. Counting streamed chunks
        # instead undercounts by roughly tau under speculative decoding,
        # because accepted draft tokens arrive together in one chunk -- which
        # makes speculation look slower than no speculation at all.
        "stream_options": {"include_usage": True},
        **sampling_params,
    }

    token_times: list[float] = []      # inter-token latencies, milliseconds
    output_chunks: list[str] = []
    usage_tokens: int | None = None
    first_token_time: float | None = None
    #: Arrival time of the previous content chunk, monotonic seconds. Kept
    #: separate from token_times because conflating the two is what produced
    #: the ITL corruption in F025.
    prev_token_time: float = 0.0
    start = time.monotonic()

    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=body,
            timeout=timeout_s,
        ) as resp:
            if resp.status_code != 200:
                error_body = ""
                async for chunk in resp.aiter_text():
                    error_body += chunk
                return RequestResult(
                    ttft_ms=0, e2e_ms=0, output_tokens=0, itl_ms=[],
                    ok=False, sampling_params=sampling_params,
                    error=f"HTTP {resp.status_code}: {error_body[:500]}",
                )

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # The final usage chunk carries no choices.
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    usage_tokens = usage["completion_tokens"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    now = time.monotonic()
                    if first_token_time is None:
                        first_token_time = now
                        prev_token_time = now
                    else:
                        # Gap since the PREVIOUS ARRIVAL, tracked separately
                        # from the latency list.
                        #
                        # This previously read token_times[-1] as the previous
                        # timestamp, but token_times holds latencies in
                        # milliseconds, not monotonic seconds. After the first
                        # append it subtracted a millisecond latency from a
                        # seconds timestamp and fed the result back in, so the
                        # values alternated sign and grew about 1000x per
                        # token, reaching +/-Infinity within ~100 tokens.
                        #
                        # Every itl_ms array recorded before 2026-09-23 is
                        # therefore unusable, and p95 ITL was Infinity, which
                        # is why every goodput figure was 0.0. TTFT, e2e,
                        # token counts, throughput and tau are computed
                        # elsewhere and are unaffected. See findings/F025.
                        token_times.append((now - prev_token_time) * 1000)
                        prev_token_time = now
                    output_chunks.append(content)

    except (httpx.TimeoutException, httpx.ReadError, httpx.ConnectError) as e:
        return RequestResult(
            ttft_ms=0, e2e_ms=(time.monotonic() - start) * 1000,
            output_tokens=0, itl_ms=[], ok=False,
            sampling_params=sampling_params, error=str(e),
        )

    end = time.monotonic()
    output_text = "".join(output_chunks)

    # Recompute ITL properly from absolute timestamps
    itl_ms = _compute_itl(first_token_time, token_times, start)

    # Prefer the server's count. Fall back to chunk count only if usage is
    # missing, and say so, because that fallback is wrong under speculation.
    return RequestResult(
        ttft_ms=((first_token_time - start) * 1000) if first_token_time else 0,
        e2e_ms=(end - start) * 1000,
        output_tokens=usage_tokens if usage_tokens is not None
                      else len(output_chunks),
        output_tokens_from_usage=usage_tokens is not None,
        n_stream_chunks=len(output_chunks),
        itl_ms=itl_ms,
        ok=True,
        output_text=output_text,
        sampling_params=sampling_params,
    )


def _compute_itl(
    first_token_time: float | None,
    raw_times: list[float],
    start: float,
) -> list[float]:
    """Validate the inter-token latencies produced by the streaming loop.

    This used to be a no-op passthrough whose docstring said the values
    "are computed incorrectly there" and returned them anyway. They were
    indeed incorrect (F025): every element after the first was garbage, up
    to +/-Infinity, and every goodput figure in the study was 0.0 as a
    result. A function that names a defect and then propagates it is worse
    than no function, because it makes the defect look considered.

    The loop is fixed, so this now checks rather than excuses. Implausible
    values raise: a wrong number that reaches a table is far more expensive
    than a run that stops, and F025's whole lesson is that a confident 0.0
    is invisible while an exception is not.
    """
    import math

    bad = itl_impossible(raw_times)
    if bad:
        sample = [v for v in raw_times
                  if not math.isfinite(v) or v < 0][:3]
        raise ValueError(
            f"{bad} of {len(raw_times)} inter-token latencies are "
            f"impossible (negative or non-finite): {sample}. Time does not "
            f"run backwards; this is a harness defect -- see findings/F025."
        )
    return raw_times


async def probe_run(
    base_url: str,
    model: str,
    prompts: list[str],
    sampling_params: dict,
    n_warmup: int = 3,
    timeout_s: float = 120.0,
) -> ProbeResult:
    """Run prompts sequentially (concurrency 1) for correctness testing.

    Sends n_warmup throwaway requests first to let CUDA graphs and
    torch.compile settle.
    """
    async with httpx.AsyncClient() as client:
        # Warmup — discard results
        for i in range(min(n_warmup, len(prompts))):
            await generate_one(
                client, base_url, model, prompts[i], sampling_params, timeout_s
            )

        # Measure
        results: list[RequestResult] = []
        outputs: list[str] = []

        for prompt in prompts:
            result = await generate_one(
                client, base_url, model, prompt, sampling_params, timeout_s
            )
            results.append(result)
            outputs.append(result.output_text if result.ok else "")

    return ProbeResult(requests=results, outputs=outputs)


async def load_run(
    base_url: str,
    model: str,
    prompts: list[str],
    sampling_params: dict,
    concurrency: int,
    n_warmup: int = 5,
    timeout_s: float = 120.0,
) -> list[RequestResult]:
    """Run prompts at fixed concurrency (N in flight at all times).

    Used by Phase 2 sweep. Warmup requests are issued and discarded first.
    """
    async with httpx.AsyncClient() as client:
        # Warmup
        warmup_tasks = [
            generate_one(client, base_url, model, prompts[i % len(prompts)],
                         sampling_params, timeout_s)
            for i in range(n_warmup)
        ]
        await asyncio.gather(*warmup_tasks)

        # Measure with fixed concurrency
        semaphore = asyncio.Semaphore(concurrency)
        results: list[RequestResult] = []

        async def _run_one(prompt: str) -> RequestResult:
            async with semaphore:
                return await generate_one(
                    client, base_url, model, prompt, sampling_params, timeout_s
                )

        tasks = [_run_one(p) for p in prompts]
        results = await asyncio.gather(*tasks)

    return list(results)
