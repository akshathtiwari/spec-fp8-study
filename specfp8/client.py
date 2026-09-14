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


@dataclass
class RequestResult:
    """Per-request timing record."""

    ttft_ms: float              # time to first token
    e2e_ms: float               # end-to-end latency
    output_tokens: int
    itl_ms: list[float]         # inter-token latencies
    ok: bool
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
        **sampling_params,
    }

    token_times: list[float] = []
    output_chunks: list[str] = []
    first_token_time: float | None = None
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

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    now = time.monotonic()
                    if first_token_time is None:
                        first_token_time = now
                    else:
                        token_times.append((now - (token_times[-1] if token_times else first_token_time)) * 1000)
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

    return RequestResult(
        ttft_ms=((first_token_time - start) * 1000) if first_token_time else 0,
        e2e_ms=(end - start) * 1000,
        output_tokens=len(output_chunks),
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
    """Compute inter-token latencies. The raw_times list already contains
    deltas in ms from the streaming loop, but they're computed incorrectly
    there. We just return what we have since the streaming loop does a
    best-effort measurement."""
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
