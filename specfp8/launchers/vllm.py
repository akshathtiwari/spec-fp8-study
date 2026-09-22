"""vLLM engine launcher.

Maps ServerCell fields to vLLM CLI flags. Launches as subprocess,
health-checks via /health, parses KV capacity from startup logs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from specfp8.cells import ServerCell
from specfp8.launchers.base import (
    ServerHandle,
    HealthResult,
    find_free_port,
    poll_health,
    stop_process,
)


class VllmLauncher:
    """Launcher for vLLM's OpenAI-compatible API server."""

    def argv(self, cell: ServerCell, port: int) -> list[str]:
        # vLLM 0.29 CLI surface: `vllm serve <model>`. The per-flag
        # speculative args of earlier releases were collapsed into a single
        # --speculative-config JSON blob, and --guided-decoding-backend into
        # --structured-outputs-config. Request logging is off by default.
        cmd = [
            "vllm", "serve", cell.model,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--seed", "42",
        ]

        # Model revision
        if cell.model_revision and cell.model_revision != "main":
            cmd += ["--revision", cell.model_revision]

        # Weight precision
        if cell.weight_precision == "fp8":
            cmd += ["--quantization", "fp8"]

        # KV cache dtype
        if cell.kv_cache_dtype != "auto":
            cmd += ["--kv-cache-dtype", cell.kv_cache_dtype]

        # Attention backend.
        #
        # --attention-backend is accepted, echoed back in the server's
        # non-default args, and validated against -- but NOT used to select
        # (findings/F018). Cells requesting TRITON_ATTN ran FlashInfer and
        # cells requesting FLASHINFER at BF16 ran FlashAttention, silently.
        # --attention-config is the structured form and is what this now
        # sends. Whether it actually takes effect is never assumed: the probe
        # reads the engine's own selection line after boot and records it
        # alongside the request (see Launcher.selected_backend).
        if cell.attn_backend != "auto":
            cmd += ["--attention-config",
                    json.dumps({"backend": cell.attn_backend})]

        # Speculative decoding
        spec = self.speculative_config(cell)
        if spec is not None:
            cmd += ["--speculative-config", json.dumps(spec)]

        # Guided decoding
        if cell.guided_decoding:
            cmd += ["--structured-outputs-config", json.dumps({"backend": "xgrammar"})]

        # Model cache
        model_cache = os.environ.get("SPECFP8_MODEL_CACHE")
        if model_cache:
            cmd += ["--download-dir", model_cache]

        return cmd

    def speculative_config(self, cell: ServerCell) -> dict | None:
        """Build the --speculative-config payload for this cell.

        Returns None for mechanism=none. Field names match
        vllm.config.SpeculativeConfig in 0.29.
        """
        if cell.mechanism == "none":
            return None

        spec: dict = {
            "method": cell.mechanism,
            "num_speculative_tokens": cell.spec_tokens or 5,
        }

        if cell.mechanism == "ngram":
            # n-gram match window — distinct from the number of tokens proposed
            spec["prompt_lookup_max"] = cell.block_size or 4
            spec["prompt_lookup_min"] = 1
            return spec

        # Draft-model-backed mechanisms (eagle3, dflash) need a checkpoint.
        # mtp reads the draft head from the target checkpoint, so draft_model
        # is optional there.
        if cell.draft_model:
            spec["model"] = cell.draft_model
            if cell.draft_revision:
                spec["revision"] = cell.draft_revision
            spec["draft_tensor_parallel_size"] = 1
        elif cell.mechanism != "mtp":
            raise ValueError(
                f"mechanism={cell.mechanism} requires draft_model to be set"
            )

        return spec

    def start(
        self, cell: ServerCell, cell_id: str, port: int, log_dir: str | Path
    ) -> ServerHandle:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{cell_id}.log"

        cmd = self.argv(cell, port)
        boot_start = time.monotonic()

        log_file = open(log_path, "w")
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ},
            # Own process group, so stop_process can signal the engine's
            # worker processes too rather than orphaning them on the GPU.
            start_new_session=True,
        )

        handle = ServerHandle(
            cell_id=cell_id,
            process_pid=process.pid,
            port=port,
            base_url=f"http://localhost:{port}",
            log_path=str(log_path),
            boot_time_s=0.0,
        )
        handle.set_process(process)
        handle._log_file = log_file  # type: ignore[attr-defined]
        handle._boot_start = boot_start  # type: ignore[attr-defined]

        return handle

    def wait_healthy(
        self, handle: ServerHandle, timeout_s: int = 300
    ) -> HealthResult:
        health_url = f"{handle.base_url}/health"
        result = poll_health(
            health_url, handle.get_process(), timeout_s, handle.log_path
        )
        boot_start = getattr(handle, "_boot_start", time.monotonic())
        handle.boot_time_s = time.monotonic() - boot_start
        return result

    def engine_version(self, handle: ServerHandle) -> str | None:
        """Read the serving vLLM version from its /version endpoint."""
        import httpx
        try:
            resp = httpx.get(f"{handle.base_url}/version", timeout=10)
            resp.raise_for_status()
            return resp.json().get("version")
        except Exception:
            return None

    def selected_backend(self, handle: ServerHandle) -> str | None:
        """The attention backend the engine actually chose, from its own log.

        Never infer this from the flag that was passed. F018 exists because a
        flag was accepted and then disregarded, turning an entire
        experimental axis into one backend repeated.
        """
        try:
            with open(handle.log_path, errors="replace") as f:
                content = f.read()
        except FileNotFoundError:
            return None
        # Two different lines, because the engine takes two different code
        # paths and words them differently:
        #   forced  (cuda.py:432): "Using AttentionBackendEnum.FLASHINFER backend."
        #   auto    (cuda.py:492): "Using FLASH_ATTN attention backend out of
        #                           potential backends: [...]"
        # Matching only the second is why a forced backend read as unverified.
        # Last match wins: the engine can log a provisional choice before the
        # final one.
        for pattern in (
            r"Using\s+AttentionBackendEnum\.([A-Za-z0-9_]+)\s+backend",
            r"Using\s+([A-Za-z0-9_]+)\s+attention\s+backend",
        ):
            found = re.findall(pattern, content)
            if found:
                return found[-1].upper()
        return None

    def kv_capacity(self, handle: ServerHandle) -> int | None:
        """Parse KV cache capacity, in tokens, from the vLLM startup log."""
        try:
            with open(handle.log_path) as f:
                content = f.read()
        except FileNotFoundError:
            return None

        # vLLM 0.29 reports tokens directly, which is what this returns:
        #   "GPU KV cache size: 159,664 tokens, Maximum concurrency for ..."
        match = re.search(
            r"GPU KV cache size:\s*([\d,]+)\s*tokens", content
        )
        if match:
            return int(match.group(1).replace(",", ""))

        # Older releases logged blocks instead. A block count is not a token
        # count, so it is only usable alongside the block size.
        match = re.search(r"# GPU blocks:\s*(\d+)", content)
        if match:
            blocks = int(match.group(1))
            size_match = re.search(r"block_size[=:\s]+(\d+)", content)
            if size_match:
                return blocks * int(size_match.group(1))

        return None

    def stop(self, handle: ServerHandle) -> None:
        stop_process(handle)
        log_file = getattr(handle, "_log_file", None)
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass
