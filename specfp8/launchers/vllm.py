"""vLLM engine launcher.

Maps ServerCell fields to vLLM CLI flags. Launches as subprocess,
health-checks via /health, parses KV capacity from startup logs.
"""

from __future__ import annotations

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
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", cell.model,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--seed", "42",
            "--disable-log-requests",
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

        # Attention backend
        if cell.attn_backend != "auto":
            cmd += ["--attention-backend", cell.attn_backend]

        # Speculative decoding
        if cell.mechanism == "ngram":
            cmd += [
                "--speculative-model", "[ngram]",
                "--num-speculative-tokens", str(cell.spec_tokens or 5),
                "--ngram-prompt-lookup-max", str(cell.spec_tokens or 5),
            ]
        elif cell.mechanism == "eagle3":
            if cell.draft_model:
                cmd += [
                    "--speculative-model", cell.draft_model,
                    "--num-speculative-tokens", str(cell.spec_tokens or 5),
                    "--speculative-draft-tensor-parallel-size", "1",
                ]
                if cell.draft_revision:
                    cmd += ["--speculative-model-revision", cell.draft_revision]
        elif cell.mechanism == "dflash":
            if cell.draft_model:
                cmd += [
                    "--speculative-model", cell.draft_model,
                    "--num-speculative-tokens", str(cell.spec_tokens or 5),
                ]
                if cell.draft_revision:
                    cmd += ["--speculative-model-revision", cell.draft_revision]
                if cell.block_size:
                    cmd += ["--speculative-dflash-block-size", str(cell.block_size)]
        elif cell.mechanism == "mtp":
            cmd += [
                "--num-speculative-tokens", str(cell.spec_tokens or 5),
                "--speculative-model", "[mtp]",
            ]

        # Guided decoding
        if cell.guided_decoding:
            cmd += ["--guided-decoding-backend", "xgrammar"]

        # Model cache
        model_cache = os.environ.get("SPECFP8_MODEL_CACHE")
        if model_cache:
            cmd += ["--download-dir", model_cache]

        return cmd

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

    def kv_capacity(self, handle: ServerHandle) -> int | None:
        """Parse num_gpu_blocks from vLLM startup log."""
        try:
            with open(handle.log_path) as f:
                content = f.read()
        except FileNotFoundError:
            return None

        # vLLM logs: "# GPU blocks: 8192, # CPU blocks: 512"
        match = re.search(r"# GPU blocks:\s*(\d+)", content)
        if match:
            return int(match.group(1))

        return None

    def stop(self, handle: ServerHandle) -> None:
        stop_process(handle)
        log_file = getattr(handle, "_log_file", None)
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass
