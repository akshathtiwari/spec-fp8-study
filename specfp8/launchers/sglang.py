"""SGLang engine launcher.

Maps ServerCell fields to SGLang CLI flags. Launches as subprocess,
health-checks via /health_generate, parses KV capacity from startup logs.
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


class SglangLauncher:
    """Launcher for SGLang's OpenAI-compatible API server."""

    def argv(self, cell: ServerCell, port: int) -> list[str]:
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", cell.model,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--random-seed", "42",
            "--disable-radix-cache",  # determinism for correctness tests
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
                "--speculative-algorithm", "NGRAM",
                "--speculative-num-draft-tokens", str(cell.spec_tokens or 5),
            ]
        elif cell.mechanism == "eagle3":
            if cell.draft_model:
                cmd += [
                    "--speculative-algorithm", "EAGLE",
                    "--speculative-eagle-path", cell.draft_model,
                    "--speculative-num-draft-tokens", str(cell.spec_tokens or 5),
                ]
        elif cell.mechanism == "dflash":
            if cell.draft_model:
                cmd += [
                    "--speculative-algorithm", "DFLASH",
                    "--speculative-dflash-path", cell.draft_model,
                    "--speculative-num-draft-tokens", str(cell.spec_tokens or 5),
                ]
                if cell.block_size:
                    cmd += ["--speculative-dflash-block-size", str(cell.block_size)]
        elif cell.mechanism == "mtp":
            cmd += [
                "--speculative-algorithm", "MTP",
                "--speculative-num-draft-tokens", str(cell.spec_tokens or 5),
            ]

        # Model cache
        model_cache = os.environ.get("SPECFP8_MODEL_CACHE")
        if model_cache:
            cmd += ["--model-cache-dir", model_cache]

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
        # SGLang uses /health_generate for readiness
        health_url = f"{handle.base_url}/health_generate"
        result = poll_health(
            health_url, handle.get_process(), timeout_s, handle.log_path
        )

        # Fallback: try /health if /health_generate doesn't exist
        if not result.ok and result.status == "timeout":
            fallback_url = f"{handle.base_url}/health"
            result = poll_health(
                fallback_url, handle.get_process(), 30, handle.log_path
            )

        boot_start = getattr(handle, "_boot_start", time.monotonic())
        handle.boot_time_s = time.monotonic() - boot_start
        return result

    def engine_version(self, handle: ServerHandle) -> str | None:
        """Read the serving SGLang version from /get_server_info."""
        import httpx
        for path, key in (("/get_server_info", "version"),
                          ("/get_server_args", "version")):
            try:
                resp = httpx.get(f"{handle.base_url}{path}", timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                continue
            for k in (key, "sglang_version", "server_version"):
                if isinstance(data, dict) and data.get(k):
                    return str(data[k])
        return None

    def kv_capacity(self, handle: ServerHandle) -> int | None:
        """Parse KV capacity from SGLang startup log."""
        try:
            with open(handle.log_path) as f:
                content = f.read()
        except FileNotFoundError:
            return None

        # SGLang logs vary by version; try common patterns
        # "max_total_num_tokens: 131072"
        match = re.search(r"max_total_num_tokens:\s*(\d+)", content)
        if match:
            return int(match.group(1))

        # "Number of KV cache tokens: 131072"
        match = re.search(r"KV cache tokens:\s*(\d+)", content)
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
