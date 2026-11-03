"""Backport, into an installed vLLM 0.29.0, the one F046 change that applies on the L4.

vLLM 0.30.0 pins two host buffers before FlashInfer's decode plan() copies them
to the GPU with non_blocking=True (F046). On SM89 FlashInfer decodes with its
fa2 backend, and in flashinfer 0.6.18 (vLLM 0.29.0's pin) the kv_lens copy is
made only on the cute-dsl and trtllm-gen paths. So the change that can matter
here is the other one: flashinfer.utils._get_range_buf's cached host arange
buffer, which plan() turns into qo_indptr and copies to the GPU on every
non-cudagraph plan. 0.29.0 leaves it pageable, so the "non_blocking" copy is
staged synchronously.

This inserts vLLM 0.30.0's pin_host_range_buf (vllm/utils/flashinfer.py at
v0.30.0, Apache-2.0) at the top of 0.29.0's fast_plan_decode, at the same point
0.30.0 calls it, and changes nothing else. Run once at image build time; it
refuses to patch anything but the exact upstream 0.29.0 file.
"""
import hashlib
import pathlib

import vllm

TARGET = pathlib.Path(vllm.__file__).parent / "v1" / "attention" / "backends" / "flashinfer.py"
UPSTREAM_029 = "95f9e66762860f94a428c0f7f9ff6564f375ed95c46fc8b9019b3d91c99a57b0"

ANCHOR = '    if not self.is_cuda_graph_enabled or getattr(self, "vllm_first_call", True):\n        self.plan(\n'
PIN = '''    if not self.is_cuda_graph_enabled or getattr(self, "vllm_first_call", True):
        # specfp8 F046 backport of vLLM 0.30.0's pin_host_range_buf: plan()
        # copies qo_indptr from flashinfer's cached host arange buffer with
        # non_blocking=True; pin the cache entry so the copy stays async.
        _specfp8_pin_host_range_buf(len(last_page_len_cpu) + 1)
        self.plan(
'''
HELPER = '''

def _specfp8_pin_host_range_buf(length: int) -> None:
    """vLLM 0.30.0 vllm/utils/flashinfer.py:pin_host_range_buf, verbatim logic."""
    import flashinfer.utils as mod
    if not torch.cuda.is_available():
        return
    get_range_buf = getattr(mod, "_get_range_buf", None)
    cache_buf = getattr(mod, "_cache_buf", None)
    ceil_pow2 = getattr(mod, "_ceil_pow2", None)
    if get_range_buf is None or cache_buf is None or ceil_pow2 is None:
        raise RuntimeError("specfp8 F046 backport: flashinfer.utils lacks the range-buffer cache")
    get_range_buf(length, "cpu")  # Ensure the cache entry exists.
    key = (f"range_{ceil_pow2(length)}", "cpu")
    buf = cache_buf[key]
    if not buf.is_pinned():
        cache_buf[key] = buf.pin_memory()
'''

src = TARGET.read_bytes()
digest = hashlib.sha256(src).hexdigest()
if digest != UPSTREAM_029:
    raise SystemExit(f"{TARGET} is not upstream vLLM 0.29.0 (sha256 {digest}); not patching")
text = src.decode()
if text.count(ANCHOR) != 1:
    raise SystemExit("fast_plan_decode anchor not found exactly once; not patching")
text = text.replace(ANCHOR, PIN) + HELPER
TARGET.write_text(text)
print(f"patched {TARGET}: sha256 {digest[:16]} -> {hashlib.sha256(text.encode()).hexdigest()[:16]}")
