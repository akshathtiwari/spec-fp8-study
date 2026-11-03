---
id: F049
title: RETRACTION of F046's candidate -- backporting vLLM 0.30.0's pin of FlashInfer's host arange buffer into 0.29.0 leaves the deficit unchanged under gVisor (0.726 against 0.736 unpatched); F046 also overstated its scope, since on the L4 the kv_lens copy it named is not made, and both arms, not only the graph arm, make the remaining copy every step
kind: retraction
status: established
confidence: high
date: 2026-11-03
evidence:
  runs: [2026-11-03T03-29Z_boot_variance, 2026-11-03T08-29Z_boot_variance]
  logs: [results/gcp/2026-11-03_gvisor_decision_rule.txt,
         results/gcp/2026-11-03_gvisor_leg4_run.log,
         results/external/vllm/v0.29.0/flashinfer.py]
  analysis: [analysis/sandbox_legs.py, analysis/out/tables/sandbox_legs.md,
             analysis/vllm_pinning.py, analysis/out/tables/vllm_pinning.md]
  code_sha: pending
  external: [cloud/gcp/f046_pin_range_buf.py,
             flashinfer v0.6.18 flashinfer/decode.py and flashinfer/utils.py,
             vllm v0.30.0 vllm/utils/flashinfer.py:pin_host_range_buf]
supersedes: [F046]
superseded_by: []
---

## Claim

F046 read the vLLM 0.29.0 -> 0.30.0 diff and proposed that 0.30.0 removed the
deficit by pinning two host buffers that FlashInfer's decode plan copies to
the GPU with `non_blocking=True`. It said this was a candidate, not tested,
at low confidence. It has now been tested, and the candidate is withdrawn.

**What was believed.** In 0.29.0 both buffers are pageable, so every decode
step makes a "non-blocking" copy that CUDA stages synchronously. 0.30.0 pins
both, keeping the copies async, which on a platform that makes host
synchronisation expensive could remove the graph-arm deficit.

**What is wrong with it.**

1. **The test fails.** Under gVisor on a GCP L4, where 0.29.0 reproduces the
   deficit (F048), the backport of the one applicable pin leaves it in place:
   0.29.0 + pin gives **0.726** [0.715, 0.738], against 0.736 unpatched and
   1.040 for 0.30.0 in the same sandbox, on the same card.
2. **One of the two buffers is not copied on this hardware.** In flashinfer
   0.6.18 (vLLM 0.29.0's pin) the kv_lens host-to-device copy is made only on
   the cute-dsl and trtllm-gen decode paths. On SM89 FlashInfer decodes with
   fa2, which does not copy it. 0.30.0's own comment says as much
   ("trtllm-gen/cute-dsl backends"). F046 listed both buffers as live on the
   L4. Only the arange buffer (`_get_range_buf`, turned into `qo_indptr`) is.
3. **Both arms make the copy.** Speculative decoding with FlashInfer falls
   back from FULL_AND_PIECEWISE to PIECEWISE graphs (every engine log says
   so). In 0.29.0 the CUDA-graph decode wrapper is used only when the decode
   mode is FULL, so in both arms `fast_plan_decode` runs the full `plan()`
   every step. F046 did not establish that the copy is specific to the arm
   that slows down. That was the weakest link, and the test removes the
   question.

**What survives from F046.** The code reading is accurate as a description
of the two releases: 0.30.0 does pin those buffers where 0.29.0 does not
(`vllm_pinning.md`). It is not the difference that matters here.

## Evidence

`cloud/gcp/f046_pin_range_buf.py` patches the installed vLLM at image build
time. It refuses any `flashinfer.py` but upstream 0.29.0's (sha256
95f9e667...), inserts 0.30.0's `pin_host_range_buf` logic at the point 0.30.0
calls it, and changes nothing else. The build log shows the patch applied
(sha256 95f9e66762860f94 -> c2c78894bc737762). Leg 4's 72 records carry
`image: specfp8-vllm:0.29.0-f046`.

The kv_lens scope and the PIECEWISE fallback were read from flashinfer
v0.6.18's `decode.py` (the `cute-dsl` and `trtllm-gen` branches are the only
ones that stage `_kv_lens_buffer`), from vLLM 0.29.0's `flashinfer.py`
(`enable_cuda_graph = decode_mode() == CUDAGraphMode.FULL`), and from the
engine logs ("CUDAGraphMode.FULL_AND_PIECEWISE is not supported with
spec-decode for attention backend FlashInferBackend", then "Capturing CUDA
graphs (PIECEWISE)").

## Reasoning

The backport removes the pageable source of the one live copy and leaves the
deficit within 1.4% of where it was. If that copy were the mechanism,
pinning it would have moved the ratio toward 1.0. It did not move, so the
copy is not the mechanism, or not enough of it to see.

One caveat is mechanical. The patch pins the cached buffer on first use, as
0.30.0 does. The run did not instrument whether `is_pinned()` held at every
call. 0.30.0's helper has the same structure and does remove the deficit, but
0.30.0 changed much else.

## What this does NOT establish

- That no host-synchronisation change between 0.29.0 and 0.30.0 is
  responsible. Only this one was tested.
- Which change is responsible. A bisection of the 0.29.0 -> 0.30.0 range
  under gVisor would answer it and has not been run.

## How to reproduce

```bash
F046=1 bash cloud/gcp/host.sh build      # prints "patched ... -> c2c78894..."
bash cloud/gcp/host.sh gvisor 20260914.0
F046=1 RUNTIME=runsc THREADS1=1 bash cloud/gcp/host.sh run --mechanism dflash --boots 12
python analysis/sandbox_legs.py          # the "0.29.0 + F046 pin" row
```
