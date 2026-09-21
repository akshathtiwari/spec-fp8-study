---
id: F010
title: RETRACTED — "FlashAttention is unavailable in stock vLLM" was wrong
kind: retraction
status: retracted
confidence: high
date: 2026-09-21
evidence:
  runs: [2026-09-21T07-41Z_backend]
  analysis: [analysis/out/tables/backend_attribution.md]
  code_sha: b294050
---

## Claim (withdrawn)

A CPU import check reported `flash_attn: ABSENT (ModuleNotFoundError)`. From
this it was concluded that FlashAttention could not be selected on SM89, that
`auto` must therefore be choosing FlashInfer or Triton, and `FLASH_ATTN` was
**excluded from the backend sweep** to save GPU time. All of that is wrong.

## Why it is wrong

vLLM vendors its own FlashAttention build; it does not depend on the standalone
`flash_attn` PyPI package. The import probe tested the wrong module. The engine
log shows FlashAttention both present and selected:

```
[cuda.py:492] Using FLASH_ATTN attention backend out of potential backends:
              ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
[flash_attn.py:897] Using FlashAttention version 2
```

FLASH_ATTN is in fact the **default** at BF16 KV (F003).

## What caught it

Reading the persisted startup log for backend attribution. Nothing in the
capacity numbers would have revealed it; the check that exposed it was looking
at what the engine said rather than at what it produced.

## Cost of the error

`FLASH_ATTN x fp8_e4m3` was never measured, leaving an untested combination
directly beneath the default configuration — recorded as T19d. Because `auto`
switches to FlashInfer as soon as FP8 KV is requested, this hole is not reachable
by testing `auto` alone and has to be forced explicitly.

## Lesson

An absence check must probe the artifact the system actually loads. "Package not
importable" is not "capability unavailable" when a project vendors its
dependencies.
