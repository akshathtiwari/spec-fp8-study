---
id: F011
title: RETRACTED — backend attribution reported AG_RS, a regex artifact
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

The first run of `backend_report.py` attributed the backend `AG_RS` to every
`attn_backend: auto` cell. `AG_RS` is an all-gather/reduce-scatter
communication setting, not an attention backend. The attribution was a pattern
match on an unrelated log line.

## Why it happened

The matcher included a permissive fallback, `backend\s*=\s*['\"]?(\w+)`,
intended to catch spelling variations across engine versions. It matched a
distributed-communication line instead. The module had been written with an
explicit fallback that dumps candidate log lines when nothing matches — but that
guard only fires when the result is `None`, and a confident wrong match bypasses
it entirely.

Separately, the pattern that should have matched did not: vLLM logs
`Using FLASH_ATTN attention backend`, with "attention" between the name and
"backend", which `Using X backend` does not match.

## What caught it

Reading the output rather than trusting it. `AG_RS` is not a plausible backend
name, and the module's own docstring had stated that a wrong attribution would
misassign the study's central result — which is exactly what it then did.

## Fix

Captures are validated against the real `AttentionBackendEnum` membership list
(36 names, enumerated from the installed engine); anything outside it falls
through to the candidate-lines path. The announcement pattern was corrected.

## Lesson

A fallback that only triggers on `None` does not protect against confident
wrong answers, which are the more dangerous failure. Validating against a closed
vocabulary is what makes a parser safe, not the breadth of its patterns.
