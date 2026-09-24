# Endorsement outreach — verified candidates and draft emails

**Endorsement code: `IGNP9M`** (cs.LG). Issued 2026-09-24.
Endorser approval URL: https://arxiv.org/auth/endorse?x=IGNP9M

Submission: `https://arxiv.org/submit/8123115` (paused at the endorsement gate).

---

## Verified qualified (checked via arXiv's own endorser lookup)

arXiv's rule: an endorser needs 3+ papers in any of ~40 `cs.*` categories,
submitted between 3 months and 5 years ago. Checked with
`arxiv.org/auth/show-endorsers/<id>` rather than assumed.

| Person | Paper | Can endorse | Why them |
|---|---|---|---|
| **Linghao Kong** | [2605.15051](https://arxiv.org/abs/2605.15051) | cs.AI, **cs.LG**, cs.PF | First author of the latency model our confound 2 bears on directly |
| **Alexandre Marques** | [2605.15051](https://arxiv.org/abs/2605.15051) | **cs.LG**, cs.CE, cs.AI, cs.CL, cs.PF | Senior author, same paper; vLLM performance ecosystem |
| **Xiaoxuan Liu** | [2601.11580](https://arxiv.org/abs/2601.11580), [2406.14066](https://arxiv.org/abs/2406.14066) | cs.DB, cs.PL, **cs.LG**, cs.PF, cs.AI, cs.CL | First author of *Performance or Illusion?* (MLSys 2026) and TurboSpec; we cite both |

**Checked and NOT qualified** — do not email for this purpose:

- Shikhar Shukla (SpecKV, 2605.02888) — not currently an endorser
- Jiaxiang Yu (2601.11580) — not currently an endorser
- Jongseok Park, Ion Stoica, Alvin Cheung, Megan Flynn, Michael Peng,
  Nir Shavit, Mark Kurtz — not registered as owners of those papers, so the
  lookup cannot confirm them

## Order to ask

1. **Linghao Kong** — best hook, first author, likely lowest inbound volume.
2. **Xiaoxuan Liu** — strongest topical fit of anyone; higher inbound volume.
3. **Alexandre Marques** — senior author fallback on the same paper as (1).

One at a time, roughly a week apart. Do not send to all three at once: if two
endorse, the second is wasted goodwill, and cc'ing a group makes it
everyone's problem and therefore no one's.

## Where to find addresses

Author emails are normally on the first page of the paper PDF or the
author's institutional page. Use the address the author published for
correspondence; do not guess a pattern.

---

## Draft: to Linghao Kong

> **Subject:** arXiv endorsement request (cs.LG) — boot-to-boot variance in vLLM speculative decoding
>
> Hi Linghao,
>
> I am an independent researcher working on LLM inference measurement. I have
> a preprint ready and need a cs.LG endorsement to post it, as it would be my
> first arXiv submission. arXiv's lookup lists you as qualified to endorse for
> cs.LG.
>
> I am writing to you specifically because of your latency model paper
> (arXiv:2605.15051). I measured something that is an unmodelled error term in
> exactly that setting: on vLLM 0.29 with a DFlash drafter on an L4, the
> throughput of a byte-identical configuration has a coefficient of variation
> of 13.92% across 12 fixed-prompt boots, while acceptance stays put. Under
> --enforce-eager it drops to 1.44%. n-gram speculation does not show it.
>
> I am not claiming your fit is wrong. You measure A100 and H100, I measure an
> L4, and I localise the effect to draft-model speculation. But your validation
> is a single sweep per configuration, and if this dispersion exists on your
> hardware too, it is inside the residual.
>
> The paper is a measurement case study. It reports four confounds, two
> retractions of my own headline claims, and ships the raw records and the
> scripts that audit them.
>
> Paper: [attach paper.pdf]
> Code and data: github.com/akshathtiwari/spec-fp8-study
>
> If you are willing, the approval is at
> https://arxiv.org/auth/endorse?x=IGNP9M and takes about two minutes. As I
> understand it, endorsement only means the work belongs in the archive, not
> that you are reviewing or vouching for it.
>
> Either way, thanks for reading, and the latency decomposition was useful to
> me regardless.
>
> Akshath Tiwari
> github.com/akshathtiwari

## Draft: to Xiaoxuan Liu

> **Subject:** arXiv endorsement request (cs.LG) — follow-on measurement to Performance or Illusion?
>
> Hi Xiaoxuan,
>
> I am an independent researcher working on LLM inference measurement. I need
> a cs.LG endorsement to post my first arXiv preprint; arXiv's lookup lists
> you as qualified.
>
> I am writing to you because my paper sits directly downstream of
> "Performance or Illusion?" and cites both it and TurboSpec. You found that
> acceptance varies across positions, requests and datasets. I measured
> something adjacent and, I think, complementary: repeated boots of a
> byte-identical configuration differ by CV 13.92% in throughput while
> acceptance does not move. That is instrument dispersion rather than
> workload variation, and it means a single-run speculative benchmark on this
> stack is not reproducible to within a factor of roughly 1.5.
>
> I also found that inter-token latency computed from stream arrivals is
> inter-chunk latency under speculation, so a p95-ITL objective scored my
> highest-throughput configuration at zero compliant goodput at concurrency
> 64, at every threshold from 20 to 75 ms.
>
> It is a case study on one L4 and one engine release, and it says so. It
> carries two retractions of its own headline claims.
>
> Paper: [attach paper.pdf]
> Code and data: github.com/akshathtiwari/spec-fp8-study
>
> If you are willing: https://arxiv.org/auth/endorse?x=IGNP9M, about two
> minutes. Endorsement as I understand it means the work belongs in the
> archive, not that you have reviewed it.
>
> Thanks either way,
> Akshath Tiwari
> github.com/akshathtiwari

---

## Notes on the drafts

- Each names a specific, checkable reason for contacting that person. A
  generic endorsement request reads as a mass mail.
- Each states the effort ("about two minutes") and what endorsement is not
  ("reviewing or vouching"). The usual reason for silence is assuming it
  costs more.
- Each concedes the limitation before being asked. Leading with "one L4, one
  engine release" is more credible than being caught on it.
- The retractions stay in. To this audience they read as competence.
- No flattery, no explanation of why you personally need it. The ask is
  administrative.
