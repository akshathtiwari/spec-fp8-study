# Endorsement outreach — verified candidates and draft emails

**Endorsement code: `IGNP9M`** (cs.LG). Issued 2026-09-24.
Endorser approval URL: https://arxiv.org/auth/endorse?x=IGNP9M

Submission: `https://arxiv.org/submit/8123115` (paused at the endorsement gate).

---

## Verified qualified (checked via arXiv's own endorser lookup)

arXiv's rule: an endorser needs 3+ papers in any of ~40 `cs.*` categories,
submitted between 3 months and 5 years ago. Checked with
`arxiv.org/auth/show-endorsers/<id>` rather than assumed.

| Person | Email | Paper | Can endorse | Why them | Status |
|---|---|---|---|---|---|
| **Linghao Kong** | `linghao@mit.edu` | [2605.15051](https://arxiv.org/abs/2605.15051) | cs.AI, **cs.LG**, cs.PF | First author of the latency model our confound 2 bears on directly | sent 2026-09-24 |
| **Xiaoxuan Liu** | `xiaoxuan_liu@berkeley.edu` | [2601.11580](https://arxiv.org/abs/2601.11580), [2406.14066](https://arxiv.org/abs/2406.14066) | cs.DB, cs.PL, **cs.LG**, cs.PF, cs.AI, cs.CL | First author of *Performance or Illusion?* (MLSys 2026) and TurboSpec; we cite both | drafted |
| **Maor Ashkenazi** | `mashkenazi@nvidia.com` | [2604.09557](https://arxiv.org/abs/2604.09557) | cs.CV, **cs.LG**, cs.CR, cs.CL, cs.AI, cs.DC | SPEED-Bench author; we cite it, and §4 is an axis it does not cover | drafted |
| **Alexandre Marques** | `almarque@redhat.com` | [2605.15051](https://arxiv.org/abs/2605.15051) | **cs.LG**, cs.CE, cs.AI, cs.CL, cs.PF | Senior author, same paper; Red Hat ships vLLM, so F023/F024 are operational for him | drafted |
| **Jongseok Park** | *(not published)* | [2601.11580](https://arxiv.org/abs/2601.11580), [2406.14066](https://arxiv.org/abs/2406.14066) | cs.AI, cs.PF, **cs.LG** | Coauthor on **both** papers we cite | need address |
| **Lanxiang Hu** | *(not published)* | [2406.14066](https://arxiv.org/abs/2406.14066) | cs.AI, cs.CL, **cs.LG**, cs.PF, cs.CV | TurboSpec coauthor | need address |
| **Xin Cheng** | *(not published)* | DSpark | cs.AI, cs.CL, cs.IR, **cs.LG** | DSpark benchmarks against DFlash — the exact drafter our §4 localises to | need address |

**Checked and NOT qualified** — do not email for this purpose:

- Shikhar Shukla (SpecKV, 2605.02888), Jiaxiang Yu (2601.11580), Yudi Zhang,
  Jerry Kaplan, Run Wang — lookup returns "not currently an endorser"
- Talor Abramovich (SPEED-Bench first author) — endorses cs.AI, cs.CL, cs.DC
  but **not cs.LG**. Qualified-looking is not qualified; check the category.
- Ion Stoica, Alvin Cheung, Woosuk Kwon, Zhuohan Li, Megan Flynn, Michael
  Peng, Nir Shavit, Mark Kurtz, Xu Han — not registered as owners of those
  papers, so the lookup cannot confirm them

Hit rate on "looks plausible" was roughly one in two. Three names on an
earlier version of this list were in the first group.

## Order to ask

Originally one at a time, a week apart. Revised 2026-09-25 to parallel asks
after four days of silence from the first: the arXiv target is 15 November,
and serialising three asks at a week each spends a month on a step that needs
one reply.

Send individually, never cc'd as a group. Each draft says plainly that
others were asked in parallel, so nobody discovers it sideways, and each
gives the reader an explicit way to not bother ("if someone else has already
helped, ignore this"). Only one endorsement is needed.

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

## Draft: to Maor Ashkenazi

Written after reading SPEED-Bench's methodology section rather than its
abstract. The hook is that their 23% is a **bias** and ours is a
**dispersion**, and better inputs do not remove a dispersion.

> **Subject:** arXiv endorsement request (cs.LG) — run-to-run dispersion as a gap in SD benchmarking
>
> Hi Maor,
>
> I am an independent researcher working on LLM inference measurement. I need
> a cs.LG endorsement to post my first arXiv preprint; arXiv's lookup lists
> you as qualified.
>
> I am writing to you because I cite SPEED-Bench, and because I measured
> something your benchmark would not currently surface.
>
> You found that synthetic benchmarking overestimates SD throughput by about
> 23%, and built the Throughput split to remove that bias. I measured a
> different quantity on the same axis: on vLLM 0.29 with a DFlash drafter on
> an L4, repeated boots of a byte-identical configuration give a coefficient
> of variation of 13.92% in output throughput, while acceptance length does
> not move. Under --enforce-eager it falls to 1.44%. n-gram speculation shows
> 2.67%.
>
> The reason I thought it worth telling you specifically: your 23% is a bias
> and mine is a dispersion, and better inputs do not remove a dispersion. If a
> speculative point on a throughput-latency Pareto curve carries roughly 14%
> run-to-run spread, then a curve built from single measurements is partly
> tracing noise, and a result like "optimal draft length shifts with batch
> size" could mis-rank adjacent draft lengths without anything looking wrong.
> I do not know whether the effect appears on your hardware — I measure one L4
> and I localise it to draft-model speculation — but a benchmark without a
> restart protocol is one of the places it would go unseen.
>
> The paper is a measurement case study, not a general claim. It reports four
> confounds, two retractions of its own headline claims, and ships the raw
> records with the scripts that audit them.
>
> Paper: [attach paper.pdf]
> Code and data: github.com/akshathtiwari/spec-fp8-study
>
> If you are willing, approval is at https://arxiv.org/auth/endorse?x=IGNP9M
> and takes about two minutes. As I understand it, endorsement means only that
> the work belongs in the archive, not that you are reviewing or vouching for
> it.
>
> I am asking a few people in parallel since only one endorsement is needed —
> apologies if this reaches you after someone else has helped.
>
> Either way, thanks — SPEED-Bench's AR/AL separation was useful to me, and I
> adopted the distinction after a reviewer pointed out I had been calling tau
> a rate.
>
> Akshath Tiwari
> github.com/akshathtiwari

## Draft: to Alexandre Marques

Deliberately different from the Kong draft, because they are coauthors on the
same paper and near-identical mail to both would read as a template. Kong's
angle is the latency model's residual; this one is operational, because Red
Hat ships vLLM to people who have to run it. It also states upfront that Kong
was contacted first — coauthors talk, and being told is better than finding
out.

> **Subject:** arXiv endorsement request (cs.LG) — reproducibility of vLLM speculative throughput
>
> Hi Alexandre,
>
> I am an independent researcher working on LLM inference measurement. I need
> a cs.LG endorsement to post my first arXiv preprint; arXiv's lookup lists
> you as qualified.
>
> In fairness I should say upfront that I wrote to Linghao a few days ago
> about the same paper, since you are coauthors. I am writing to you as well
> because your angle on it is different, and because Red Hat ships vLLM to
> people who have to operate it.
>
> The measurement: on vLLM 0.29 with a DFlash drafter on an L4, repeated boots
> of a byte-identical configuration give a coefficient of variation of 13.92%
> in throughput, while acceptance length stays put. Under --enforce-eager it
> falls to 1.44%, and every eager boot beat every default boot.
>
> Two things fell out of that which are operational rather than academic:
>
> - --enforce-eager cannot boot a speculative configuration with fp8_e4m3 KV
>   on a 22 GiB card at all. Disabling CUDA graphs skips a memory reservation,
>   so the engine hands that memory to the KV cache and an 892 MiB float32
>   logits buffer then has nowhere to go. It fails identically across three
>   separate containers.
> - --enforce-eager costs FP8-weight configurations between a fifth and a half
>   of their throughput, and costs bf16 essentially nothing.
>
> So the flag people reach for to stabilise speculative benchmarks is not
> free, and how much it costs depends on weight precision in a way I have not
> seen documented.
>
> It is a case study on one card and one engine release, and says so. It
> carries two retractions of its own headline claims, and ships the raw
> records with the scripts that audit them.
>
> Paper: [attach paper.pdf]
> Code and data: github.com/akshathtiwari/spec-fp8-study
>
> If you are willing: https://arxiv.org/auth/endorse?x=IGNP9M, about two
> minutes. Endorsement as I understand it means the work belongs in the
> archive, not that you have reviewed it. If Linghao has already helped,
> please ignore this.
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
- No two drafts are interchangeable. Kong and Marques are coauthors and
  Abramovich and Ashkenazi are coauthors; a template sent to both halves of a
  pair is visible the moment they compare notes.
- Since the switch to parallel asks, every draft says so and gives the reader
  permission to ignore it. Asking several people quietly is the version that
  costs goodwill; asking several people openly is just logistics.
- Where the recipient built something we cite, the mail tells them something
  about their own work they can check. That is the only part of the mail with
  any value to them.
