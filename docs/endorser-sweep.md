# Endorser sweep — papers to verify, then contact

**Endorsement code `IGNP9M`** · approve at https://arxiv.org/auth/endorse?x=IGNP9M

## Do these in this order

1. **Verify.** Open `https://arxiv.org/auth/show-endorsers/<id>` for each paper below.
   One click, needs your logged-in arXiv session. It tells you which authors
   are registered endorsers and **for which categories** — you need **cs.LG**.
2. **Then find the address**, only for authors who passed. Most papers no
   longer print emails (1 of 4 did in this sweep), so the reliable source is
   the author's personal homepage, not the PDF.
3. **Then send**, using the template at the bottom with a one-line hook.

Doing it the other way round wastes hours chasing addresses for people who
turn out not to qualify.

## Who to pick once a paper passes

**Target down, not up.** A fourth-year PhD student with three cs.LG papers
endorses exactly as well as a famous last author, replies far more often, and
publishes their email because they want collaborators. Prefer the **second
through fourth** names on a paper. Skip the last author unless nothing else
qualifies.

---

## Tranche 1 — closest topic fit

| # | arXiv | paper | cat | note |
|---|---|---|---|---|
| 1 | `2604.19157` | SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving | cs.LG | 11 authors. Jinda Jia, Jisen Li, Zhongzhu Zhou, Jung Hwan Heo, Jue Wang, Tri Dao, Shuaiwen Leon Song, Ben Athiwaratkun, Chenfeng Xu, Tianyi Zhang, Xiaoxia Wu. Best single target: KV quant *under serving constraints*, your exact axis |
| 2 | `2609.21704` | SpecQuant: Speculative Decoding with Multi-Parent Quantization | cs.LG | Harish KB, Jagadeeswaran M, Pradheep P, Yuvanesh S, Sivakumar T. Spec decoding × quantization, i.e. your combination |
| 3 | `2608.13756` | The Integer Alibi: Localizing Cross-Kernel Divergence in INT8 | cs.LG | Teng-Ruei Chen, `luka@krixvon.com` **(email known)**. Same SM89 architecture. Single author, may not clear the 3-paper bar |
| 4 | `2605.19537` | The Silent Hyperparameter | cs.LG | CISPA. `{david.pape, jonathan.evertz, schoenherr}@cispa.de` **(emails known)**. You cite them; your §4 is the throughput axis of their null |
| 5 | `2604.09603` | ECHO: Elastic Speculative Decoding with Sparse Gating | — | high-concurrency spec decoding |
| 6 | `2607.01831` | Lynx: Progressive Speculative Quantization for KV Transfer | cs.DC | Wenchen Han (UCL) + Huawei. No emails printed |
| 7 | `2603.03251` | Speculative Speculative Decoding | — | Tanishq Kumar, Tri Dao, Avner May |
| 8 | `2603.11053` | Speculative Decoding Scaling Laws | cs.CL, cs.IT, cs.LG | Bozorgkhoo, Molybog |

## Tranche 2 — serving and measurement

| # | arXiv | paper | cat | note |
|---|---|---|---|---|
| 9 | `2609.16085` | Is INT8 Portable? Cross-Platform Measurement Study | cs.AR, cs.LG, cs.PF | Yuyeong Shin. Cross-hardware measurement, same shape as your A10 result |
| 10 | `2603.11340` | Black-Box Online Tuning: Adding System Specs to Factsheets | — | aimed squarely at your thesis |
| 11 | `2605.24217` | Systemic Measurement Bias in Production LLM Benchmarks | cs.AI, cs.DC | Google. `achandrasekar@google.com`, `jkramberger@google.com` **(emails known)**. Note: **no cs.LG on this paper** |
| 12 | `2607.02525` | PEEK: Predictive Queue-Informed KV Cache Management | cs.DC | Bing Xie, Zhipeng Wang, Masahiro Tanaka, Zheng Zhen |
| 13 | `2604.09562` | StreamServe: Adaptive Speculative Flows | — | |
| 14 | `2608.24004` | AgentSpec: Speculative Decoding for Batch Agent Inference | — | |
| 15 | `2607.08057` | Survey on System-Aware KV Cache Optimization | — | surveys have many authors; good yield per lookup |
| 16 | `2603.20397` | KV Cache Optimization Strategies for Scalable LLM Inference | — | |
| 17 | `2601.03067` | Joint Encoding of KV-Cache Blocks | — | |
| 18 | `2604.04722` | Don't Waste Bits! Adaptive KV-Cache Quantization | cs.CV | Clemson. `shaerib@`, `nmehrab@`, `pnwoods@`, `ghilles@`, `arazi@g.clemson.edu` **(all emails known)**. Caution: **cs.CV, not cs.LG** |
| 19 | `2607.21804` | Adversarial Prompts for Acceptance Collapse in Spec Decoding | cs.CR, cs.CL, cs.LG | 10 authors. Run Wang already checked and is NOT an endorser; the other nine are unchecked |

## Already done — do not redo

| person | status |
|---|---|
| Linghao Kong, `linghao@mit.edu` | sent 24 Sep |
| Maor Ashkenazi, `mashkenazi@nvidia.com` | sent 25 Sep |
| Xiaoxuan Liu, `xiaoxuan_liu@berkeley.edu` | sent 25 Sep |
| Alexandre Marques | address bounced, dropped |
| Jongseok Park, Lanxiang Hu | verified cs.LG, **no address found** — homepage 404s / not published |
| Xin Cheng (DSpark) | verified cs.LG, but it is a 33-author DeepSeek paper. Low reply odds |
| Shukla, Jiaxiang Yu, Yudi Zhang, Kaplan, Run Wang | checked, **not endorsers** |
| Talor Abramovich | endorses cs.AI/cs.CL/cs.DC, **not cs.LG** |

---

## The draft

Short on purpose. The only part that should change per person is the one
bracketed line. If you can't write that line truthfully for someone, they are
the wrong person to email.

> **Subject:** arXiv endorsement request (cs.LG)
>
> Hi [name],
>
> I'm an independent researcher working on LLM inference measurement. I need a
> cs.LG endorsement to post my first arXiv preprint and arXiv's lookup lists
> you as qualified.
>
> I'm writing to you because [one specific line: what of theirs you read, and
> what in your work it connects to].
>
> The work is a measurement study of speculative decoding under FP8 on vLLM:
> four confounds, two retractions of my own headline claims, and the raw
> records with the scripts that audit them.
> https://doi.org/10.5281/zenodo.23033168
>
> If you're willing, it's about two minutes:
> https://arxiv.org/auth/endorse?x=IGNP9M (code `IGNP9M`). As I understand it
> endorsement means only that the work belongs in the archive, not that you're
> reviewing or vouching for it.
>
> I'm asking a few people in parallel since only one endorsement is needed, so
> apologies if this reaches you after someone else has helped.
>
> Thanks either way,
> Akshath Tiwari

### Notes

- Lead with the DOI, not the GitHub link. Archived work reads differently from
  a side project.
- Keep "asking a few people in parallel". It costs nothing and it is true.
- Do not attach the PDF to a cold email; the link is enough and attachments
  trip spam filters.
- One nudge at day 10–14, two sentences, no new argument. Then stop.
- Target 12–15 asks total. At ~15–20% conversion that is a >90% chance of at
  least one yes. A hundred generic sends converts worse and risks an arXiv
  spam complaint against your own submission.
