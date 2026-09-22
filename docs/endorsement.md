# arXiv endorsement — who, when, and what to send

**rev 1 · 2026-09-23.** Prepared ahead of the conversation so the draft
exists to react to rather than to be written under time pressure.

---

## 1. How the system actually works

arXiv requires a first-time submitter to a given archive to be **endorsed**
by someone who already has posting privileges in that archive. Points worth
being precise about, because getting them wrong wastes an ask:

- **Endorsement is per-archive, not per-person.** An endorser for `cs.LG`
  must themselves have posting standing in `cs.LG`. A professor who
  publishes in, say, `cs.SE` or a non-arXiv venue cannot endorse a `cs.LG`
  submission even if they are senior and willing.
- **The endorser does almost nothing.** You submit first and arXiv issues
  you a **six-character endorsement code** tied to your account. You send
  them the code and a link; they open the arXiv endorsement page, enter it,
  and click approve. It is roughly a two-minute action.
- **The endorser is not vouching for correctness.** arXiv's own framing is
  that endorsement means the work is appropriate for the archive, not that
  it is correct or that the endorser has reviewed it. Saying this explicitly
  in the ask lowers the cost of saying yes, because the most common
  hesitation is "I don't have time to review this."
- **Many institutional accounts are auto-endorsed.** If you can register
  with a `.edu`/recognised institutional email that has prior activity, you
  may never need to ask. Check this first; it costs nothing.
- **Endorsement is one-time per archive.** Once through for `cs.LG`, future
  submissions there do not need it.

Verify the current rules at `arxiv.org/help/endorsement` before sending
anything. The thresholds have changed before and this document should not
be trusted over the source.

---

## 2. Can your college professors help?

Only if they post to `cs.LG` themselves, and recently. That is the whole
test. Before asking, check:

```
arxiv.org/a/<their_surname>_<initial>_1
```

or just search their name on arXiv and look at the **archive tags** on
their recent listings. If you do not see `cs.LG` (or `cs.AI`, `cs.CL`, with
`cs.LG` cross-lists) in the last couple of years, they cannot help with
this and asking costs you goodwill for nothing.

The honest read: for an ML systems preprint, a professor is usually a
*worse* target than the alternative below, unless they are actively
publishing in the area. Seniority is not the qualification here; recent
`cs.LG` posting is.

---

## 3. Better targets, in order

The work is an empirical measurement study of vLLM behaviour. The natural
endorsers are people whose own recent arXiv output is in exactly that space.

1. **Authors of the papers cited in the related-work section.** They post to
   `cs.LG` by definition, the topic overlap is obvious, and the ask is
   short. Best single option. Pick the ones whose work you actually
   position against, so the email is not generic.
2. **vLLM / SGLang contributors who publish.** Several maintainers post
   systems papers. There is already a live upstream issue thread from this
   study, which is a real, non-cold point of contact.
3. **Anyone at your workplace who posts to `cs.LG` personally.** Careful
   here: this must be a personal endorsement of a personal project. Nothing
   from the employer enters the artifact, and the ask should not imply the
   work is connected to them. (The provenance rule in `requirements.md` §1
   applies to the outreach too, not only to the data.)
4. **Alumni network / former professors who are now in industry research.**

---

## 4. When

- **Register the arXiv account now** and check whether you are
  auto-endorsed. Free, and may make the rest unnecessary.
- **Send asks in early October.** Far enough ahead of the 15 November target
  that a non-reply is survivable, and after the preprint is complete enough
  to link.
- **Ask two or three people, staggered.** Not a mass mail. If the first
  does not reply in a week, send the next.
- **Do not ask before the draft is postable.** The strongest version of this
  email links to a finished artifact. A promise reads worse than a repo.

---

## 5. Draft email

Plain, short, no build-up. Says what it is, what is being asked, and how
long it takes. Replace the bracketed parts.

> **Subject:** arXiv endorsement for cs.LG — measurement study on speculative decoding + FP8
>
> Hi [Name],
>
> I am an ML engineer working on LLM inference. I have written up an
> independent measurement study and I need a cs.LG endorsement to post it,
> since it would be my first arXiv submission. I am writing to you because
> [specific reason: "your paper on speculative decoding under quantization
> is the closest prior work and I position against it in the related work
> section" / "you maintain the speculative decoding path in vLLM and I filed
> issue #NNNNN against it"].
>
> The study measures speculative decoding under FP8 on vLLM 0.29 on an L4.
> The main result is not the speedup number, it is that three measurement
> artifacts each turn out to be large enough to produce that number on their
> own:
>
> - attention backend selection is conditioned on KV cache dtype, so a
>   precision A/B silently swaps the kernel, and the target and draft models
>   select backends independently
> - speculative throughput is bimodal across boots of an identical config,
>   up to 1.64x between two boots, selected by the CUDA graph path, while
>   acceptance does not move
> - tau has a boot to boot noise floor of 1.32% that I could not find
>   reported anywhere, and effects smaller than that have been published
>
> Everything is reproducible from the repo, including the raw records and
> the six corrections I had to make to my own results along the way.
>
> Paper: [link]
> Code and data: github.com/akshathtiwari/spec-fp8-study
>
> If you are willing, arXiv sends you a code and the approval takes about
> two minutes. As I understand it endorsement only means the work belongs in
> the archive, not that you are reviewing or vouching for it.
>
> Either way, thanks for reading.
>
> [Your name]
> [link to your GitHub or site]

### Why it is written this way

- **The reason for picking them is specific.** A generic endorsement email
  reads as a mass mail and gets deleted. One concrete sentence about their
  work is the difference.
- **The three bullets are the actual contribution**, not a summary of the
  topic. Someone in this field can tell in five seconds whether it is worth
  two minutes, and the bullets let them.
- **"Six corrections I had to make" stays in.** It is unusual, it is true,
  and to this audience it reads as competence rather than weakness. It is
  also the paper's thesis.
- **The effort is stated explicitly** ("about two minutes", "not reviewing
  it"). The usual reason for silence is an assumption that it costs more.
- **No apology, no flattery, no long preamble.** Nothing that has to be read
  twice.

### Things to avoid

- Do not attach a PDF. Link it.
- Do not mention your employer, or write anything that implies the work is
  connected to them.
- Do not say "it would mean a lot" or explain why you need it personally.
  The ask is administrative; keep it administrative.
- Do not send the same text to several people the same day.

---

## 6. If nobody endorses

Not a dead end.

- Some categories accept submissions without endorsement once an account has
  institutional standing. Re-check your own status.
- A co-author with posting standing removes the requirement entirely.
- The artifact is public regardless. The repo, the findings, and the
  upstream issue exist and are citable whether or not the preprint posts,
  and for MS applications a public, auditable repo plus a live upstream
  contribution is already evidence. arXiv is the preferred wrapper, not the
  substance.
