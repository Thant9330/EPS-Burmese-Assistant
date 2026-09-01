# Phase 2 — Retrieval Smoke Test Results

**Run:** 2026-09-01 · `scripts/phase2_retrieval_smoketest.py` · encoder `BAAI/bge-m3` · k=5
**Corpus:** 402 chunks (Phase 1) · **Questions:** 20 real E-9/EPS questions, 17 distinct topics

## The question this answered

The architecture assumes corpus + retrieval in **English**, generation in **Burmese** — chosen
specifically because Burmese embedding support looked unreliable. That left one open question:
can BGE-M3 match a **Burmese** query against **English** chunks directly, or must the query be
translated to English first?

Two conditions, identical corpus / encoder / labels:

- **A** — Burmese question embedded directly (what production does, no MT)
- **B** — human English translation (the **ceiling** any MT pipeline could reach)

## Headline

| Condition | hit@5 | MRR | prec@5 |
|---|---:|---:|---:|
| A) Burmese query → EN corpus | 80.0% | 0.647 | 0.410 |
| B) English query → EN corpus | 90.0% | 0.735 | 0.530 |
| **gap (B − A)** | **+10.0%** | +0.088 | +0.120 |

**BGE-M3 handles Burmese cross-lingually.** The concern that drove this test — that Burmese is
too low-resource for the embedding side — was largely unfounded. Burmese queries land a relevant
chunk in the top 5 four times out of five, with no translation step at all.

## Three things the headline hides

**1. The 90% is unreachable in production.** Condition B used *human* translation. A real
Burmese→English MT step (NLLB) would score somewhere below it. So the true benefit of adding
MT is *less* than 10 points — bought at the cost of another model, more latency, and a new
failure mode.

**2. Weak labels inflate both numbers.** Relevance here is "chunk contains a gold phrase", not
human judgement. Some gold terms are far too generic:

| Question | gold chunks | why |
|---|---:|---|
| q18 employer_report | 144 | "report" / "notify" appear everywhere |
| q05 insurance_premium | 110 | "premium" is ubiquitous |
| q09 industrial_accident | 85 | broad term |

Hitting those is close to automatic. On the **7 discriminating questions** (fewest gold chunks),
the picture is sharper and worse:

```
Burmese  3/7        English  5/7
```

So the real language gap is wider than +10 points — roughly 43% vs 71% where it counts.

**3. Two questions fail in BOTH languages — that is a corpus problem, not a language problem.**

| Question | gold | MY | EN | what came back instead |
|---|---:|---|---|---|
| q11 alien registration | 6 | miss | miss | Immigration Act chunks, wrong ones |
| q12 re-entry | 7 | miss | miss | Immigration Act chunks, wrong ones |

Both retrieve *plausible* Immigration Act material and still miss the gold. Translating the
query would not have helped. This is exactly the gap Phase 1 predicted: statutes carry the
principle, but the procedural detail lives in eps.go.kr / hikorea.go.kr guidance that is
not yet ingested.

## Decisions

1. **Keep the English-corpus + BGE-M3 design.** Confirmed working. No Burmese embedding model
   needed, which was the original worry.
2. **Do not add a translation step yet.** The measured upper bound is +10 points, real MT would
   capture less, and it adds a dependency and a failure mode. Revisit only if Phase 6 shows
   retrieval is the binding constraint.
3. **Corpus coverage is the bottleneck, not language.** The highest-value next work is ingesting
   practical/procedural sources (eps.go.kr, hikorea.go.kr — both robots-permitted), not building
   an MT pipeline.
4. If retrieval later needs a boost, try **dual-query** (embed Burmese *and* English, union the
   hits) before committing to an MT stage.

## Caveats

- **n=20 with weak labels.** Directional only. q19 scored better in Burmese (rank 1) than English
  (rank 5), which is noise at this sample size — do not over-read individual rows.
- **`gold_terms` need tightening** before this becomes a real eval. Generic terms should be
  replaced with article-level references.
- **CPU encoding is impractical at scale.** 402 chunks took ~35 minutes on CPU-only torch
  (~2,000 CPU-seconds). Any larger run belongs on Colab GPU.

## Next — Phase 3

Dataset build (~800–1500 pairs), the phase that carries ~80% of the project's real work.
Two findings above feed straight into it: the corpus gap (q11/q12) should be closed first,
and the `license_note` on every chunk must surface in training answers so the model never
presents an unofficial translation as binding law.
