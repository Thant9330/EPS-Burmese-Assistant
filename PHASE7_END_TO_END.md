# Phase 7 — End-to-end evaluation with real retrieval

**Run: 2026-09-04 · 50 held-out questions · fine-tuned adapter (unmerged) · bge-m3 retrieval**

Every earlier evaluation handed the model the **correct** law text from the dataset's
`context` field. That measures "given the right passage, does it answer well?" — not what
a worker experiences. This phase closes that gap: real retrieval over the 602-chunk
corpus, whatever it returns fed to the model.

## Retrieval accuracy

Source-level match (did any retrieved chunk come from the document the gold answer cites):

| | hit@1 | hit@5 |
|---|---|---|
| all evaluable rows (n=44) | 55% | 77% |
| grounded only (n=39) | 54% | 79% |

Consistent with Phase 2's 80% hit@5, so the encoder is behaving as documented.

**A tag-naming mismatch had to be fixed first.** The dataset and corpus use different
identifiers for the same documents, which made the raw number look far worse (53%):

```
lba_eng          -> labor_standards_act_eng
ia_eng           -> industrial_accident_act_eng
gwra_eng         -> retirement_benefits_act_eng
minwage_act_eng  -> minimum_wage_act_eng
```

This aliasing is a real project inconsistency, not just a measurement artifact — anything
that checks citations against the corpus will break on it.

**Three questions are unanswerable**: `d557`, `d558`, `d589` all cite `hikorea_e74`, which
was never ingested into the corpus. No retrieval setting fixes that.

## Generation quality end to end

| all 50 | loops | stops | cites | Korean | len |
|---|---|---|---|---|---|
| gold context (ceiling) | 1/50 | 44/50 | 34/45 | 18/30 | 252 |
| retrieved top-1 | 1/50 | 42/50 | 25/45 | 12/30 | 238 |
| retrieved top-5 | 0/50 | 35/50 | 30/45 | 15/30 | 292 |

**Top-5 beats top-1**, contradicting the prediction that an unseen multi-chunk prompt
shape would hurt. On the 40 rows where retrieval found the right source:

```
gold context     34 answer /  7 refuse
retrieved top-5  34 answer /  6 refuse   <- matches the ceiling
retrieved top-1  25 answer / 15 refuse   <- refuses far too often
```

Top-1 fails because a source-level hit is not a passage-level hit: one chunk from a
100-chunk statute usually is not the relevant one, and the model correctly declines.
**Serve top-5, not top-1.**

## The safety finding

On the 10 questions where retrieval returned the wrong law, judged by hand:

| behaviour | n |
|---|---|
| refused safely | 4 |
| answered acceptably | 2 |
| **confidently wrong** | **4** |

The four failures all arrive formatted like good answers, with a `ရင်းမြစ် —` source line
and an article number:

- `d779` — asked how severance is calculated; cites `근로기준법` Article 2 and calls
  severance pay "shutdown allowance". Wrong law, wrong concept.
- `p311` — asked whether failing to register can mean deportation; answers "yes" (correct)
  but cites departure-cost insurance law. Right conclusion, entirely wrong reasoning.
- `p108` — asked about workplace-transfer conditions; answers about weekly overtime limits.
- `p308` — asked about staying abroad 1–2 years; answers about extending a residence permit.

**Roughly 20% of questions get wrong context, and about 40% of those become confident
wrong answers — on the order of 8% of all questions.** For legal guidance aimed at migrant
workers this is the number that governs deployment risk, not the citation rate.

Note the refusal detector used earlier undercounted: several owner-written refusal
variants carry no မ-negation (e.g. `မှန်ကန်သော ... 고용센터 သို့ ဆက်သွယ် မေးမြန်းပေးပါရန်`).
Detection was corrected to "names a referral office AND asks the user to contact it".

## A confidence gate does not work

Testing "refuse when top-1 similarity < t":

```
miss rows: similarity mean 0.585  (0.543 - 0.665)
hit rows:  similarity mean 0.614  (0.494 - 0.717)
33 of 40 good rows score below the worst-scoring miss
```

At the only threshold catching 3 of the 4 bad answers (t=0.60), 18 of 40 good answers are
lost. The distributions overlap almost entirely. **The model cannot tell it has been given
the wrong law, and neither can the retriever's own score.** This is architectural.

## Diverse retrieval — the fix that does work

The misses share a pattern: big documents crowd out small ones. HiKorea pages are 1–2
chunks; the statutes are 60–100, so they have far more chances to match, and slots get
spent on duplicates from one document.

Capping at one chunk per source:

| strategy | hit@5 |
|---|---|
| plain top-5 (current) | 77% |
| diverse top-5 (1/source) | **82%** |
| diverse top-5 (2/source) | 80% |
| diverse top-8 (1/source) | **93%** |

Diverse top-5 newly finds `p108` and `d707` and loses nothing — strictly better at
identical context length, so it is a free upgrade. Diverse top-8 reaches 93% but at a
context-length cost (below).

Generation under diverse retrieval was **not** measured — the GPU quota ran out before the
first checkpoint. The retrieval gain is established; whether it reduces the
confidently-wrong count from 4 remains open.

## Context length mismatch

```
TRAINING prompts:   median   199 tokens (max 326)
production top-5:   median 2,589 tokens
production top-8:   median 3,924 tokens
```

Production prompts are ~13x longer than anything seen in training, and already exceed the
`max_length=2048` used for training (the model supports 8192, so they fit — they are just
unfamiliar). Training contexts were short focused excerpts of roughly 100-150 tokens;
corpus chunks are ~500 tokens each.

That the model degrades only mildly under a 13x shift is the surprising part. But it means
any future retrain should train on realistic retrieved contexts, not curated excerpts.

## What this changes

1. **Serve diverse top-5**, one chunk per source. Free accuracy, no retraining.
2. **Ingest `hikorea_e74`** and re-check corpus coverage — 3 of 50 held-out questions have
   no answerable source.
3. **Normalise source tags** between dataset and corpus, or ship the alias map.
4. **Accept an ~8% confident-wrong rate** as the current risk level, or mitigate it in
   product design rather than in the model — the model and the retriever both fail to
   detect the condition.

## Artifacts

Saved to the Drive checkpoint folder:

- `full50_base_vs_tuned.json` — gold-context eval, base vs tuned, all 50
- `end2end_retrieved.json` + `.BACKUP.json` — 100 generations, top-1 and top-5
- `checkpoint_sweep_results.json` — the earlier (merged, invalid) sweep, kept for the record
