# Phase 3 — First Dataset Batch (18 examples) and a Corpus Gap-Fill

**Run:** 2026-09-02 · corpus **432 → 502 chunks**, 24 → 27 documents · dataset **0 → 18 examples**

## What happened

Drafted 18 natural-phrasing questions across the project's 6 target topics (workplace
change, insurance, wages, re-entry, alien registration/stay, contract renewal, E-9→E-7-4
transition) and ran them through the Phase 2 retrieval method. Most did **not** cleanly
retrieve the right source in the real top-5, even for topics the corpus does cover —
confirmed by hand: the correct article often existed in the corpus but ranked outside
top-5 for these phrasings (e.g. EPS Act Article 25's workplace-change eligibility list
never appeared in top-5 for 3 different natural phrasings of "can I change jobs if my
employer treats me unfairly").

**This does not mean production retrieval is broken** — Phase 2 already measured that at
~80% hit@5, separately. It means: **building good training data needs a higher bar than
trusting automatic top-5**, the same way the original 18-example seed set was almost
certainly hand-verified rather than pulled from raw retrieval. Every grounded example in
this batch has its context read and confirmed by hand (grep/targeted search against the
corpus), not assumed from similarity score.

## Corpus gap-fill (3 new sources)

Checking coverage before writing content surfaced 4 genuinely missing facts. Added 3 new
official sources (`data/sources.json`) to cover them, same pattern as Phase 2b:

| Gap | Source added | Result |
|---|---|---|
| Overtime pay rate | `labor_standards_act_eng` (law.go.kr, the full Act — only its Enforcement Decree was in the corpus before) | Found: Article 56, 50% premium minimum |
| Unpaid-wage complaint office | `wage_arrears_easylaw` (easylaw.go.kr) | Found: local Employment and Labor office |
| Health insurance premium rate | `health_insurance_premium_easylaw` (easylaw.go.kr) | **Did not work** — see below |
| Lost registration card reissue | *(not added)* | No official source found by search; still a gap |

The health-insurance page extraction failed: the fetched URL turned out to be a
navigation/table-of-contents page, not the actual premium-calculation content — the
"statute" extractor pulled only menu items. Needs the correct sub-page URL, not a fix to
this batch. Left as a refusal example for now rather than guessing.

## The 18 examples

**10 grounded, 8 refusal (44% refusal rate)** — much higher than the ~15% Phase 3 targets,
because this batch deliberately tested corpus coverage across many topics rather than
optimizing for a natural mix. Contract renewal and E-9→E-7-4 transition remain almost
entirely uncovered by the corpus (consistent with the existing seed set, where contract
questions were already refusal-only) — those two topics may need their own source-finding
pass before they can carry real grounded weight in the full dataset.

Saved to `data/samples/phase3_dataset.jsonl`. Build scripts:
- `scripts/phase3_retrieve.py` — automatic top-5 retrieval (diagnostic, not used to
  select final context)
- `scripts/phase3_build_dataset.py` — the actual dataset, hand-verified context and answers

## Next

1. **Native-reader check** (you) — Korean-term accuracy, factual accuracy, tone, on this
   18-example batch, before writing more.
2. Decide: fix the health-insurance source, and look harder for a lost-card source, before
   or after scaling up.
3. Decide whether contract renewal / E-9→E-7-4 transition need their own targeted source
   search, or should stay mostly-refusal topics in the real dataset (which may itself be
   the honest, correct answer for those two).
4. If the batch looks right, continue building toward 800-1500, in further batches.
