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

## The 18 examples — update: user filled in the 8 gaps

First pass: 10 grounded, 8 refusal. Reviewed by the user (native Burmese reader) — the
Burmese was judged mostly accurate. For the 8 refusals, the user supplied the correct
answers directly (they know the current official rates/rules — health insurance 7.19%
split 50:50, employment insurance 0.9% each side, overtime/wage-check channels, lost-card
reissuance procedure, contract-renewal dispute channels, and the E-7-4 skilled-worker
visa's K-Point criteria including the 4-year work history requirement).

These were folded in as **grounded**, with `context` marked `[user-verified]` rather than
a scraped document — honest about the source being a confirmed fact from the native-speaker
reviewer, not a corpus excerpt. This is now **18/18 grounded, 0 refusal** for this batch.

**This batch is no longer representative of the target ~15% refusal rate** — it was
deliberately built to stress-test coverage across many topics, and every gap it found got
closed. The next batches should include real, natural refusal examples (questions the
corpus genuinely doesn't cover) so the model still learns to decline rather than invent.

Saved to `data/samples/phase3_dataset.jsonl`. Build scripts:
- `scripts/phase3_retrieve.py` — automatic top-5 retrieval (diagnostic, not used to
  select final context)
- `scripts/phase3_build_dataset.py` — the actual dataset, hand-verified/user-verified
  context and answers

## Next

1. Continue building toward 800-1500 in further batches, now that this first batch is
   confirmed good.
2. Keep including genuine refusal examples in future batches (~15% target) — this batch's
   0% is an artifact of gap-closing, not the real target ratio.
3. The health-insurance-rate *source page* is still broken (pulled a nav menu, not
   content) — the fact itself is now covered via the user's answer, but if we want a
   citable document instead of `[user-verified]`, the right sub-page still needs finding.
4. Lost-card reissuance and contract-renewal/E-7-4 facts are now covered by user-verified
   answers rather than corpus documents — fine for training data, but worth finding real
   citable sources eventually for full provenance.
