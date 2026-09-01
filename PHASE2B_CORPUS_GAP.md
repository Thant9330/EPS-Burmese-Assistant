# Phase 2b — Closing the Corpus Gap (and fixing the eval that hid it)

**Run:** 2026-09-01 · corpus **402 → 432 chunks**, **7 → 24 documents**

## What was wrong

Phase 2 found that "alien registration" and "re-entry" questions missed in *both* languages —
a corpus problem, not a language problem. The statutes state principles; the procedure
(who must register, which exemptions, what deadline) lives in HiKorea guidance.

## What was added

17 HiKorea procedural pages (`hikorea.go.kr/info/InfoDatail.pt?CAT_SEQ=<n>&locale=EN`),
including Foreign Resident Registration, Re-entry Permit, Change/Addition of Workplace,
Extension of Stay, Status Change, Reporting Obligations, and Immigration Offender pages.

**The extraction problem:** every HiKorea page embeds the same large sidebar nav, so naive
extraction returned ~90% boilerplate — pages were 4,900–6,200 chars of which only ~800–2,700
were unique. Fix: fetch the pages as a **group** and drop any line appearing on *every* page.
Requires ≥3 pages, otherwise genuinely shared content would be eaten. This is now a second
extractor (`hikorea_info`) alongside the existing `statute` one, in the same builder.

`hikorea_528` was rejected by the `min_chars` guard (353 chars — the invalid-page signature).
`immigration.go.kr` robots.txt was checked before use; it disallows `/bbs/`, so an `artclView`
link found there was skipped.

## Did it work? Yes — but the eval said no

Re-running the Phase 2 eval unchanged produced an apparent **regression**: Burmese hit@5 fell
80% → 75%. That was wrong. Direct rank inspection, which does not depend on labels:

| Question | Target page | MY rank | EN rank |
|---|---|---:|---:|
| q11 alien registration | `hikorea_176` | **1** | **1** |
| q02 workplace change | `hikorea_189` | **1** | **1** |
| q12 re-entry | `hikorea_7203` | 8 | **1** |

The newly added pages rank **first** for exactly the questions that used to fail. The eval
scored q11 as a miss in both languages anyway, because its `gold_terms` said *"alien
registration"* while HiKorea writes *"Foreigner Registration"*. **A correct rank-1 retrieval
was scored as a failure on a vocabulary mismatch.**

## The real lesson: the eval was the weak link

Phase 2 flagged weak labels as a caveat. Once the corpus vocabulary diversified, that caveat
became the dominant error source. Both labelling schemes tried so far are bad in opposite ways:

| Labelling | Burmese hit@5 | English hit@5 | Problem |
|---|---:|---:|---|
| keyword `gold_terms` | 75% | 90% | too strict — vocabulary-dependent false misses |
| whole-document `gold_sources` | 100% | 90% | too loose — q05 has 167/432 chunks gold |

Neither number should be quoted as retrieval quality. `gold_sources` is now stored alongside
`gold_terms` in `data/eval/phase2_questions.json`, but the honest position is that **chunk-level
relevance needs human judgement**, which belongs in Phase 6 eval design.

What survives both schemes, and the direct rank check:
- the corpus gap is closed
- Burmese-query → English-corpus retrieval works
- English still ranks gold slightly higher on average (MRR 0.850 vs 0.764)

## Also added

Embedding cache (`data/eval/embcache.npz`, gitignored) keyed by chunk-text hash. CPU encoding
of 432 chunks takes ~35 minutes; re-runs now encode only genuinely new chunks. This should have
existed before the first run.

## Next

Check whether a newer base model supersedes SEA-LION v3 / Gemma 2 (Gemma 4 and newer SEA-LION
releases exist), then the training spike before committing to full Phase 3 dataset work.
