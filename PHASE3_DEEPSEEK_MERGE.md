# Phase 3 — Merging DeepSeek-generated batches

**Run:** 2026-09-02 · dataset **61 → 440 examples** (2 merge rounds + dedup) · corpus
**502 → 602 chunks** (2 new sources)

## Round 2 (same day, 9 more files / 350 rows)

The user kept adding files while round 1 was being processed. Re-ran the same citation
check (`scripts/phase3_verify_batch.py`, now a reusable script — see below) against just
the 9 new files: **the exact same 33 mismatch patterns showed up again**, confirming
DeepSeek's wrong EPS-Act/Immigration-Act numbering is a consistent, repeatable error in
its training data, not a one-off. Reused the identical `RELABEL`/`DROP_PATTERNS` from
round 1 — kept 250/350 (71%). Also cross-checked 6 more `hikorea_e74` rows against the
user's verified E-7-4 facts, all passed.

**New issue found: cross-batch duplicate questions.** The two DeepSeek sessions covered
overlapping ground (mostly deep Labor Standards Act detail — night work, maternity leave,
overtime, dismissal notice). 72 questions ended up asked twice (a few three times) across
the merged set. None were contradictory this time (same kind, similar answers) — but
training on the identical input with two different gold answers isn't ideal even when both
are individually correct, so deduped down to one answer per unique question (kept the
longer/more complete one). **74 rows dropped**, final count 440.

## Round 1 (below, unchanged)

## What happened

The user generated 300 candidate examples with DeepSeek, following `PHASE3_EXTERNAL_AI_PROMPT.md`.
Format compliance was excellent — exact schema, exact disclaimer/refusal text, correct
Korean-term-glossing convention, right topic mix (including two topics we'd flagged as
gaps: severance pay and E-9→E-7-4).

**Citation accuracy was a real problem, not a cosmetic one.** Checking every row's cited
article number against the real text (fetched directly from law.go.kr/elaw.klri.re.kr,
same as every other source in this project) found **155 of 230 checkable rows cited an
article that doesn't say what the row claims**. Example: it labeled EPS Act Articles 21-24
as the four insurance types (departure guaranty, wage guarantee, personal injury, return-
home expense) — but the real Article 21 is "Projects Related to Foreign Workers," and the
actual insurance articles are 13, 15, and 23 (which we'd already built correctly in round 3).
The pattern looks systematic — like DeepSeek is working from an older/different numbering
of the EPS Act rather than randomly guessing — but the effect is the same: a citation a
worker or reviewer could not actually verify against the real law.

## What was done about it

1. Fetched 2 more real sources to check the genuinely-new claims it made: **Guarantee of
   Workers' Retirement Benefits Act** and **Industrial Accident Compensation Insurance
   Act** (law.go.kr). This also answered a question we'd left as a refusal ourselves —
   Article 10 of the Industrial Accident Act establishes COMWEL as the administering body.
2. Checked every row citing a source we already had against the real article heading.
3. **~10 rows/patterns**: same real topic, just a paraphrased title or an adjacent article
   number — relabeled to the real citation, content kept (it was accurate, just mis-cited).
4. **12 rows**: duplicated our own already-correct insurance content — dropped, no value added.
5. **87 rows**: cited a genuinely different real article than claimed — dropped rather than
   guessed at. This is exactly the accuracy problem the whole redo exists to prevent.
6. **10 rows** cited a source we don't have (a specific HiKorea E-7-4 page that doesn't seem
   to have an official page we could find) — cross-checked against the facts the user
   personally verified earlier instead of dropping them outright; all 10 passed.

**Net: 201 of 300 rows kept (67%)**, merged in as `d501`-onward. Final dataset: 262 examples,
227 grounded / 35 refusal (13%).

Script: `scripts/phase3_merge_deepseek.py` — the verification logic and the specific
relabel/drop decisions are documented in its docstring and the `RELABEL`/`DROP_PATTERNS`
tables, so the reasoning is traceable, not just the result.

## Takeaway for future batches

The prompt format worked well — reuse it as-is. The failure mode wasn't the Burmese
language quality (that looked fine throughout, including in dropped rows) — it was
specific legal citations from an AI without real document access. **Any future batch from
an AI without browsing/document access needs this same citation-verification pass before
merging** — don't trust "Article N" claims without checking them against the real fetched
text, the same discipline this project has used since Phase 1.
