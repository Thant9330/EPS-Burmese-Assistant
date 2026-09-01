# Phase 1 — Corpus Build Results

**Run:** 2026-09-01 · `scripts/phase1_build_corpus.py` · manifest `data/sources.json`

## What exists now

**402 chunks · 199,961 tokens · 7 primary legal documents**, all English, each chunk carrying
full provenance (source URL, authority, retrieval timestamp, license note).

Chunked at 500 tokens with 50-token overlap using the **same tokenizer locked in Phase 0**
(`aisingapore/gemma2-9b-cpt-sea-lionv3-instruct`), so chunk sizes mean the same thing in
Phase 5 training as they do here. Mean chunk: 498 tokens.

Output `data/processed/corpus.jsonl` is **gitignored** — we do not commit scraped government
content. A 7-row schema sample is committed at `data/samples/corpus_sample.jsonl`.

| Source | Type | Chunks |
|---|---|---:|
| Act on the Employment, etc. of Foreign Workers | statute | 32 |
| Immigration Act | statute | 72 |
| Enforcement Decree of the Act on the Employment of Foreign Workers | decree | 35 |
| Minimum Wage Act | statute | 19 |
| Enforcement Decree of the Labor Standards Act | decree | 43 |
| Enforcement Decree of the Industrial Accident Compensation Insurance Act | decree | 101 |
| Act on Collection of Insurance Premiums (Employment + Industrial Accident) | statute | 100 |

## Compliance

`robots.txt` checked for all five candidate domains **before** any fetching:

| Domain | Rule | Used |
|---|---|---|
| `law.go.kr` | `Allow: /` | yes |
| `elaw.klri.re.kr` | blocks only search/admin endpoints; `viewer.do` allowed | yes |
| `eps.go.kr` | `Allow: /` | not yet |
| `hikorea.go.kr` | blocks `/search/`, `/bbs/`, `/oidc/` | not yet |
| `moel.go.kr` | blocks `/info/defaulter/` (personal data), `/portal/`, `/v2024/` | not yet |

Fetcher uses an identifying User-Agent and a 1.5s inter-request delay.

**Licensing:** every chunk carries `license_note`. The English texts are *unofficial*
translations with no legal effect — the Korean original governs. This must survive into
Phase 3 training data so the model never presents a translation as binding law.

## Two working endpoints (both took reverse-engineering)

Standard pages return only a JS shell. The endpoints that actually serve statute text:

- `https://www.law.go.kr/LSW/engLsInfoR.do?lsiSeq=<id>`
- `https://elaw.klri.re.kr/eng_mobile/viewer.do?hseq=<id>&type=part&key=40`

The obvious candidates (`lawView.do`, `lsInfoP.do`, `lsInfoR.do`, the ExtJS search pages)
all return shells of 800–1,650 chars. The builder guards against this: any source yielding
under 2,000 chars is rejected as "TOO SHORT" rather than silently written as garbage.

## Verification run

- **Duplicates:** 402/402 chunks unique by SHA-1. No cross-source contamination.
- **Title match:** each source's text was checked against its claimed title. **One mismatch
  found and fixed** — `hseq=67169` serves the *Enforcement Decree* of the Industrial Accident
  Compensation Insurance Act, not the parent Act. Left uncorrected this would have produced
  wrong citations.
- **Topic coverage:**

| Question area | Chunks |
|---|---:|
| industrial accident | 112 |
| wages / minimum wage | 54 |
| employment permit | 23 |
| re-entry | 13 |
| departure guaranty (출국만기보험) | 12 |
| workplace change (사업장 변경) | 7 |
| alien registration | 6 |

## Known gaps — carry into Phase 2

1. **Workplace change has only 7 chunks**, yet it is the single most-asked E-9 question.
   Statutes state the principle; the operational detail (change limits, regional rules,
   deadlines) lives in MOEL notices and EPS guidance, which are not yet ingested.
2. **No Korean-language originals.** law.go.kr Korean pages are JS-rendered and returned a
   1,489-char shell. Retrieval is English-side by design so this is not blocking, but Korean
   official terms will need a Phase 3 glossary rather than corpus extraction.
3. **No practical/procedural sources yet** — `eps.go.kr` and `hikorea.go.kr` are both
   permitted by robots.txt but not yet fetched. This is where "which form, which office,
   what deadline" answers live.
4. **The corpus skews to insurance law.** 201 of 402 chunks are the two insurance documents,
   simply because they are long. Retrieval quality matters more than raw balance, but this
   should be watched in Phase 2 — if insurance chunks crowd out workplace-change results,
   the fix is per-document caps or topic-weighted retrieval.

## Next — Phase 2

Retrieval smoke test: 20 real Burmese questions, comparing (A) embedding the Burmese query
directly with BGE-M3 versus (B) translating to English first, then embedding. Measure
recall@5 and pick with the number. Gap 1 above should be the first thing the test exposes.
