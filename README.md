# EPS-Burmese-Assistant

A Burmese-language question-answering system for Myanmar workers on Korean E-9/EPS visas,
grounded in Korean labour and immigration law.

Korean employment law governs the daily life of hundreds of thousands of migrant workers —
how to change workplace, what overtime pays, what happens when an employer refuses to renew
a contract. The statutes exist in Korean and in official English translation. Almost none of
it exists in Burmese.

This repository is the full record of building that system on free hardware: corpus
construction, cross-lingual retrieval, dataset creation, QLoRA fine-tuning, and — the part
most write-ups skip — an evaluation honest enough to say where it fails.

**Status: research prototype. Not a service, and not safe to act on without verification.**

---

## Results

Fine-tuned SEA-LION v3 9B vs the same model untuned, on 50 held-out questions it never saw
during training. Identical prompts, identical decoding, adapter applied unmerged.

| | base | fine-tuned |
|---|---|---|
| Keeps Korean official terms (`고용센터`, `산재보험`) | 5/30 | **18/30** |
| Cites a source | 26/45 | **34/45** |
| Refuses correctly when the law does not answer | 2/5 | **4/5** |
| Names the correct government office | 0/5 | **2/5** |
| Article number correct, when cited | 13/13 | 20/21 |
| Repetition loops | 1/50 | 1/50 |
| Stopped naturally | 44/50 | 44/50 |

Every quality measure improved; none degraded. Read the article-number row carefully: base
cited an article 13 times and was right every time; the fine-tune cited 21 times and was
right 20. It attempts citation far more often at slightly lower precision.

The fine-tuning did **not** teach the model Burmese — SEA-LION already writes fluent Burmese.
It taught task conventions: cite the source, keep Korean terms, decline honestly.

Full detail: [`PHASE4_6_TRAINING_RESULTS.md`](PHASE4_6_TRAINING_RESULTS.md).
Raw generations: `data/eval/full50_base_vs_tuned.json`.

### What it fixes, concretely

Asked where to apply for a workplace change, the base model invented an office that does not
exist and then explained what it was:

```
BASE : ... **ရပ်ကွက် လူမျိုးရင်း ဌာန** (Local Immigration Office) မှာ တင်သွင်းရမည်။
       **မှတ်ချက်:** လူမျိုးရင်း ဌာန ဆိုသည်မှာ ...
TUNED: ... 출입국관리사무소 (လူဝင်မှုကြီးကြပ်ရေးရုံး) သို့ လျှောက်ထားရပါသည်။
       ရင်းမြစ် — hikorea_189 — Permission to Change or Add Workplace
```

Asked about overtime pay, the base wrote `50/100` and glossed it in Burmese as *"eighty
percent"* — a wrong number in a wage answer. The fine-tune gets `၅၀% (၅၀/၁၀၀)` right.

---

## Two bugs worth reading about

Three days went into a repetition collapse that had nothing to do with the model or the data.

### 1. The training library returned logits shifted by one position

Every checkpoint degenerated into repeating loops. Hyperparameters were tuned, the dataset was
rebuilt, nothing helped — because training was optimising **"predict the previous token"**,
which is a repetition machine by construction.

The tell had been in the very first loss curve:

```
step 10 loss 18.76
```

A 256,000-token vocabulary means uniform random guessing scores `ln(256000) = 12.45`.
**A loss of 18.7 is worse than random — that means confidently wrong, not "hard data".**

One identical example through both paths, same `transformers 5.5.0`:

| | Unsloth | plain transformers |
|---|---|---|
| reported loss | 14.72 | **1.297** |
| recomputed with standard shift | 221.35 | **1.297** (matches) |
| `argmax(logits[i]) == input_ids[i+1]` | **0.0%** | **58.7%** |
| `argmax(logits[i]) == input_ids[i]` | 98.8% | 0.0% |
| logit range | -304 … 641 | -29.9 … 28.1 |

Gemma2's `final_logit_softcapping=30.0` was also not applied, hence logits reaching 641.

### 2. `merge_and_unload()` silently deletes a LoRA on 4-bit weights

After the first fix, base and fine-tuned produced *byte-identical* output. The LoRA delta is
finer than the 4-bit quantisation step, so merging rounds it away:

| model | loss on a training row |
|---|---|
| base, no adapter | 1.314 |
| adapter, **not** merged | **0.078** |
| adapter, merged into 4-bit | 1.235 — back to base |

`peft` warns about this. The warning should be treated as an error. **Evaluate unmerged.**

### The lesson that generalises

Twice, an evaluation silently compared the model against itself — once via the merge, once
because `PeftModel.from_pretrained` injects LoRA layers into the base model *in place*, so
`base` and the wrapper are the same object.

Both times the tell was identical: **aggregate metrics matching exactly between two supposedly
different models.** That is a bug signal, never a finding.

---

## How it works

```
Burmese question
   ↓
bge-m3 embedding  ──►  602 English law chunks
   ↓                    (one chunk per document — diverse top-5)
retrieved context
   ↓
SEA-LION v3 9B + LoRA  ──►  Burmese answer + ရင်းမြစ် source line
   ↓                          or an honest refusal
disclaimer appended by the application
```

**Burmese queries match English chunks directly, with no translation step.** That assumption
was measured, not assumed: 80% hit@5 for Burmese queries against a 90% ceiling set by human
English translation ([`PHASE2_RETRIEVAL_RESULTS.md`](PHASE2_RETRIEVAL_RESULTS.md)).

### Why SEA-LION

Burmese tokenises expensively, and that single ratio decides base model, sequence length,
VRAM, and whether free Colab is viable at all. Measured on 8 parallel Burmese/English
sentence pairs from the real domain:

| model | vocab | MY/EN tokens |
|---|---:|---:|
| **aisingapore/gemma2-9b-cpt-sea-lionv3-instruct** | 256,000 | **3.32×** |
| Qwen/Qwen3-8B | 151,669 | 6.03× |
| sail/Sailor2-8B | 151,665 | 6.03× |
| aisingapore/Llama-SEA-LION-v3-8B-IT | 128,256 | 7.79× |

An hour of measurement changed the base-model choice
([`PHASE0_TOKENIZER_RESULTS.md`](PHASE0_TOKENIZER_RESULTS.md)).

### Training

QLoRA on a free T4. `r=16`, `alpha=32`, `dropout=0`, all seven projections, batch 1 ×
grad-accum 4, 3 epochs, `lr=1e-4`, fp16, seed 0. Loss 0.932 → 0.054 over 393 steps in
~76 minutes, peak 12.98 GB.

Plain `transformers` + `peft` + `trl` — **not** Unsloth, for the reason above.

---

## Limitations

Measured, not guessed. These matter more than the results table.

**~8% of answers are confidently wrong end to end.** Retrieval returns the wrong law about
20% of the time, and roughly 40% of those become fluent, correctly-formatted, wrongly-cited
answers. Examples: severance pay described as "shutdown allowance"; a deportation question
answered correctly but justified with departure-insurance law
([`PHASE7_END_TO_END.md`](PHASE7_END_TO_END.md)).

**A confidence threshold does not catch them.** Similarity scores for hits and misses overlap
almost entirely — 33 of 40 good rows score below the worst-scoring miss. Neither the model
nor the retriever can detect the condition.

**It refuses long questions.** Training questions had a median length of 52 characters. Real
questions run 90–170. The same question asked long refuses and asked short answers, with
identical retrieved sources — so this is a training-distribution problem, not a coverage one.
30 interactions logged during live testing show the pattern.

**Retrieval hit@5 is 77–82%**, and that figure is measured on questions written *from* the
corpus, so it is a ceiling rather than real-world performance.

**Coverage is unmeasured.** `hikorea_e74` is cited by 3 held-out questions and was never
ingested. What real users ask has not been characterised.

---

## Reproduce

The corpus is **not** committed — it is derived from Korean government sources and is
rebuilt from a manifest rather than redistributed here.

```bash
pip install -r requirements-phase0.txt

python scripts/phase0_tokenizer_fertility.py   # base-model choice
python scripts/phase1_build_corpus.py          # fetch + chunk -> data/processed/corpus.jsonl
python scripts/phase2_retrieval_smoketest.py   # cross-lingual retrieval check
python scripts/build_phase3_v2.py              # dataset v2 from the raw dataset
python scripts/phase4_build_eval_split.py      # held-out split
```

Training and evaluation ran in Colab; the notebook is `notebooks/`.

To check the headline numbers without rerunning anything:

```bash
python scripts/verify_results.py
```

It recomputes every figure in the results table from `data/eval/full50_base_vs_tuned.json`
and asserts that the two conditions are not accidentally the same model — the failure that
caught this project twice.

### What is committed

| path | what |
|---|---|
| `data/sources.json` | source manifest — URLs, authority, licence per document |
| `data/samples/phase3_dataset_v2.jsonl` | the 572-example dataset |
| `data/train/phase3_train_v2.jsonl` | 522 training rows |
| `data/eval/phase4_holdout_v2.jsonl` | 50 held-out questions |
| `data/eval/full50_base_vs_tuned.json` | headline result, 100 generations |
| `data/eval/end2end_retrieved.json` | end-to-end with real retrieval |
| `data/eval/demo_log.jsonl` | 30 logged interactions from live testing |
| `data/refusal_variants.txt` | 15 owner-written refusal phrasings |

---

## Reading order

The phase documents are a chronological record, including the wrong turns.

1. [`PHASE0_TOKENIZER_RESULTS.md`](PHASE0_TOKENIZER_RESULTS.md) — measuring before choosing
2. [`PHASE1_CORPUS_RESULTS.md`](PHASE1_CORPUS_RESULTS.md) — corpus with provenance
3. [`PHASE2_RETRIEVAL_RESULTS.md`](PHASE2_RETRIEVAL_RESULTS.md) — cross-lingual retrieval
4. [`PHASE2B_CORPUS_GAP.md`](PHASE2B_CORPUS_GAP.md) — a corpus gap, and an eval that hid it
5. [`PHASE3_FIRST_BATCH.md`](PHASE3_FIRST_BATCH.md), [`PHASE3_DEEPSEEK_MERGE.md`](PHASE3_DEEPSEEK_MERGE.md) — dataset construction
6. [`PHASE4_6_TRAINING_RESULTS.md`](PHASE4_6_TRAINING_RESULTS.md) — **the main document**: the collapse, both bugs, the successful run
7. [`PHASE7_END_TO_END.md`](PHASE7_END_TO_END.md) — evaluation with real retrieval
8. [`PHASE8_DEPLOYMENT_PLAN.md`](PHASE8_DEPLOYMENT_PLAN.md) — deployment constraints
9. [`SPIKE_RUN_STATE.md`](SPIKE_RUN_STATE.md) — earlier stack bugs and their fixes

Pre-project planning material that the measurements overturned is kept in
[`docs/superseded/`](docs/superseded/) rather than deleted.

---

## Provenance and licence

Code in this repository is MIT licensed (see [`LICENSE`](LICENSE)).

The **corpus is not redistributed here.** It derives from:

- **law.go.kr** (Korea Ministry of Government Legislation) — unofficial English translations
- **elaw.klri.re.kr** (Korea Legislation Research Institute) — unofficial English translations
- **hikorea.go.kr** (Korea Immigration Service) — official procedural guidance
- **easylaw.go.kr** — plain-language government legal guides

Every source carries its own note in `data/sources.json`, and every chunk produced by the
builder carries `source_url`, `authority`, and `license_note`. All English legal text is
**unofficial translation with no legal force; the Korean original governs.**

Burmese answers in the dataset are the author's own work, grounded in and citing that
material. The base model is `aisingapore/gemma2-9b-cpt-sea-lionv3-instruct`; any use of the
adapter is additionally subject to that model's licence terms and to Gemma's Terms of Use.

**Nothing here is legal advice.** Verify with 고용센터 (employment centre) or 출입국관리사무소
(immigration office) before acting.
