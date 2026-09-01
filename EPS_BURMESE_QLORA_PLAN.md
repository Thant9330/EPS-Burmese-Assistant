# Burmese EPS/E-9 Visa Assistant — QLoRA Learning Project

## Context

The goal is **learning QLoRA/Unsloth hands-on** on a project that is genuinely useful, not a
portfolio prop. The original plan (English Q&A on F-2-7/F-5) fights the tool: it's a *facts*
task, and Qwen already speaks English well — so the fine-tune would show no measurable gain
over simply prompting the base model, and you'd learn nothing from the result.

This plan picks a task where **the base model visibly fails first**, so the before/after delta
is obvious and the learning is real.

**The concern about Burmese embeddings is valid but applies to retrieval, not fine-tuning.**
A web check confirms BGE-M3 advertises 100–194 languages but never explicitly lists Burmese,
and its own docs admit training data is "highly unbalanced" across languages. So the design
avoids depending on it: **corpus and retrieval stay in English/Korean; only generation is
Burmese.** Phase 2 measures this rather than assuming it.

**Audience:** Myanmar nationals on E-9/EPS visas in Korea — one of the largest and most
underserved expat groups. They do not read the English HiKorea pages. This is the part that
makes the project useful rather than trash.

---

## Architecture

```
Burmese question
      ↓
  [translate/normalize to English query]
      ↓
  English/Korean corpus  ──BGE-M3 or e5-large──►  top-k chunks (English)
      ↓
  FINE-TUNED Qwen  ← this is the only trained component
      ↓
  Burmese answer, grounded in chunks, Korean terms preserved + glossed,
  citation, disclaimer, or explicit refusal
```

**What the LoRA actually learns** (all behavior/style, zero facts):
1. Write fluent, natural Burmese (base model is weak here → the measurable delta)
2. Read English context, answer in Burmese without drifting into English
3. Preserve Korean official terms verbatim with a Burmese gloss —
   e.g. `사업장 변경` (workplace change) must survive, because that's what the worker
   has to say at the 고용센터
4. Refuse when retrieved context is insufficient, instead of inventing rules

Facts live in the retrieval corpus, so they can be updated without retraining. This is the
correct division of labor and is the single most important thing to internalize from the project.

---

## Scope

**In:** E-9/EPS core needs — workplace change (사업장 변경) rules and limits, contract renewal,
re-entry special system, insurance (4대보험 / 출국만기보험), wage and labor-rights basics,
and the **E-9 → E-7-4 transition** (the aspirational path, high value, poorly documented in Burmese).

**Out:** F-5/F-2-7 professional track (well-served in English already), anything requiring
individualized legal judgment.

---

## Phases

### Phase 0 — Tokenizer fertility check (~30 min, do this FIRST)
Measure tokens-per-character for Burmese script across candidate bases:
Qwen3-8B, Gemma-3, Llama-3.1-8B. Burmese is typically 3–5x more expensive than English.

This single number decides base model, `max_seq_length`, VRAM, and whether free Colab suffices.
It's also the cheapest possible first lesson in why tokenizers matter.

### Phase 1 — Corpus build
50–150 authoritative documents: HiKorea, EPS (eps.go.kr), MOEL notices, Immigration Act
enforcement decrees. Keep Korean originals + English versions. Chunk ~500 tokens, keep
source URL + retrieval date on every chunk.

### Phase 2 — Retrieval smoke test (directly tests the embedding fear)
Write 20 real Burmese questions. Compare recall@5 under two strategies:
- **A:** embed Burmese query directly with BGE-M3 → English chunks
- **B:** translate query to English first → embed → English chunks

Pick whichever wins. Expect B, but **measure it** — this converts your worry into a number.

### Phase 3 — Dataset (~800–1500 pairs) ← 80% of the real work
Format: `{context: [English chunks], question: Burmese, answer: Burmese}`

Bootstrap by distillation from a strong model, then **human-review a slice yourself** — you
are the quality bar, and no automated metric replaces that. Reuse the existing
`aihub-burmese-writer` agent for tone calibration (it already targets this exact audience).

Deliberately include **~15% unanswerable examples** where context doesn't support an answer
and the gold output is a refusal. Without these the model learns to always answer, which is
the dangerous failure mode in an immigration context.

### Phase 4 — Baseline eval BEFORE training (do NOT skip)
Run the full eval suite on the untuned base model. This is the entire scientific value of the
project — skip it and you can never claim the fine-tune did anything.

### Phase 5 — QLoRA train (Unsloth)
4-bit, r=16–32, target all attention + MLP projections, 2–3 epochs, bf16, packing on.
Checkpoint to Drive every N steps (Colab disconnects).

### Phase 6 — Eval again, compare to Phase 4
Held-out ~100 examples:
- **Groundedness** — answer uses only provided context (LLM judge)
- **Refusal accuracy** — on the unanswerable set
- **Burmese fluency** — human rating on 30 samples (you), plus chrF++ vs reference
- **Korean term retention** — regex for required Hangul terms surviving in output

### Phase 7 — Ship
Merge adapter → 16-bit → GGUF → Gradio on HF Spaces free tier. Publish dataset card,
eval table (both columns), and training notebook.

---

## Hardware

Start on **free Colab T4** — QLoRA on a 7–8B fits fine. Only pay for Pro if Phase 0 shows
Burmese fertility forces `max_seq_length` past ~2048. Decide with the number, not upfront.

Note: the `colab-mcp` server failed to connect this session (CONNECT_TIMEOUT). Worth fixing
or retrying before Phase 5, otherwise drive Colab manually.

---

## Honest risks

- **Data quality is the project.** Training is one afternoon; the dataset is weeks. Budget accordingly.
- **You are the only Burmese evaluator.** No metric substitutes. Plan the review time.
- **Liability.** Every answer needs a citation + disclaimer, and the model must never say
  "you qualify" — only "the rule says X; verify at 출입국." Bake this into the training data,
  not into a system prompt that a user can strip.

---

## Verification

End-to-end: ask a Burmese E-9 question the corpus covers → confirm grounded Burmese answer
with correct Korean terms and a citation. Then ask one the corpus does *not* cover → confirm
it refuses instead of inventing. Compare both against the Phase 4 baseline outputs side by side.

Sources consulted: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3),
[M3-Embedding paper](https://arxiv.org/abs/2402.03216),
[BGE docs](https://bge-model.com/bge/bge_m3.html)
