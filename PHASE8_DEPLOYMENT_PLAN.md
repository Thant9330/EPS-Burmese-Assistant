# Phase 8 — Deployment plan (Hugging Face Space, Gradio, free tier)

**Goal:** a testable public demo of the EPS-Burmese assistant at zero hosting cost, so real
questions can be put in front of it before deciding on anything larger.

**Decided:** Gradio on Hugging Face Spaces. Budget: free tier only.

## The constraint that shapes everything

HF Spaces free tier is **2 vCPU / 16 GB RAM, no GPU**. The current serving path
(SEA-LION 9B, bitsandbytes 4-bit + LoRA) requires CUDA and cannot run there at all.

The only way a 9B model fits in 16 GB of RAM is a **GGUF Q4_K_M export** (~5.5 GB) run
through `llama.cpp`. Expect roughly **1-3 minutes per answer** on two cores. That is
acceptable for a test; it is not a live chat product.

### The export trap

To produce GGUF the LoRA must be merged into **fp16** weights. Merging into 4-bit rounds
the adapter delta away to nothing — this is a measured failure on this stack (base loss
1.314, unmerged 0.078, merged-into-4bit 1.235; see `PHASE4_6_TRAINING_RESULTS.md`). An
fp16 merge of a 9B model needs ~18 GB of RAM, which free Colab does not provide.

Options for the merge step, in order of preference:

1. **Colab high-RAM runtime** (51 GB when offered) — load base in fp16 on CPU, apply the
   adapter, `merge_and_unload()`, save, convert with `llama.cpp/convert_hf_to_gguf.py`,
   quantise to Q4_K_M, push to the Hub. Needs ~40 GB of disk headroom.
2. **Streamed per-shard merge** — load one safetensors shard at a time, apply the matching
   LoRA deltas, write out. Fits in ordinary RAM but is fiddly and needs care with the
   `base_model.model.` key prefixes.
3. **llama.cpp runtime LoRA** (`convert_lora_to_gguf.py` + `--lora`) — avoids the merge
   entirely, but applying an adapter over a quantised base risks the same precision loss
   that broke the 4-bit merge. **Verify against the unmerged reference before trusting it**
   (see verification below).

This step needs a GPU/high-RAM session and is the only genuinely hard part of shipping.

## What can be built now, with no GPU and no export

Everything except the generation call:

- **Retrieval.** 602 chunks; corpus embeddings precomputed offline and shipped as a `.npy`
  (602 x 1024 float32 = ~2.5 MB). `bge-m3` loads on CPU and embeds a single query in
  ~1-2 s.
- **Diverse top-5 selection** — one chunk per `source_id`. Measured at hit@5 82% vs 77%
  for plain top-5, at identical context length, with nothing lost. Ship this, not plain
  top-k.
- **Prompt assembly** — the exact system prompt used in training, byte-identical. Do not
  reword it; the model was trained against that specific string.
- **Output scaffolding** (see safety below).
- **The whole Gradio UI.**

Put generation behind a single `generate(prompt) -> str` interface with two
implementations: a stub for local development and `llama_cpp` for production. The export
then becomes a one-line swap.

## Safety scaffolding — not optional

Phase 7 measured that roughly **8% of questions produce a confident, well-cited, wrong
answer**, and that neither the model nor the retriever can detect the condition (a
similarity gate does not separate the cases). The mitigation has to be in the product.

Mandatory in every response:

1. **Show the retrieved source text**, collapsed but present. A worker who can read the
   passage can catch a wrong answer; one who only sees the conclusion cannot.
2. **Append the disclaimer** — the 131-char `*မှတ်ချက် — ...*` block stripped from training
   data. It was removed so the model would not waste capacity memorising it; the app adds
   it back at output time.
3. **Show the referral prominently** — 고용센터 for labour/wage/insurance, 출입국관리사무소
   for visa/residence. Not buried in the answer text.
4. **Log every question, retrieved sources, and answer** for later review. This is the only
   way to learn the real-world failure rate, which the 50-question held-out set can only
   estimate.

State plainly on the page that this is an unofficial reference tool, not legal advice.

## Free wins to apply first (no GPU, no retraining)

1. **Diverse top-5 retrieval** — one chunk per source. 77% -> 82% hit@5, free.
2. **Ingest `hikorea_e74`** — three held-out questions cite a page that was never added to
   the corpus, so they are unanswerable at any retrieval setting. Re-run
   `scripts/phase1_build_corpus.py` coverage after adding.
3. **Normalise source tags** — the dataset and corpus disagree (`lba_eng` vs
   `labor_standards_act_eng`, `ia_eng`, `gwra_eng`, `minwage_act_eng`). Either fix the
   corpus `source_id` values or ship the alias map; any citation checking breaks otherwise.

## Build order

1. Apply the three free wins above.
2. Build the Space: retrieval + diverse top-5 + prompt assembly + UI + safety scaffolding,
   with generation stubbed. Testable end to end without a model.
3. Do the fp16 merge and GGUF export in a high-RAM session.
4. Swap in `llama_cpp`, measure real latency on the Space hardware.
5. Private pilot: share the link with a small group, review the logs.

Steps 1-2 need no GPU and can proceed immediately. Step 3 is the blocker.

## Verification

The export must be checked against the known-good reference, not assumed:

- On a handful of **training** rows, the GGUF build should reproduce the unmerged adapter's
  behaviour, not the base model's. The reference numbers are base loss 1.314 vs adapter
  0.078; if the export scores like the base model, the adapter did not survive.
- On the 50 held-out questions, spot-check that Korean-term retention and citation rate
  stay near the measured 18/30 and 34/45. A sharp drop means the quantisation or the merge
  damaged the adapter.
- Confirm output is coherent Burmese with no repetition loops — the failure mode that
  dominated Phases 4-6.

## Explicitly not in scope

- Always-on GPU hosting. Revisit only if real usage justifies it.
- Retraining. The model is not the bottleneck; retrieval and hosting are.
- Dataset expansion. Owner has deferred this.
