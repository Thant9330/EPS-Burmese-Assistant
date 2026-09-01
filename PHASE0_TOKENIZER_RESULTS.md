# Phase 0 — Tokenizer Fertility Results

**Run:** 2026-09-01 · `scripts/phase0_tokenizer_fertility.py` · transformers 5.16.1 (tokenizers only, no torch)

## Why this ran first

Burmese script is far more expensive to tokenize than English. That single ratio decides the
base model, `max_seq_length`, VRAM, and therefore whether free Colab is enough. Measuring it
took under an hour and changed the plan's base-model choice.

## Method

8 **parallel** sentence pairs — same meaning in Burmese and English, drawn from the real
E-9/EPS domain (workplace change, contract extension, re-entry, insurance, E-7-4, unpaid wages).
Parallel matters: comparing unrelated sentences would measure content length, not the tokenizer.
`MY/EN` = tokens Burmese costs versus English **for identical meaning**. Lower is better.

## Results

| Model | Vocab | MY tok | EN tok | MY/EN | verdict |
|---|---:|---:|---:|---:|---|
| **aisingapore/gemma2-9b-cpt-sea-lionv3-instruct** | 256,000 | 289 | 87 | **3.32x** | **best instruct model** |
| Qwen/Qwen3-8B | 151,669 | 519 | 86 | 6.03x | byte fallback |
| Qwen/Qwen2.5-7B-Instruct | 151,665 | 519 | 86 | 6.03x | byte fallback |
| sail/Sailor2-8B | 151,665 | 519 | 86 | 6.03x | SEA vocab excludes Burmese |
| aisingapore/Llama-SEA-LION-v3-8B-IT | 128,256 | 670 | 86 | 7.79x | worst |
| BAAI/bge-m3 *(embedding, not a base)* | 250,002 | 119 | 92 | 1.29x | reference |
| facebook/nllb-200-distilled-600M *(MT, not a base)* | 256,204 | 88 | 93 | 0.95x | floor |

Not tested — gated, no HF token: `meta-llama/Llama-3.1-8B`, `google/gemma-3-4b-it`,
`CohereLabs/aya-expanse-8b`.

## The mechanism

Tokenizing **ဗီဇာ** ("visa" — 4 characters, 12 UTF-8 bytes):

- **Qwen** → 7 tokens: `['áĢ','Ĺ','áĢ','®','áĢĩ','áĢ','¬']`
- **BGE-M3** → 3 tokens: `['▁','ဗီ','ဇာ']`
- **SEA-LION/Gemma2** → 4 tokens: `['ဗ','ီ','ဇ','ာ']`

Qwen's fragments are raw UTF-8 **bytes**, not Burmese subwords — its 151k vocab contains
essentially no Burmese, so it falls back to byte-level BPE at 1.47 tokens *per character*.
Gemma2's 256k SentencePiece vocab holds real Burmese characters.

## Sequence budget (realistic example)

English retrieved context + grounded Burmese answer:

| Model | context | answer | total |
|---|---:|---:|---:|
| SEA-LION/Gemma2 | 295 | 311 | **606** |
| Qwen3-8B | 295 | 551 | 846 |

The gap is concentrated in the **generated** half (311 vs 551, 1.77x) — exactly the part that
drives inference cost and generation quality.

## Decisions

1. **Base model → `aisingapore/gemma2-9b-cpt-sea-lionv3-instruct`**, replacing Qwen 2.5/3.
   Roughly half the tokens per Burmese answer, *and* already continued-pretrained on SEA
   languages including Burmese. Ungated, so no HF token needed.
2. **`max_seq_length` = 2048.** Real examples land near 600 tokens; 2048 leaves ample headroom.
3. **Free Colab T4 (16GB) is sufficient.** 9B in 4-bit ≈ 5.5GB weights; at 2048 tokens with
   gradient checkpointing this fits. No paid tier required — revisit only if Phase 3 data
   turns out much longer than sampled here.

## Caveats

- 8 sentence pairs is a small sample. The Qwen-vs-Gemma2 gap is large enough to be decisive,
  but do not treat 3.32x as precise.
- 9B > 8B. Still fits 4-bit on T4, but it is tighter than the 7B originally planned.
- Unsloth support for Gemma 2 should be confirmed before Phase 5.
