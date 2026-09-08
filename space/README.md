# Hugging Face Space — built, not deployed

A complete Gradio app for the assistant. **Not deployed**, because hosting it costs money.

## Why it is not live

Hugging Face now requires a PRO subscription ($9/month) to host *any* Gradio Space —
including on free `cpu-basic` hardware. The API returns:

```
402 Payment Required
Static Spaces are free for everyone, but hosting Gradio and Docker Spaces
on free cpu-basic requires a PRO subscription.
```

ZeroGPU hosting (NVIDIA RTX Pro 6000 Blackwell, 48 GB) is also PRO-gated in practice,
despite documentation describing a free tier for accounts in good standing.

## What is here

| file | purpose |
|---|---|
| `app.py` | the full application — retrieval, prompt assembly, generation, sources panel |
| `requirements.txt` | dependencies |
| `SPACE_CARD.md` | the Space README, with front-matter ready to use |

## To deploy it

1. Obtain PRO, or a community grant
2. Create a Gradio Space, hardware **ZeroGPU**
3. Upload these three files, renaming `SPACE_CARD.md` back to `README.md`
4. Add the two retrieval artefacts, which are not in this repo because they are generated:
   - `corpus_emb.npy` — 602 × 1024 float32, `bge-m3` embeddings of the corpus
   - `corpus_meta.jsonl` — `source_id`, `title`, `text` per chunk

   Generate both by running `bge-m3` over `data/processed/corpus.jsonl` (see
   `scripts/phase1_build_corpus.py` for how the corpus is built).

## Design notes

**Retrieval runs on CPU.** Corpus embeddings are precomputed and shipped, so `bge-m3` only
embeds one short query per request. Only generation is wrapped in `@spaces.GPU`, keeping the
GPU window — and therefore the visitor's daily quota — as small as possible.

**The adapter is loaded unmerged.** `merge_and_unload()` on 4-bit weights rounds the LoRA
delta away and silently returns the base model.

**Retrieval caps at one chunk per document.** Without the cap, 100-chunk statutes crowd out
2-chunk procedural pages; hit@5 measured 77% plain versus 82% capped, at identical context
length.

## The free alternative that does work

Running the same app from Colab with `demo.launch(share=True)` gives a public link for as
long as the notebook runs. Temporary, but free, and it needs no changes to `app.py` beyond
removing the `@spaces.GPU` decorator.
