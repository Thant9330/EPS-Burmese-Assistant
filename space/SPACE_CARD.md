---
title: EPS Burmese Assistant
emoji: 🇲🇲
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: gemma
short_description: Korean labour law answered in Burmese, grounded in the statutes
models:
  - MYOTHANTZIN/eps-burmese-sealion-9b-lora
datasets:
  - MYOTHANTZIN/eps-burmese-qa
---

# EPS Burmese Assistant

Answers Korean labour and immigration law questions **in Burmese**, for Myanmar workers on
E-9/EPS visas in South Korea. Every answer is grounded in the actual statute text and cites
its source; when the law does not cover the question, it declines rather than inventing a rule.

**Research prototype, not a service.** About 8% of answers are confidently wrong — correct
formatting, plausible citation, wrong law. Neither the model nor the retriever can detect
when this happens, which is why the retrieved passages are shown for every answer. Read them.

Always confirm with 고용센터 (employment centre) or 출입국관리사무소 (immigration office)
before acting on anything here.

## How it works

1. Your Burmese question is embedded with `bge-m3` and matched **directly** against 602
   English law chunks — no translation step. That assumption was measured: 80% hit@5.
2. Five passages are selected, capped at one per document so large statutes cannot crowd
   out short procedural pages.
3. SEA-LION v3 9B with a QLoRA adapter answers in Burmese, citing the source.

## Known limitations

- **Long questions get refused.** Training questions had a median length of 52 characters;
  real questions run 90–170. Ask short and specific for now.
- **Narrow coverage** — E-9/EPS topics only. Not tax, housing, healthcare or schooling.
- **Point in time** — sources fetched 2026-09-01. Korean law changes.

## Links

- Adapter: [MYOTHANTZIN/eps-burmese-sealion-9b-lora](https://huggingface.co/MYOTHANTZIN/eps-burmese-sealion-9b-lora)
- Dataset & benchmark: [MYOTHANTZIN/eps-burmese-qa](https://huggingface.co/datasets/MYOTHANTZIN/eps-burmese-qa)
- Code, evaluation and write-up: [GitHub](https://github.com/MYOTHANTZIN-THANT/EPS-Burmese-Assistant)

Legal text is unofficial English translation with no legal force; the Korean original governs.

```
Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms
```
