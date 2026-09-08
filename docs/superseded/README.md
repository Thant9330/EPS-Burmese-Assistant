# Superseded planning material

Kept deliberately. These documents predate the measurements and were overturned by them.

## `korean_immigration_ai_fine_tuning_infographic.html`

The original project blueprint, written before any code ran. It recommends
**Qwen 2.5 7B Instruct** as the base model.

Phase 0 then measured tokenizer fertility on 8 parallel Burmese/English sentence pairs
from the real domain:

| model | Burmese/English token ratio |
|---|---:|
| aisingapore/gemma2-9b-cpt-sea-lionv3-instruct | **3.32x** |
| Qwen/Qwen2.5-7B-Instruct | 6.03x |

Qwen falls back to byte-level encoding for Burmese script, costing nearly twice as many
tokens per sentence — which drives sequence length, VRAM, and whether free Colab is viable
at all.

The blueprint was wrong, and an hour of measurement was enough to show it. That is why it is
kept rather than quietly deleted: it is the clearest illustration in this repository of why
the project measures before deciding.

See [`PHASE0_TOKENIZER_RESULTS.md`](../../PHASE0_TOKENIZER_RESULTS.md).


## `EPS_BURMESE_QLORA_PLAN.md`

The project plan, written before any code ran. Three of its assumptions were disproved by
the work itself:

| the plan assumed | what was measured |
|---|---|
| Train with **Unsloth** | Unsloth returned logits shifted one position on `transformers 5.5.0`, training the model to predict the *previous* token. Switched to plain `transformers` + `peft` + `trl`. |
| The Burmese query must be **translated to English** before retrieval | Not needed. `bge-m3` matches Burmese queries against English chunks directly — 80% hit@5. |
| The base model is **weak at Burmese**, so fluency is "the measurable delta" | False. SEA-LION already wrote fluent Burmese. The fine-tune taught task conventions, not the language. |

What it got right, and which held all the way through: keeping facts in the retrieval corpus
rather than the weights, preserving Korean official terms verbatim so a worker can say them
at the 고용센터, and choosing a task where the base model visibly fails — sound reasoning,
even though the predicted failure turned out to be the wrong one.
