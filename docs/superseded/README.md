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
