# Phase 4/5/6 — first real training run, results

**Run: 2026-09-03 · 572-row dataset (520 train / 50 held-out) · SEA-LION v3 9B QLoRA**

## What happened

Ran the full pipeline for real for the first time: baseline eval (Phase 4), QLoRA
training (Phase 5), fine-tuned eval (Phase 6), automated comparison. Training itself
completed cleanly — 390/390 steps, 3 epochs, loss 18.64 → ~3.0, healthy monotonic
convergence, no NaN/Inf, peak VRAM 7.88GB (~65 min on a free T4). Along the way found
and fixed a real bug (Bug 4, see `SPIKE_RUN_STATE.md`): a `PeftModel`-wrapped adapter
produced total-breakdown garbage at inference on this 4-bit-bnb + Gemma2 stack;
`merge_and_unload()` fixed that specific problem.

**But the fine-tuned model still fails at inference, in a different and very
consistent way: repetition-loop collapse.** Every one of the 50 held-out questions
follows the same pattern — the model starts on-topic, in correct Burmese, then locks
onto a single word or short phrase and repeats it for the rest of the 300-token budget.
Examples (from the random 8-sample printout):

- `အလုပ်လုပ်လို့ရလား၊ အလုပ်လုပ်လို့ရလား၊ အလုပ်လုပ်လို့ရလား...` (repeats the question's own
  phrase back)
- `အလုပ်ခွင် အလုပ်ခွင် အလုပ်ခွင် အလုပ်ခွင်...` ("workplace workplace workplace...")
- `အောက်ပါ အောက်ပါ အောက်ပါ...` ("as follows as follows as follows...")

This is not the same failure as Bug 3/Bug 4 (which produced scrambled garbage across
random scripts, a sign of a broken computation path). This is coherent Burmese that
gets stuck — a much more standard LLM failure mode, but a real one, and it happened on
100% of the sample, not occasionally.

## Ruled out: greedy-decoding artifact

Tested whether `repetition_penalty` + `no_repeat_ngram_size` (standard fixes for
greedy-decoding loops) would surface a better answer underneath:

- Mild (`repetition_penalty=1.1, no_repeat_ngram_size=4`): degenerated into Burmese
  syllable salad — not real words, just script fragments.
- Aggressive (`repetition_penalty=1.3, no_repeat_ngram_size=3`): degenerated into
  complete garbage mixing English and Devanagari script into the output.

**Both made it worse, not better.** This rules out "greedy decoding is just hiding a
good second choice" — the model's probability distribution at these positions is
genuinely narrow/peaked on the repeated token, and forcing it away from that peak finds
noise, not signal. That's diagnostic: it points to the *training*, not the decoding
strategy.

## Automated metrics (BEFORE = untrained base, AFTER = fine-tuned)

| metric | BEFORE | AFTER |
|---|---:|---:|
| Korean-term retention (avg) | 2.8% | 0.0% |
| refusal/grounded correctness | 90.0% | 90.0% |
| article-number match (heuristic) | 61.9% | 4.8% |

**The refusal/grounded correctness number is misleading and should not be read at face
value.** It only checks whether the output contains the fixed refusal boilerplate
string. Since AFTER's repetition-loop outputs never contain that string, they
automatically "pass" on every grounded row (90% of the set) regardless of whether the
content is usable — the metric can't distinguish a real answer from a repetition loop
that happens not to say "I don't know." The article-number match (61.9% → 4.8%) and the
qualitative sample are the honest signal here: the fine-tuned model got dramatically
worse on both, because repetition loops rarely reach the citation-line portion of the
answer before running out of tokens.

## Likely cause and what to try next

Loss converging cleanly does not guarantee generation quality — a model can reach low
teacher-forced loss while its free-running (autoregressive) generation degrades, especially
with a small, structurally repetitive dataset (every gold answer shares heavy
boilerplate: the same disclaimer sentence, the same `ပုဒ်မ N (Article N) အရ` citation
phrasing) and enough epochs to overfit those surface patterns rather than the
underlying content. Plausible contributors, roughly in order of suspicion:

1. **3 epochs was probably too many** for 520 examples this repetitive in structure —
   worth retrying with 1-2 epochs.
2. **`learning_rate=1e-4` may still be too high** for this dataset size even after
   already lowering it from the spike's `2e-4`.
3. **No generation-quality check during training** — only training loss was monitored;
   loss looked perfectly healthy right up to the end, so it would not have caught this.
   A future run should sample a generation from a held-out prompt every N steps, not
   just log loss.

**Not recommended**: more decoding-time patches (repetition penalty, etc.) — already
shown to make things worse, and it would be masking a training problem rather than
fixing it.

## Next step (needs a decision, not more automation)

Retrain with fewer epochs and/or a lower learning rate is the most likely fix, but that
costs another ~65 min of free-tier GPU time (now on a second Google account after the
first hit its quota) — not something to spend without confirming direction first.
