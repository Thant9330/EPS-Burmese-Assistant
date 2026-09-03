# Phase 4/5/6 — first real training run, results

**Run: 2026-09-03 · 572-row dataset (520 train / 50 held-out) · SEA-LION v3 9B QLoRA**

> **RESOLVED (2026-09-03).** The conclusions in the "What we think is going on" and
> "Next step" sections below are **wrong** and are kept only as a record of the
> investigation. The real cause was not epochs, learning rate, or the dataset: Unsloth's
> patched forward pass silently misaligned the logits by one position on this stack, so
> the model was trained to predict the *previous* token. See
> **[Root cause: Unsloth logit misalignment](#root-cause-unsloth-logit-misalignment)** at
> the end of this document.


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

---

# Root cause: Unsloth logit misalignment

Found 2026-09-03, after the fix above (response-only loss masking) failed to stop the
collapse and a second run reproduced it exactly.

## The tell

Training loss started at **18.76** on the first run and **19.29** on the second. With a
256,000-token vocabulary, a model guessing uniformly at random scores `ln(256000) =
12.45`. A loss of 19 is *worse than random* — the model was confidently wrong, which is
not something hard data can cause. At step 0 the LoRA is initialised to zero, so that
number describes the untouched base model, the same base model that writes fluent
Burmese. It should have been ~1-3.

That single arithmetic check is what turned the investigation around. It should have
been applied to the first run's loss curve.

## The measurement

One real collated training example, run through both paths on the same machine, same
`transformers 5.5.0`, same example, same 4-bit quantisation:

| measurement | Unsloth | plain transformers |
|---|---|---|
| loss reported by the model | 14.72 (training logged ~19) | **1.297** |
| loss recomputed by hand, standard shift | 221.35 | **1.297** (matches) |
| loss recomputed by hand, no shift | 0.58 | 15.33 |
| `argmax(logits[i]) == input_ids[i+1]` (correct) | **0.0%** | **58.7%** |
| `argmax(logits[i]) == input_ids[i]` (copying) | **98.8%** | **0.0%** |
| logit range | **-304 … 641** | **-29.9 … 28.1** |
| `final_logit_softcapping` in config | 30.0 | 30.0 |

The labels were verified correct in both paths (`labels[i] == input_ids[i]` at 100%),
so the collator and the response-only masking were never at fault.

## Two faults, both in Unsloth's patched forward

1. **Logits shifted by one position.** Unsloth returns logits already offset, so the
   loss function's standard shift becomes a *double* shift and every prediction is
   compared against the wrong token. Printing predictions makes it unmistakable — the
   model outputs the previous target every time, at 100% confidence:

   ```
    i    input[i]   input[i+1]   argmax(logits[i])
   320   'ါ'        '်'          'ါ'
   321   '်'        'မ'          '်'
   322   'မ'        'ူ'          'မ'
   ```

2. **Gemma2 softcapping not applied.** `final_logit_softcapping=30.0` is present in the
   config but logits reach 641. Uncapped logits make the softmax effectively one-hot,
   which is why every prediction reads as 100.0%.

The internal loss is wrong too (14.72 vs the correct 1.297), not merely the returned
logits — so the **gradients were wrong**, not just the reported number.

## Why this produced exactly the symptoms we saw

Training optimised "given this context, emit the token that came before." That is a
repetition machine by construction. It explains, without any further hypothesis:

- the repetition-loop collapse on 100% of held-out questions
- why it appeared from generated token 0 rather than accumulating
- why it damaged general Burmese, not just the E-9 domain
- why decoding-time repetition penalties made things worse
- why scaling the adapter down to 50% "fixed" it — that scales down a corrupted update
  toward the untouched base model, which is why quality also fell back to base level

## What this invalidates

- The λ-scaling conclusion ("adapter is over-trained, reduce LR") — wrong diagnosis.
- The hyperparameter suspicion (epochs, learning rate, rank) — untested, since no run
  ever had correct gradients.
- The dataset suspicion. The v2 dataset cleanup (see `scripts/build_phase3_v2.py`) was
  worth doing on its own merits, but it was not the cause: loss stayed at ~19 with the
  cleaned data, and plain transformers scores **1.297** on the *uncleaned* data.

## Standing check for future runs

Before trusting any training run on this project, on one real collated example:

1. `argmax(logits[i]) == input_ids[i+1]` must be high, and `== input_ids[i]` must be ~0%.
2. The model's reported loss must match a hand-recomputed `cross_entropy` with the
   standard shift.
3. Logits must respect the model's `final_logit_softcapping`.
4. The first logged training loss must be well under `ln(vocab_size)`.

These are implemented as hard `assert` gates in the plain-transformers training cell, so
a broken run stops before spending GPU time rather than after.

## Consequence for the stack

Training moves to **plain transformers + peft + TRL**, no Unsloth. Measured on a free T4
with the 904-token worst-case example: 6.35 GB after model+LoRA, **10.72 GB peak** for a
full forward/backward/optimizer step, 4.6 GB headroom. Slower than Unsloth but correct.

Unsloth-specific workarounds recorded earlier in `SPIKE_RUN_STATE.md` (Bug 2's compiled-
cache clear, Bug 3's separate-kernel rule for training vs generation) no longer apply.
Bug 4's `merge_and_unload()` requirement is a peft/bitsandbytes issue and still stands.
