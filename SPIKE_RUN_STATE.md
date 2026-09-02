# Training spike — run state (updated 2026-09-02)

**Decision: base model locked to SEA-LION v3 9B.** E2B is dropped from the
comparison — its tokenizer advantage doesn't survive a bf16-less T4 (Unsloth
forces fp32 for gemma4, so E2B loaded *larger* than 9B in 4-bit: 7.45GB vs
6.16GB), and this matches Phase 0's original choice before the re-check.
`notebooks/spike_compare_base_models.ipynb` now only runs 9B. Bug 1 (E2B's
`text=` keyword crash) is moot and no longer needs fixing.

Status: **Bug 2 is fixed — 9B trains cleanly. Bug 3 (generation) is still
broken and is deeper than first thought** — the eos/pad-token-id hypothesis
was tested with diagnostics and ruled out; the model's raw next-token
predictions are themselves degenerate (immediate EOS before training, endless
newline after). See Bug 3 below for the full diagnostic trail and three
remaining hypotheses. **The spike has not yet met its actual goal** (confirm
the base model visibly fails and LoRA visibly fixes it in Burmese) and
Phase 3 should not start until this is resolved.

## Environment (confirmed working)

- Colab free tier, **Tesla T4, 15360 MiB**, `bf16 supported: False`
- `unsloth 2026.8.22`, `transformers 5.5.0`, `torch 2.11.0+cu128`, Triton 3.6.0
- Install cell works as-is. Packages persist across kernel restarts.

## Seed data — solved

The repo is private, so the notebook's raw-GitHub URL 404s and it falls back to
`files.upload()`, which blocks on a native file picker that automation cannot
drive. Worked around by writing `spike_seed.jsonl` into the Colab CWD from a
gzip+base64 blob cell (57690 bytes, byte-identical to
`data/samples/spike_seed.jsonl`), and teaching the seed cell to prefer a local
file. That cell now reports:

```
loaded from local spike_seed.jsonl
18 examples | grounded= 13 refusal= 5
train= 16  held_out= 2
held-out Q1: ထွက်ခွာချိန် အာမခံ (출국만기보험) ဆိုတာ ဘာလဲ။ | grounded
held-out Q2: ကိုရီးယား နိုင်ငံသား ဖြစ်ချင်ရင် ဘယ်လိုလုပ်ရမလဲ။ | refusal
```

The blob cell is **Colab-session-only** — it is deliberately not committed, to
avoid duplicating the seed data in the repo. Recreate it with:
`gzip -9` + `base64` of `data/samples/spike_seed.jsonl`, written to
`spike_seed.jsonl` in the Colab CWD.

## Measurements obtained so far

| model | load | VRAM at load | content format |
|---|---|---|---|
| Gemma-SEA-LION-v4.5-E2B-IT | ~60 s | **7.45 GB** | parts (`[{"type":"text",...}]`) |
| Gemma-SEA-LION-v3-9B-IT | ~94 s | **6.16 GB** | bare string |

**This already complicates the base-model decision.** E2B is supposed to be the
cheap option, but Unsloth refuses fp16 for gemma4 (`Using float16 precision for
gemma4 won't work! Using float32.`), so on a T4 the ~2B model loads *larger*
than the 9B in 4-bit — 7.45 GB vs 6.16 GB. The 1.85x tokenizer advantage does
not survive contact with a bf16-less GPU. Worth re-testing on an A100/L4 where
bf16 exists before concluding anything.

## Bug 1 — E2B generation (moot, model dropped)

E2B is no longer in scope — see the decision note at the top. Left here only
so the original diagnosis isn't lost: Gemma 4 loads a **processor**, not a
plain tokenizer, whose first positional parameter is `images`, so `tok(text, ...)`
silently bound the prompt to `images` and left `text=None`. Would have been
fixed by `tok(text=text, ...)`.

## Bug 2 — 9B training dtype clash — FIXED (2026-09-02)

```
File "unsloth/kernels/utils.py", line 1167, in matmul_lora
    out.addmm_(XA, B.to(dtype), alpha = s)
RuntimeError: self and mat2 must have the same dtype, but got Half and BFloat16
```

**Root cause confirmed: stale `/content/unsloth_compiled_cache`.** A fresh
kernel with `!rm -rf /content/unsloth_compiled_cache` run right after the
install cell, before unsloth is imported anywhere, resolved it completely.
`trainer.accelerator.mixed_precision` printed `fp16` as expected (not `bf16`),
confirming the earlier autocast-context diagnosis was right — it just needed
the stale compiled artifact gone. 30/30 steps completed:

```
loaded in 371.4 s | VRAM 6.16 GB | content_parts = False
accelerator.mixed_precision = fp16
trained 30 steps in 460.6 s | 15.35 s/step | loss 12.2985 | peak 7.96 GB
```

Loss trend across logged steps: 17.77 → 14.90 → 11.56 → 10.57 → 9.80 → 9.19
(step 5 → 30). Monotonic decrease, no NaN/divergence — mechanically the
training loop is healthy on this stack.

The fix is now baked into the notebook (`notebooks/spike_compare_base_models.ipynb`,
cell after the install cell) so it doesn't need rediscovering.

## Bug 3 — generation returns unusable output (STILL NOT fixed — deeper than token ids)

Both BEFORE and AFTER generation on the two held-out questions fail, in two
different ways:

```
--- BEFORE training ---
[ 0 ] ''
[ 1 ] ''
--- AFTER training ---
[ 0 ] '\n\n\n\n\n...' (220 newline characters, no other content)
[ 1 ] '\n\n\n\n\n...' (220 newline characters, no other content)
```

**First hypothesis (wrong eos/pad ids) — tested and ruled out.** `generate()`
was rewritten to read `model.generation_config.eos_token_id` fresh at call
time (not the tokenizer's possibly-stale value) and to fix the
`tok.pad_token_id or tok.eos_token_id` bug (breaks when `pad_token_id==0`,
since `0` is falsy in Python — confirmed real: `tok.pad_token_id` really is
`0` here). Diagnostic prints added to every `generate()` call confirm the ids
are now read correctly:

```
tok bos/eos/pad ids: 2 1 0
model.generation_config.eos_token_id (at load): [1, 107]
    [diag] eos_ids= [1, 107] pad_id= 0 gen_len= 1 gen_ids[:15]= [1]      <- BEFORE, both calls
model.generation_config.eos_token_id (post-trainer): [1, 107]
    [diag] eos_ids= [1, 1, 107] pad_id= 0 gen_len= 220 gen_ids[:15]= [108]*15   <- AFTER, both calls
```

**This proves the real behavior, not a code bug:**
- **BEFORE:** the model's very first generated token *is* `1` (a real member
  of `eos_ids`) — `gen_len=1`. The untrained base model is correctly
  recognizing this exact prompt as "already over" and stopping immediately.
  `generate()` is working correctly; the model + prompt combination produces
  a 1-token (immediate-EOS) response.
- **AFTER:** the model outputs token `108` (`\n`) 220 times straight —
  `108` is *not* in `eos_ids`, so it never stops on its own, hence
  `gen_len=220` (hit `max_new_tokens`). Not an eos-detection bug either —
  the model just never predicts anything but newline.

So the root cause is upstream of `generate()` — most likely one of:
1. **Prompt/template mismatch.** `generate()` renders only a single "user"
   turn via `apply_chat_template(..., add_generation_prompt=True)`, while the
   training data (`build_text()`) renders a full user+assistant 2-turn
   conversation through the *same* template but without
   `add_generation_prompt`. Unsloth logs "We found double BOS tokens - we
   shall remove one automatically" for the **training** text specifically
   (not generation) — meaning `build_text()`'s manually-rendered text plus
   the trainer's own default tokenization already double up on `<bos>`, only
   caught for the train path. Worth rendering and printing the literal
   `text` string fed to `generate()` and to `build_text()` side by side to
   check they actually agree on structure past the BOS handling.
2. **Severely undertrained LoRA.** `final_loss=12.2985` and even step-30 loss
   ~9.19 is very high for cross-entropy — no sign of the model producing
   coherent output at all. 30 steps on 16 examples (effective batch 4, so
   ~7.5 epochs) at `lr=2e-4`, `r=16` may simply not be enough to move the
   model off a degenerate mode once it's in one, especially for a task this
   far from the base model's native behavior. This is a real possibility
   independent of (1).
3. `aisingapore/Gemma-SEA-LION-v3-9B-IT` loaded **"as a legacy tokenizer"**
   per Unsloth's own log line — worth checking if that legacy path handles
   `apply_chat_template` differently than the fast tokenizer would, which
   could explain a malformed prompt underlying both (1) and the BEFORE
   immediate-stop behavior.

**Not yet tried:**
1. Print the literal rendered `text` string (not token ids) for one held-out
   question, both from `generate()`'s path and from `build_text()`'s path,
   and diff them by eye.
2. Try generation with `parts=True` (the "typed content" format) even though
   `needs_parts()` picked `False`, in case the probe itself is unreliable on
   this legacy-tokenizer path.
3. If (1)/(2) don't explain it, test whether more steps / lower LR fixes the
   post-training degeneracy — i.e. rule out "just needs more training" before
   assuming a template bug.

## Operational notes for the next session

- **VRAM does not free itself between models.** `except Exception as e:` binds
  the traceback, which pins `run()`'s frame, which pins the model. Cleanup must
  happen *after* the except block exits (already restructured this way in the
  notebook). Even so, 1.97–6.51 GB stayed allocated after a failure, so
  **restart the kernel between attempts** rather than trusting `empty_cache()`.
  A cell containing `os.kill(os.getpid(), 9)` does this; `spike_seed.jsonl` and
  the installed packages survive it.
- After a restart, re-run in order: seed cell -> `MODELS` cell -> experiment cell.
- The experiment cell exceeds the MCP 120 s tool limit and completes as a
  background task — that is normal, not a hang.
- Colab `files.upload()` cannot be driven by automation; if a cell blocks on it
  the kernel queues everything behind it and later cells look wedged.

## Next actions, in order

1. Print the literal rendered prompt `text` (not ids) from both `generate()`
   and `build_text()` for the same held-out question, and diff by eye —
   cheapest way to catch a template mismatch, and doesn't require a full
   train run (only a model load, ~90-370s).
2. If templates look fine, retry generation with `parts=True` to rule out a
   bad `needs_parts()` probe on this legacy-tokenizer path.
3. If still stuck, treat it as "undertrained, not broken" — bump `STEPS`
   (e.g. 100) and/or lower `learning_rate`, re-run, and see if AFTER output
   stops being degenerate. This doesn't require solving (1)/(2) first, so it
   can be tried in parallel or first if it's cheaper to just try.
4. Once generation produces real text: read the BEFORE/AFTER Burmese and
   judge it (native-reader call, no metric substitutes) — only then is the
   spike's actual goal met and Phase 3 safe to start.
