# Training spike — run state (updated 2026-09-02)

**Decision: base model locked to SEA-LION v3 9B.** E2B is dropped from the
comparison — its tokenizer advantage doesn't survive a bf16-less T4 (Unsloth
forces fp32 for gemma4, so E2B loaded *larger* than 9B in 4-bit: 7.45GB vs
6.16GB), and this matches Phase 0's original choice before the re-check.
`notebooks/spike_compare_base_models.ipynb` now only runs 9B. Bug 1 (E2B's
`text=` keyword crash) is moot and no longer needs fixing.

Status: **Bug 2 is fixed — 9B trains cleanly. But generation is broken (new
Bug 3), so Burmese quality still cannot be judged.** The 5th Colab run
completed all 30 LoRA steps with no dtype error. However both BEFORE and AFTER
generation on the held-out questions returned unusable output (empty strings
before, an unbroken string of newline characters after) — see Bug 3 below.
**The spike has not yet met its actual goal** (confirm the base model
visibly fails and LoRA visibly fixes it in Burmese) and Phase 3 should not
start until this is resolved or a workaround is found.

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

## Bug 3 — generation returns unusable output (NEW, NOT fixed)

Both BEFORE and AFTER generation on the two held-out questions failed, in two
different ways:

```
--- BEFORE training ---
[ 0 ] ''
[ 1 ] ''
--- AFTER training ---
[ 0 ] '\n\n\n\n\n...' (220 newline characters, no other content)
[ 1 ] '\n\n\n\n\n...' (220 newline characters, no other content)
```

BEFORE: immediate EOS, nothing generated. AFTER: the model runs to the full
`max_new_tokens=220` budget but predicts nothing but the newline token —
never emits real content, never emits EOS either. **This is not a Burmese
quality problem, it's a generation-config/EOS problem that made the whole
BEFORE/AFTER comparison uninformative** — we still don't know whether LoRA
training actually improves Burmese output on this stack.

Suspect: during trainer setup a warning fired —
`The tokenizer has new PAD/BOS/EOS tokens that differ from the model config
and generation config. ... Updated tokens: {'eos_token_id': 1}`. This rewrites
`model.generation_config.eos_token_id` to `1` mid-session. But `generate()` in
the notebook computes `pad_token_id=tok.pad_token_id or tok.eos_token_id` from
the **tokenizer**, not from the (now-patched) model generation config — if
those disagree, `model.generate()` may never see a stop condition it
recognizes, and/or padding gets mishandled, producing exactly this kind of
degenerate run-to-max-length output. The empty-string BEFORE result (before
any patching) suggests a related but distinct mismatch already existed at
load time.

**Not yet tried:**
1. Print `tok.eos_token_id`, `tok.pad_token_id`, and `model.generation_config.eos_token_id`
   right after load and again after `SFTTrainer` construction, to see exactly
   where they diverge.
2. Explicitly set `tok.pad_token = tok.eos_token` after load, and pass
   `eos_token_id=model.generation_config.eos_token_id` explicitly into
   `model.generate(...)` instead of relying on the tokenizer's possibly-stale
   value.
3. Try `skip_special_tokens=False` on one sample to see if the model is
   actually emitting a real EOS-like token that `skip_special_tokens=True` is
   silently eating, versus truly never stopping.

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

1. Diagnose and fix Bug 3 (generation returns empty/degenerate output) per the
   three untried steps listed above — start with printing the three
   eos/pad token id values at both checkpoints, that will likely make the
   mismatch obvious.
2. Once generation produces real text: re-run BEFORE/AFTER on the held-out
   questions (no need to retrain — Bug 3 is a generation-time issue, so it can
   be re-tested against a freshly loaded/trained model, or even by fixing
   `generate()` and calling it against the already-saved `out/checkpoint-30`
   adapter to avoid a full retrain).
3. Read the BEFORE/AFTER Burmese and judge it (native-reader call, no metric
   substitutes) — only then is the spike's actual goal met and Phase 3 safe
   to start.
