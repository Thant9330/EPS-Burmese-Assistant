# Training spike — run state (updated 2026-09-02)

**Decision: base model locked to SEA-LION v3 9B.** E2B is dropped from the
comparison — its tokenizer advantage doesn't survive a bf16-less T4 (Unsloth
forces fp32 for gemma4, so E2B loaded *larger* than 9B in 4-bit: 7.45GB vs
6.16GB), and this matches Phase 0's original choice before the re-check.
`notebooks/spike_compare_base_models.ipynb` now only runs 9B. Bug 1 (E2B's
`text=` keyword crash) is moot and no longer needs fixing.

Status: **spike has not produced numbers yet.** Four Colab runs on a free T4; no
model has completed a single LoRA step. The remaining blocker is Bug 2 below,
which is specific to 9B. Everything below is from real tracebacks, not
guesses.

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

## Bug 2 — 9B training dtype clash (NOT fixed, hypothesis only, now the only blocker)

```
File "unsloth/kernels/utils.py", line 1167, in matmul_lora
    out.addmm_(XA, B.to(dtype), alpha = s)
RuntimeError: self and mat2 must have the same dtype, but got Half and BFloat16
```

Reached inside `backward()` -> gradient checkpointing -> `apply_lora_qkv`, under
`torch.amp.autocast_mode.decorate_fwd`.

Ruled out: it is not the model weights. The dtype census shows **no bfloat16
parameters at any point**:
- after load: `{float16: 170, uint8: 294}`
- after `get_peft_model`: `{float16: 170, uint8: 294, float32: 588}`

Also already tried and insufficient: `dtype=torch.float16` on `from_pretrained`,
and explicit `fp16=True, bf16=False` on `SFTConfig`.

So the bf16 is coming from the **autocast context**, not the weights. Leading
hypothesis, untested: `/content/unsloth_compiled_cache/UnslothSFTTrainer.py` is
a stale compiled artifact generated during the first (differently-configured)
run of the session and reused afterwards — the traceback runs through it. Next
thing to try: `rm -rf /content/unsloth_compiled_cache` before the run, on a
fresh kernel. If that does not do it, inspect the accelerator's mixed-precision
setting at train time (`trainer.accelerator.mixed_precision`) rather than
trusting the SFTConfig flags.

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

1. Fresh kernel, `rm -rf /content/unsloth_compiled_cache`, re-run 9B. Tests the
   stale-cache hypothesis for Bug 2.
2. If it still clashes, print `trainer.accelerator.mixed_precision` and force
   the autocast dtype directly (or run on a bf16-capable GPU — A100/L4 — where
   the clash may not occur at all, since it's the T4's lack of bf16 hardware
   that puts autocast in a weird state to begin with).
3. Once 9B takes 30 steps: read the BEFORE/AFTER Burmese on the two held-out
   questions. This is now a stack-verification check, not a model decision —
   the base model is locked. A native reader should still confirm the output
   is coherent and grounded before moving to Phase 3.
