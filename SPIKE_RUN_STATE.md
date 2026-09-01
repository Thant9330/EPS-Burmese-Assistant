# Training spike — run state (paused 2026-09-01)

Status: **spike has not produced numbers yet.** Four Colab runs on a free T4; no
model has completed a single LoRA step. Both models still fail, for two
different and now precisely-located reasons. Everything below is from real
tracebacks, not guesses.

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

## Bug 1 — E2B generation (fix applied, NOT yet verified by a run)

```
File "unsloth_zoo/tokenizer_utils.py", line 602, in patched_call
    return original_call(self, images=images, text=text, videos=videos, **kwargs)
File "transformers/models/gemma4/processing_gemma4.py", line 130, in __call__
    elif not isinstance(text, list) and not isinstance(text[0], str):
TypeError: 'NoneType' object is not subscriptable
```

Gemma 4 loads a **processor**, not a plain tokenizer. Its first positional
parameter is `images`. So `tok(text, ...)` bound our prompt to `images` and left
`text=None`. Fix: pass `text=` as a keyword — `tok(text=text, ...)`. Applied to
the notebook; needs a run to confirm.

Related, already fixed: `needs_parts()` must probe with `tokenize=True`. A
`tokenize=False` probe passes for both model families and reports `False` for
Gemma 4, which is wrong — the multimodal path only engages when tokenizing.

## Bug 2 — 9B training dtype clash (NOT fixed, hypothesis only)

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

1. Fresh kernel, `rm -rf /content/unsloth_compiled_cache`, re-run. This tests
   the E2B keyword fix and the 9B stale-cache hypothesis in one pass.
2. If the 9B still clashes, print `trainer.accelerator.mixed_precision` and
   force the autocast dtype directly.
3. Only once both models take 30 steps: read the BEFORE/AFTER Burmese on the two
   held-out questions and decide the base model. That judgement needs a native
   reader — no metric in this notebook substitutes for it.
4. Re-run the comparison on a bf16-capable GPU (A100/L4) before committing to
   E2B or 9B, since the T4's lack of bf16 is currently distorting the VRAM
   comparison against E2B.
