# Training spike — run state (updated 2026-09-02)

**Decision: base model locked to SEA-LION v3 9B.** E2B is dropped from the
comparison — its tokenizer advantage doesn't survive a bf16-less T4 (Unsloth
forces fp32 for gemma4, so E2B loaded *larger* than 9B in 4-bit: 7.45GB vs
6.16GB), and this matches Phase 0's original choice before the re-check.
`notebooks/spike_compare_base_models.ipynb` now only runs 9B. Bug 1 (E2B's
`text=` keyword crash) is moot and no longer needs fixing.

Status: **Bug 2 fixed. Bug 3 root cause found and fixed: Unsloth's generation
path is broken on this stack, not the model.** Training via Unsloth works
correctly (confirmed twice). Generation via `FastLanguageModel`/Unsloth
produces garbage regardless of prompt, template, or token config — proven by
a clean-kernel test with plain `transformers`+`bitsandbytes` (Unsloth never
imported) generating perfectly coherent English on the identical checkpoint.
**Practical fix: train with Unsloth, generate/evaluate with plain
`transformers`+`peft` in a separate process** — Unsloth monkeypatches
`transformers` process-wide and irreversibly, so the two can't share a
kernel. See Bug 3 for the full trail.

**New finding that matters more than the bug: with generation actually
working, the untrained base model already produces decent, grounded,
on-topic Burmese** on the one seed example tested — which cuts against the
project's founding premise that the base model "visibly fails" at this task.
The one 30-step toy LoRA adapter trained so far **makes output worse, not
better** (collapses to near-empty output) — almost certainly because 30 steps
on 16 examples is far too little/unstable, not because the direction is
wrong. **This needs your judgment as the native-Burmese reader before Phase 3
decisions are finalized** — see "What this means for the project" below.

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

## Bug 3 — generation returns unusable output — ROOT CAUSE FOUND (2026-09-02)

Both BEFORE and AFTER generation on the two held-out questions initially
failed, in two different ways: empty string before training, 220 repeated
newline characters after. Investigation ruled out, in order:

1. **Wrong eos/pad ids** — `generate()` was rewritten to read
   `model.generation_config.eos_token_id` fresh at call time and to fix the
   `tok.pad_token_id or tok.eos_token_id` bug (breaks when `pad_token_id==0`,
   which it legitimately is here). Diagnostics confirmed ids were read
   correctly (`eos_ids=[1,107]`, `pad_id=0`) — the degenerate output
   persisted regardless, proving this wasn't the cause.
2. **Prompt/template mismatch** — printed the literal rendered prompt text.
   It's textbook-correct Gemma chat format: `<bos><start_of_turn>user\n...
   <end_of_turn>\n<start_of_turn>model\n`, no double-BOS, correct
   structure. Not the cause either.
3. **Fundamentally broken model/quantization** — tested a trivial prompt
   ("Hello, how are you?") on the loaded model: greedy decoding gave
   immediate EOS (identical symptom), sampled decoding gave pure gibberish
   (random tokens across scripts, glitch text). But training loss had
   decreased sensibly (17.77→9.19), meaning the model's forward pass
   produces meaningful logits during training — so the problem is specific
   to the **inference/`generate()`** code path, not the weights.
4. **Confirmed: it's Unsloth's generation path, not the model.** Attempting
   to bypass `FastLanguageModel.for_inference()` by calling `.eval()` +
   `.generate()` directly didn't help — because Unsloth monkeypatches
   `transformers`' `Gemma2Model.forward` **globally and process-wide** the
   moment `import unsloth` runs, regardless of how a model is later loaded.
   Confirmed by attempting a "plain" `AutoModelForCausalLM.from_pretrained`
   load in the *same* (already-`unsloth`-imported) kernel — it still routed
   through Unsloth's patched `LlamaModel_fast_forward` and crashed with
   `AttributeError: 'Gemma2Model' object has no attribute 'max_seq_length'`
   (an attribute only `FastLanguageModel`'s loader sets). **Decisive test**:
   restarted the Colab kernel (`os.kill(os.getpid(), 9)` — packages and
   `spike_seed.jsonl` survive), and in that fresh kernel — `unsloth` never
   imported — loaded the identical checkpoint via plain
   `transformers.AutoModelForCausalLM` + `BitsAndBytesConfig(load_in_4bit=True)`
   and generated on the same trivial prompt:

   ```
   CLEAN (no unsloth ever imported) greedy decoded:
   "I am an AI, so I don't have feelings, but I'm here and ready to assist you!"
   ```

   Perfectly coherent. **The checkpoint, the 4-bit quantization, and the
   prompt were never the problem — Unsloth 2026.8.22's Gemma2
   fast-generation path is broken on this T4/legacy-tokenizer setup.**

**Practical fix — train and generate in separate processes.** Since Unsloth's
patch is global and irreversible within a process, training (fast, use
Unsloth, confirmed working) and generation/evaluation (must avoid Unsloth's
patch entirely) cannot coexist in one kernel. Workflow going forward:
1. Train with Unsloth as before; `SFTTrainer` already saves the LoRA adapter
   to `out/checkpoint-N` on disk.
2. In a **separate kernel that never imports `unsloth`**, load the base model
   via plain `transformers` + `BitsAndBytesConfig`, then apply the adapter
   with `peft.PeftModel.from_pretrained(base_model, "out/checkpoint-N")`, and
   generate from there. This applies to Phase 4 (baseline eval) and Phase 6
   (final eval) too — both need this clean generation path, not Unsloth's.

## What this means for the project — needs your judgment

With generation actually working, real BEFORE/AFTER output for held-out Q1
(grounded, about 출국만기보험/departure guarantee insurance) and Q2 (refusal,
about becoming a Korean citizen):

```
=== BEFORE (base model, no adapter) ===
[Q1] ထွက်ခွာချိန် အာမခံ (출국만기보험) ဆိုသည်မှာ ကော်ရီးယား လုပ်ငန်းမှ ပြန်လည်ထွက်ခွာသော
     အလုပ်သမားများအတွက် ကာကွယ်ရေး စီမံကိန်း တစ်ခုဖြစ်သည်။ ဤအာမခံ စီမံကိန်းသည်
     လုပ်ငန်းရှင်များ ကြိုးစားရမည့် စည်းမျဉ်းများ ပါဝင်သည်။
     (EPS 법률 제13조에 따라 보험 정책 또는 신탁 증서를 구매해야 하는 고용주는 다음
     요건을 모두 충족해야 합니다. - EPS 법률 제12... [cut off at max_new_tokens=220]
[Q2] အဆိုပါ ပညာရေး ပြည်သူ့ လုပ်ငန်းမှာ ကိုရီးယား နိုင်ငံသား ဖြစ်လာရန် နည်းလမ်းအကြောင်း
     မရှိပါ။ [i.e. roughly "there's no method described in this material for
     becoming a Korean citizen" - an appropriate-shaped refusal]

=== AFTER (base + 30-step toy LoRA adapter) ===
[Q1] " အ" then the model produces nothing further of substance (pads out to
     max_new_tokens without stopping)
[Q2] " အ" then same collapse
```

**Two things are true at once, and both matter:**
- The **base model already produces grounded, on-topic Burmese** with a real
  legal citation on Q1, and an appropriately-shaped refusal on Q2 — this cuts
  against the plan's founding premise ("the base model visibly fails at
  Burmese, that's the measurable delta"). It's one example, greedy-decoded,
  not a rigorous eval, but it's a real data point that should have been
  visible from Phase 0/2 and wasn't, because generation was broken until now.
- The **toy 30-step LoRA (16 examples, `lr=2e-4`) makes output worse, not
  better** — collapses to a single character. `final_loss` was still ~9.19,
  very high — this reads as **severely undertrained/unstable**, not as
  evidence the *approach* is wrong. The real Phase 3 dataset (800-1500
  examples, more epochs, presumably a saner LR schedule) is a completely
  different regime from this throwaway spike config, so this collapse
  doesn't predict what a real training run would do.

**Recommended next step:** you read the BEFORE Burmese above yourself (no
metric substitutes for a native reader, per the project's own rule) and judge
whether it's actually good — check the Korean-term handling, whether the
citation is genuine or plausible-sounding hallucination, whether the answer
is fully grounded in context or drifting. If BEFORE already looks solid, the
project's core premise needs re-examining before Phase 3 (maybe the
measurable gain is elsewhere: consistency at scale, refusal calibration,
Korean-term glossing format, not raw "can it answer at all"). If BEFORE has
real problems a fluent reader would catch, the premise still holds and it's
safe to continue toward Phase 3, treating this specific 30-step AFTER
collapse as "config needs tuning" rather than "approach is broken."

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

1. **You read the BEFORE Burmese samples above and judge them** — this is the
   actual blocking decision now, not a technical bug.
2. Depending on your read: either accept the premise needs adjusting (base
   model is already decent; figure out what the real measurable gain from
   fine-tuning should be before writing 800-1500 examples aimed at the wrong
   target), or confirm the premise holds and proceed.
3. If proceeding: retrain with a saner config (more steps, e.g. 100-300;
   consider a lower `learning_rate`, e.g. `1e-4`) using the same
   train-with-Unsloth pipeline, and re-evaluate using the clean
   `transformers`+`peft` generation pipeline (never in the same kernel as the
   training import). Update `notebooks/spike_compare_base_models.ipynb`
   to formally split into a training cell (Unsloth) and a separate
   evaluation cell/notebook (plain transformers + peft), since the two must
   never share a kernel.
4. Carry the clean-generation requirement forward into Phase 4 (baseline
   eval) and Phase 6 (final eval) planning — both need this same
   Unsloth-free generation pipeline.
