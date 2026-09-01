"""
Phase 0 - Tokenizer fertility check.

Measures how expensive Burmese script is, per tokenizer, relative to English
carrying the SAME meaning. That ratio drives base-model choice, max_seq_length,
VRAM budget, and therefore whether free Colab is enough.

Run:  .venv/Scripts/python.exe scripts/phase0_tokenizer_fertility.py
"""
import sys
import io

# Windows console defaults to cp949 here; force UTF-8 so Burmese/Korean print.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from transformers import AutoTokenizer  # noqa: E402

# Parallel corpus: same meaning in Burmese and English, drawn from the actual
# E-9/EPS domain this project targets. Parallel matters -- comparing unrelated
# sentences would measure content length, not tokenizer efficiency.
PAIRS = [
    ("ကျွန်တော် E-9 ဗီဇာနဲ့ ကိုရီးယားမှာ အလုပ်လုပ်နေပါတယ်။",
     "I am working in Korea on an E-9 visa."),
    ("အလုပ်ခွင် ပြောင်းချင်ရင် ဘာလုပ်ရမလဲ။",
     "What should I do if I want to change my workplace?"),
    ("စာချုပ် သက်တမ်း တိုးဖို့ ဘယ်လို လျှောက်ရမလဲ။",
     "How do I apply to extend my contract?"),
    ("ပြန်လည်ဝင်ရောက်ခွင့် အထူးစနစ်အတွက် ဘာတွေ လိုအပ်သလဲ။",
     "What is required for the re-entry special system?"),
    ("အလုပ်သမား အာမခံကြေးကို ဘယ်သူက ပေးရမလဲ။",
     "Who has to pay the worker insurance premium?"),
    ("E-7-4 ဗီဇာ ပြောင်းဖို့ အမှတ် ဘယ်လောက် လိုအပ်သလဲ။",
     "How many points do I need to switch to an E-7-4 visa?"),
    ("လစာ မရသေးရင် ဘယ်မှာ တိုင်ကြားရမလဲ။",
     "Where do I file a complaint if my wages are unpaid?"),
    ("ကျွန်တော့် ပတ်စ်ပို့ သက်တမ်း ကုန်သွားပါပြီ။",
     "My passport has expired."),
]

# Korean official terms the model must reproduce verbatim in its answers.
KOREAN_TERMS = "사업장 변경 고용센터 출입국관리사무소 외국인등록증 재입국특례"

MODELS = [
    ("Qwen/Qwen3-8B",                     "candidate base (current gen)"),
    ("Qwen/Qwen2.5-7B-Instruct",          "original plan's base"),
    ("BAAI/bge-m3",                       "Phase 2 retrieval candidate (XLM-R)"),
    ("google/mt5-base",                   "multilingual reference"),
    ("facebook/nllb-200-distilled-600M",  "explicit Burmese support (floor)"),
    # Gated -- need an HF token + accepted license. Included so the failure is explicit.
    ("meta-llama/Llama-3.1-8B",           "GATED"),
    ("google/gemma-3-4b-it",              "GATED"),
]

my_text = " ".join(p[0] for p in PAIRS)
en_text = " ".join(p[1] for p in PAIRS)

rows = []
for name, note in MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
    except Exception as e:
        msg = str(e).split("\n")[0][:70]
        rows.append((name, note, None, None, None, None, None, None, msg))
        print(f"[skip] {name}: {msg}", flush=True)
        continue

    n_my = len(tok(my_text, add_special_tokens=False)["input_ids"])
    n_en = len(tok(en_text, add_special_tokens=False)["input_ids"])
    n_ko = len(tok(KOREAN_TERMS, add_special_tokens=False)["input_ids"])

    my_per_char = n_my / len(my_text)
    en_per_char = n_en / len(en_text)
    ratio = n_my / n_en  # <-- the decision number

    rows.append((name, note, len(tok), n_my, n_en, my_per_char,
                 en_per_char, ratio, n_ko))
    print(f"[ok]   {name}: my={n_my} en={n_en} ratio={ratio:.2f}x", flush=True)

print()
print(f"Burmese sample: {len(my_text)} chars | English sample: {len(en_text)} chars")
print(f"({len(PAIRS)} parallel sentence pairs, same meaning both sides)")
print()

hdr = (f"| {'Model':<36} | {'Vocab':>7} | {'MY tok':>6} | {'EN tok':>6} | "
       f"{'MY/char':>7} | {'EN/char':>7} | {'MY/EN':>6} | {'KO terms':>8} |")
print(hdr)
print("|" + "-" * (len(hdr) - 2) + "|")
for r in rows:
    name, note, vocab, n_my, n_en, mpc, epc, ratio, n_ko = r
    if vocab is None:
        print(f"| {name:<36} | {'--':>7} | {'--':>6} | {'--':>6} | "
              f"{'--':>7} | {'--':>7} | {'--':>6} | {'--':>8} |  <- {n_ko}")
    else:
        print(f"| {name:<36} | {vocab:>7} | {n_my:>6} | {n_en:>6} | "
              f"{mpc:>7.3f} | {epc:>7.3f} | {ratio:>5.2f}x | {n_ko:>8} |")

print()
print("MY/EN = tokens Burmese costs vs English for identical meaning.")
print("Lower is better. This sets max_seq_length and VRAM, hence Colab tier.")
