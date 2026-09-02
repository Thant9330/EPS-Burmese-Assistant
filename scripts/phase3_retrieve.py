"""
Phase 3 - retrieve context for a new batch of dataset questions.

Reuses the Phase 2 approach (BGE-M3, embed Burmese question directly against the
English corpus - Phase 2 found this good enough, no translation step needed) and the
same embedding cache, so the 432 corpus chunks are not re-embedded.

Writes data/eval/phase3_contexts.json in the same shape as spike_contexts.json:
  {qid: {"my": question, "topic": topic, "ctx": [{"src","title","url","text"}, ...]}}

Run: .venv/Scripts/python.exe scripts/phase3_retrieve.py
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"
SOURCES = ROOT / "data" / "sources.json"
CACHE = ROOT / "data" / "eval" / "embcache.npz"
OUT = ROOT / "data" / "eval" / "phase3_contexts.json"
MODEL = "BAAI/bge-m3"
K = 5

# First real batch of Phase 3 questions. Weighted by what the corpus actually
# supports (checked via grep + sources.json topic tags before writing these):
# workplace change / insurance / wages / re-entry / registration are well covered.
# "contract renewal" and "E-9 -> E-7-4 transition" have almost no corpus coverage
# (same finding the existing seed set already reflects - q14/q15 there are
# refusal for exactly this reason), so those two topics get mostly refusal
# examples here on purpose, not grounded guesses.
QUESTIONS = [
    # workplace change (사업장 변경) - well covered
    ("p001", "workplace_change", "အလုပ်ရှင်က အကြောင်းမပြဘဲ ငါ့ကို အလုပ်ထုတ်ရင် အလုပ်ခွင် ပြောင်းလို့ရလား။"),
    ("p002", "workplace_change", "လုပ်ငန်းရှင်က လုပ်ငန်း ပိတ်လိုက်ရင် ငါ့ဗီဇာ ဘာဖြစ်မလဲ။"),
    ("p003", "workplace_change", "အလုပ်ခွင် ပြောင်းဖို့ ဘယ်နေရာမှာ လျှောက်ရမလဲ။"),
    ("p004", "workplace_change", "အလုပ်ရှင်က ရိုက်တာ၊ ဆဲတာ ခံရရင် အလုပ်ခွင် ပြောင်းလို့ရလား။"),
    # insurance (출국만기보험, 산재보험, 건강보험, 고용보험)
    ("p005", "insurance", "ထွက်ခွာချိန် အာမခံငွေ ဘယ်တော့ ထုတ်လို့ရမလဲ။"),
    ("p006", "insurance", "အလုပ်ခွင်ထိခိုက်မှု အာမခံ (산재보험) ကို ဘယ်သူက ပေးဆောင်ရမလဲ။"),
    ("p007", "insurance", "ကျန်းမာရေးအာမခံ (건강보험) ကြေးကို လစဉ် ဘယ်လောက် ပေးရမလဲ။"),
    ("p008", "insurance", "အလုပ်အကိုင်အာမခံ (고용보험) ကြေးကို ဘယ်သူတွေ ပေးရမလဲ။"),
    # wages (최저임금, 근로기준법, 체불)
    ("p009", "wages", "အပိုအလုပ်ချိန် (야근) အတွက် ငွေ ဘယ်လောက် ပိုရမလဲ။"),
    ("p010", "wages", "လစာ နှုန်းထား သတ်မှတ်ချက်ကို ဘယ်သူ့ဆီမှာ စစ်ကြည့်လို့ရမလဲ။"),
    ("p011", "wages", "လစာ အချိန်မှန် မရရင် ဘယ်ဌာနကို တိုင်ကြားရမလဲ။"),
    # re-entry (재입국허가) + alien registration / stay extension
    ("p012", "reentry", "၁ နှစ်ထက် ကြာကြာ ကိုရီးယားပြင်ပ သွားချင်ရင် ဘာလုပ်ရမလဲ။"),
    ("p013", "alien_registration", "နိုင်ငံခြားသား မှတ်ပုံတင်ကတ် ပျောက်သွားရင် ဘယ်လို လုပ်ရမလဲ။"),
    ("p014", "stay", "နေထိုင်ခွင့် သက်တမ်းတိုးဖို့ ဘယ်အချိန်ကတည်းက လျှောက်ရမလဲ။"),
    # contract renewal - thin corpus coverage, expect refusal
    ("p015", "contract_renewal", "အလုပ်စာချုပ် အသစ် ရေးဖို့ အလုပ်ရှင်က ငြင်းရင် ဘာလုပ်ရမလဲ။"),
    ("p016", "contract_renewal", "စာချုပ် သက်တမ်း ကုန်ခါနီးရင် ဘယ်နှစ်ရက်အလိုမှာ အသိပေးရမလဲ။"),
    # E-9 -> E-7-4 transition - almost no corpus coverage, expect refusal
    ("p017", "e7_transition", "E-9 က E-7-4 ဗီဇာ ပြောင်းချင်ရင် ဘယ်လို အရည်အချင်း လိုအပ်သလဲ။"),
    ("p018", "e7_transition", "ကျွမ်းကျင်လုပ်သား ဗီဇာ ပြောင်းဖို့ ဘယ်လောက်နှစ် အလုပ်လုပ်ဖူးရမလဲ။"),
]


def main():
    chunks = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    print(f"corpus: {len(chunks)} chunks | new questions: {len(QUESTIONS)}", flush=True)

    print(f"loading {MODEL} ...", flush=True)
    model = SentenceTransformer(MODEL)

    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        cache = {k: z[k] for k in z.files}
        print(f"embedding cache: {len(cache)} vectors loaded", flush=True)

    texts = [c["text"] for c in chunks]
    keys = [hashlib.sha1(x.encode("utf-8")).hexdigest() for x in texts]
    todo = [(k, x) for k, x in zip(keys, texts) if k not in cache]
    if todo:
        print(f"embedding {len(todo)} new corpus chunks ...", flush=True)
        vecs = model.encode([x for _, x in todo], batch_size=8,
                            normalize_embeddings=True, show_progress_bar=False)
        for (k, _), v in zip(todo, vecs):
            cache[k] = v
        np.savez(CACHE, **cache)
    C = np.stack([cache[k] for k in keys])

    sources = {s["id"]: s for s in json.loads(SOURCES.read_text(encoding="utf-8"))["sources"]}

    print("embedding new questions ...", flush=True)
    Q = model.encode([q[2] for q in QUESTIONS], normalize_embeddings=True)

    out = {}
    for i, (qid, topic, question) in enumerate(QUESTIONS):
        sims = C @ Q[i]
        top = np.argsort(-sims)[:K]
        ctx = []
        for j in top:
            src_id = chunks[j]["source_id"]
            src = sources.get(src_id, {})
            ctx.append({"src": src_id, "title": src.get("title", src_id),
                       "url": src.get("url", ""), "text": chunks[j]["text"],
                       "sim": round(float(sims[j]), 3)})
        out[qid] = {"my": question, "topic": topic, "ctx": ctx}
        print(f"{qid:<6} {topic:<18} top_sim={ctx[0]['sim']:.3f} "
              f"top_src={ctx[0]['src']}")

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
