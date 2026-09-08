"""
EPS Burmese Assistant — Hugging Face Space (ZeroGPU).

Answers Korean labour-law questions in Burmese, grounded in the statutes.

Architecture note: retrieval runs on CPU because the corpus embeddings are precomputed and
shipped with the Space, so bge-m3 only has to embed one short query per request. Only
generation needs the GPU, which keeps the @spaces.GPU window — and therefore the visitor's
daily quota — as small as possible.

The adapter is loaded UNMERGED. merge_and_unload() on 4-bit weights rounds the LoRA delta
away and silently returns the base model.
"""
import collections
import json
import os
import time

import gradio as gr
import numpy as np
import spaces
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TextIteratorStreamer)

BASE = "aisingapore/gemma2-9b-cpt-sea-lionv3-instruct"
ADAPTER = "MYOTHANTZIN/eps-burmese-sealion-9b-lora"

SYSTEM = ("You are an assistant for Myanmar workers on E-9/EPS visas in Korea. "
          "Answer ONLY from the provided context. Reply in Burmese. Keep Korean official "
          "terms in Korean with a Burmese gloss. Always cite the source. If the context "
          "does not answer the question, say so and refuse — never invent a rule.")

# Stripped from the training data so the model would not spend capacity memorising it;
# the application appends it instead.
DISCLAIMER = ("*မှတ်ချက် — ဤအချက်အလက်သည် ရည်ညွှန်းချက်သာဖြစ်ပြီး တရားဝင် အာဏာမရှိပါ။ "
              "လက်ရှိစည်းမျဉ်းကို 출입국관리사무소 သို့မဟုတ် 고용센터 တွင် အတည်ပြုပါ။*")

# ---------------------------------------------------------------- retrieval (CPU)
CORPUS = [json.loads(l) for l in open("corpus_meta.jsonl", encoding="utf-8") if l.strip()]
EMB = np.load("corpus_emb.npy")
SRC_OF = [c["source_id"] for c in CORPUS]
assert len(CORPUS) == EMB.shape[0], "corpus and embeddings out of sync"

encoder = SentenceTransformer("BAAI/bge-m3", device="cpu")


def retrieve(question, k=5, per_source=1):
    """Diverse top-k: best chunks, at most `per_source` from any one document.

    Without the per-source cap the 100-chunk statutes crowd out the 2-chunk HiKorea
    pages, which measured 77% hit@5. Capping raises it to 82% at identical context length.
    """
    q = encoder.encode([question], normalize_embeddings=True)[0]
    sims = EMB @ q
    picked, seen = [], collections.Counter()
    for j in np.argsort(-sims):
        s = SRC_OF[j]
        if seen[s] >= per_source:
            continue
        picked.append((int(j), float(sims[j])))
        seen[s] += 1
        if len(picked) == k:
            break
    return picked


# ---------------------------------------------------------------- model (GPU)
tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True),
    device_map="cuda")
model = PeftModel.from_pretrained(base, ADAPTER)   # UNMERGED — see module docstring
model.eval()


@spaces.GPU(duration=90)
def generate(prompt, max_new_tokens=420):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         eos_token_id=[1, 107], pad_token_id=0)
    ids = out[0][enc["input_ids"].shape[1]:].tolist()
    finished = (1 in ids) or (107 in ids)
    return tok.decode(ids, skip_special_tokens=True).strip(), finished, len(enc["input_ids"][0])


def answer(question, k=5):
    if not question or not question.strip():
        return "မေးခွန်း ရေးထည့်ပါ။", "", ""
    t0 = time.time()

    hits = retrieve(question, k=k)
    context = "\n\n".join(f"[{CORPUS[j]['source_id']}] {CORPUS[j]['text']}" for j, _ in hits)
    prompt = SYSTEM + "\n\n### Context\n" + context + "\n\n### Question\n" + question

    gen, finished, n_prompt = generate(prompt)
    if not finished:
        gen += "\n\n*(အဖြေ ပြတ်တောက်သွားပါသည် — ထပ်မံ မေးမြန်းပါ)*"

    used = [CORPUS[j]["source_id"] for j, _ in hits]
    meta = (f"⏱ {time.time()-t0:.0f}s · {n_prompt} prompt tokens · "
            f"sources: {', '.join(used)}\n\n"
            f"❓ မသေချာပါက **고용센터** (အလုပ်ကိစ္စ) သို့မဟုတ် **출입국관리사무소** (ဗီဇာကိစ္စ) သို့ "
            f"တိုက်ရိုက် ဆက်သွယ် မေးမြန်းပါ။")
    sources = "\n\n---\n\n".join(
        f"**[{CORPUS[j]['source_id']}]** · similarity {s:.3f} · {CORPUS[j].get('title','')}\n\n"
        f"> {CORPUS[j]['text'][:700].strip()}…" for j, s in hits)
    return f"{gen}\n\n{DISCLAIMER}", meta, sources


EXAMPLES = [
    "အလုပ်ခွင် ပြောင်းဖို့ ဘယ်နေရာမှာ လျှောက်ရမလဲ။",
    "အပိုအလုပ်ချိန် အတွက် ငွေ ဘယ်လောက် ပိုရမလဲ။",
    "အလုပ်ရှင်က အလုပ်ထုတ်ခင် ဘယ်နှစ်ရက်အလို အသိပေးရမလဲ။",
    "နေထိုင်ခွင့် သက်တမ်းတိုးဖို့ ဘယ်အချိန်ကတည်းက လျှောက်ရမလဲ။",
    "ဒီနှစ် အနည်းဆုံးလုပ်အားခ ဘယ်လောက်လဲ။",
]

with gr.Blocks(title="EPS Burmese Assistant") as demo:
    gr.Markdown(
        "# EPS ဗမာ အကူအညီ · EPS Burmese Assistant\n"
        "ကိုရီးယား E-9/EPS ဗီဇာ အလုပ်သမားများအတွက် **စမ်းသပ်ဆဲ** စနစ်။\n\n"
        "Answers Korean labour-law questions in Burmese, grounded in the statutes. "
        "**Research prototype — roughly 8% of answers are confidently wrong.** "
        "Always confirm with 고용센터 or 출입국관리사무소 before acting.")
    with gr.Row():
        q = gr.Textbox(label="မေးခွန်း · Question (Burmese)", lines=2, scale=4,
                       placeholder="ဥပမာ — အလုပ်ခွင် ပြောင်းဖို့ ဘယ်နေရာမှာ လျှောက်ရမလဲ။")
        btn = gr.Button("မေးမည် · Ask", variant="primary", scale=1)
    out = gr.Markdown()
    meta = gr.Markdown()
    with gr.Accordion("📄 အသုံးပြုထားသော ဥပဒေစာသား · sources the answer came from", open=False):
        srcs = gr.Markdown()
    gr.Examples(examples=EXAMPLES, inputs=q)
    gr.Markdown(
        "---\n"
        "Adapter: [MYOTHANTZIN/eps-burmese-sealion-9b-lora](https://huggingface.co/MYOTHANTZIN/eps-burmese-sealion-9b-lora) · "
        "Dataset: [MYOTHANTZIN/eps-burmese-qa](https://huggingface.co/datasets/MYOTHANTZIN/eps-burmese-qa) · "
        "Code & evaluation: [GitHub](https://github.com/MYOTHANTZIN-THANT/EPS-Burmese-Assistant)\n\n"
        "*Legal text is unofficial English translation with no legal force; the Korean "
        "original governs. Not legal advice.*")

    btn.click(answer, inputs=q, outputs=[out, meta, srcs])
    q.submit(answer, inputs=q, outputs=[out, meta, srcs])

demo.queue(max_size=12).launch()
