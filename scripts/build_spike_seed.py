"""
Build the throwaway seed dataset for the training spike.

NOT the Phase 3 dataset. Purpose is only to answer:
  1. does the stack load and take LoRA steps on a free T4?
  2. does the model emit coherent, grounded Burmese?
  3. how do E2B (~2B) and 9B compare on speed, VRAM and Burmese quality?

Answers are hand-written and grounded in the retrieved English context. Korean official
terms are kept verbatim with a Burmese gloss, every answer cites, and refusal examples
are included so the model learns to decline when context is insufficient.

Run: .venv/Scripts/python.exe scripts/build_spike_seed.py
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
CTX = json.loads((ROOT / "data" / "eval" / "spike_contexts.json").read_text(encoding="utf-8"))
OUT = ROOT / "data" / "samples" / "spike_seed.jsonl"

DISC = "*မှတ်ချက် — ဤအချက်အလက်သည် ရည်ညွှန်းချက်သာဖြစ်ပြီး တရားဝင် အာဏာမရှိပါ။ လက်ရှိစည်းမျဉ်းကို 출입국관리사무소 သို့မဟုတ် 고용센터 တွင် အတည်ပြုပါ။*"

# Hand-written Burmese answers, grounded in the retrieved context for that question id.
ANS = {
 "q02": "အလုပ်ခွင် ပြောင်းလဲခြင်း/ထပ်တိုးခြင်း ခွင့်ပြုချက် (사업장 변경·추가 허가) သည် နေထိုင်ခွင့် အဆင့်အတန်း အတွင်း၌ အလုပ်ခွင် ပြောင်းရန် ခွင့်ပြုသော အခွင့်အရေး ဖြစ်ပါသည်။ အလုပ်ခွင်ဆိုသည်မှာ သတ်မှတ်ထားသော လုပ်ငန်းနေရာ ဖြစ်ပြီး စာချုပ်နယ်ပယ်အတွင်း အလုပ်ရှင် တာဝန်ပေးနိုင်သော နေရာများလည်း အကျုံးဝင်ပါသည်။ ပြောင်းလဲနိုင်သည့် အကြိမ်အရေအတွက် ကန့်သတ်ချက်ကို ဥပဒေက သတ်မှတ်ထားပါသည်။",
 "q03": "အလုပ်ခွင် ပြောင်းလဲခွင့် လျှောက်ထားပြီးနောက် အလုပ်သစ်ရှာရန် **၃ လ** အချိန်ရပါသည်။ ထိုကာလအတွင်း အလုပ်သစ် မရရှိပါက နေထိုင်ခွင့် ပြဿနာ ဖြစ်နိုင်ပါသည်။ လျှောက်လွှာကို 고용센터 (အလုပ်အကိုင်စင်တာ) တွင် တင်ရပါမည်။",
 "q04": "အလုပ်ရှင်သည် နိုင်ငံခြားအလုပ်သမားအတွက် 출국만기보험 (ထွက်ခွာချိန် အာမခံ) သို့မဟုတ် ယုံကြည်အပ်နှံမှု စာချုပ်ကို ဝယ်ယူရန် တာဝန်ရှိပါသည်။ ၎င်းကို 외국인근로자의 고용 등에 관한 법률 ပုဒ်မ ၁၃ အရ သတ်မှတ်ထားပါသည်။ အချို့ လုပ်ငန်းအမျိုးအစားများမှာ ကင်းလွတ်ခွင့် ရှိပါသည်။",
 "q05": "လစဉ် အာမခံကြေးကို 사업주 (အလုပ်ရှင်) က ပေးဆောင်ရပါသည်။ 건강보험공단 (ကျန်းမာရေးအာမခံ ဌာန) က ပေးဆောင်ရမည့် ရက်မတိုင်မီ ၁၀ ရက်အလိုတွင် အလုပ်ရှင်ထံ အကြောင်းကြားပါသည်။",
 "q06": "အနည်းဆုံးလုပ်ခလစာကို 고용노동부장관 (အလုပ်သမားဝန်ကြီး) က **နှစ်စဉ် ဩဂုတ်လ ၅ ရက်နေ့** မတိုင်မီ သတ်မှတ်ပါသည်။ သတ်မှတ်ရာတွင် 최저임금위원회 (အနည်းဆုံးလုပ်ခ ကော်မရှင်) ၏ ဆွေးနွေးသုံးသပ်ချက်ကို တောင်းခံရပါသည်။",
 "q07": "လစာ နောက်ကျပေးမှုနှင့် ပတ်သက်၍ 고용노동부 (အလုပ်သမားဝန်ကြီးဌာန) က မှတ်တမ်းများ ထိန်းသိမ်းထားပြီး တောင်းဆိုသူထံ ပေးအပ်နိုင်ပါသည်။ လစာ မရသေးပါက 고용노동부 ၏ ဒေသဆိုင်ရာ ရုံးတွင် တိုင်ကြားနိုင်ပါသည်။",
 "q08": "အလုပ်ချိန်၊ အပိုအလုပ်ချိန်နှင့် နားရက်များအတွက် စံနှုန်းများကို 근로기준법 (အလုပ်သမား စံနှုန်းဥပဒေ) နှင့် ၎င်း၏ အကောင်အထည်ဖော်မှု အမိန့်တွင် သတ်မှတ်ထားပါသည်။",
 "q09": "အလုပ်ခွင် ထိခိုက်မှုအတွက် 산업재해보상보험 (အလုပ်ခွင်ထိခိုက်မှု လျော်ကြေးအာမခံ) မှ အကျိုးခံစားခွင့်များ ရရှိနိုင်ပါသည်။ ၎င်းတွင် အလုပ်ပြန်ဝင်ရေး ထောက်ပံ့ကြေး၊ အလုပ်နှင့် လိုက်လျောညီထွေဖြစ်စေရေး သင်တန်းစရိတ် နှင့် ပြန်လည်ထူထောင်ရေး လေ့ကျင့်ခန်း စရိတ်များ ပါဝင်ပါသည်။",
 "q11": "ကိုရီးယားသို့ ဝင်ရောက်ပြီးနောက် **၉၀ ရက်ထက် ပိုကြာ** နေထိုင်မည့် နိုင်ငံခြားသားများသည် 외국인등록 (နိုင်ငံခြားသား မှတ်ပုံတင်ခြင်း) ပြုလုပ်ရပါမည်။ သံတမန် (A-1)၊ ရုံးကိစ္စ (A-2) စသည့် အဆင့်အတန်းရှိသူများမှာ ကင်းလွတ်ခွင့် ရှိပါသည်။",
 "q12": "မှတ်ပုံတင်ထားသော နိုင်ငံခြားသားနေထိုင်သူ အများစုသည် ထွက်ခွာပြီး **၁ နှစ်အတွင်း** ပြန်ဝင်ပါက 재입국허가 (ပြန်လည်ဝင်ရောက်ခွင့်) မလိုအပ်ဘဲ ကင်းလွတ်ခွင့် ရရှိပါသည်။ သို့သော် နေထိုင်ခွင့် သက်တမ်း ကျန်ရှိရပါမည်။",
 "q13": "အလုပ်ရှင်သည် 고용허가서 (အလုပ်ခန့်ထားခွင့် လက်မှတ်) ကို ထုတ်ပေးခြင်း ဆိုင်ရာ လုပ်ထုံးလုပ်နည်းအရ လျှောက်ထားရပါသည်။ ထို့အပြင် သတ်မှတ်ထားသော အလုပ်ရှင်များသည် အာမခံ မူဝါဒနှင့် အာမခံ လက်မှတ် ဝယ်ယူရန် တာဝန်ရှိပါသည်။",
 "q16": "고용보험 (အလုပ်အကိုင် အာမခံ) ကြေးကို 근로복지공단 (COMWEL) က တာဝန်ခံ သတ်မှတ်ပြီး 건강보험공단 က လစဉ် ကောက်ခံပါသည်။ ဆုံးဖြတ်ချက်များကို 고용보험위원회 ၏ ဆွေးနွေးမှုဖြင့် ချမှတ်ပါသည်။",
 "q19": "နေထိုင်ခွင့် သက်တမ်း ကျော်လွန်ပါက 출입국관리법 အရ ပြစ်ဒဏ် သို့မဟုတ် ပြည်နှင်ဒဏ် ခံရနိုင်ပါသည်။ သက်တမ်းမကုန်မီ 체류기간 연장 (နေထိုင်ခွင့် သက်တမ်းတိုး) လျှောက်ထားသင့်ပါသည်။",
}

# Questions whose retrieved context does NOT actually answer them -> teach refusal.
REFUSE_IDS = ["q14", "q15", "q17"]   # context genuinely does not answer these
REFUSAL = "ပေးထားသော အချက်အလက်များတွင် ဤမေးခွန်းအတွက် တိကျသော အဖြေ မပါဝင်ပါ။ မှားယွင်းသော အချက်အလက် မပေးလိုပါ။ ကျေးဇူးပြု၍ 고용센터 (အလုပ်အကိုင်စင်တာ) သို့မဟုတ် 출입국관리사무소 တွင် တိုက်ရိုက် စုံစမ်းပါ။"

SYS = ("You are an assistant for Myanmar workers on E-9/EPS visas in Korea. Answer ONLY from "
       "the provided context. Reply in Burmese. Keep Korean official terms in Korean with a "
       "Burmese gloss. Always cite the source. If the context does not answer the question, "
       "say so and refuse — never invent a rule.")


def build():
    rows = []
    for qid, d in CTX.items():
        ctx = "\n\n".join(f"[{c['src']}] {c['text']}" for c in d["ctx"])
        cite = f"\n\nရင်းမြစ် — {d['ctx'][0]['title']} ({d['ctx'][0]['url']})"
        if qid in ANS:
            answer = ANS[qid] + cite + "\n\n" + DISC
            kind = "grounded"
        elif qid in REFUSE_IDS:
            answer = REFUSAL
            kind = "refusal"
        else:
            continue
        rows.append({"id": qid, "kind": kind, "system": SYS,
                     "context": ctx, "question": d["my"], "answer": answer})

    # extra refusals: real questions with deliberately irrelevant context
    extra = [
        ("r01", "အိမ်ငှားခ ဘယ်လောက် ကျသလဲ။"),
        ("r02", "ကိုရီးယား နိုင်ငံသား ဖြစ်ချင်ရင် ဘယ်လိုလုပ်ရမလဲ။"),
    ]
    filler = CTX["q06"]["ctx"][0]["text"]
    for rid, q in extra:
        rows.append({"id": rid, "kind": "refusal", "system": SYS,
                     "context": f"[minimum_wage_act_eng] {filler}",
                     "question": q, "answer": REFUSAL})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_g = sum(1 for r in rows if r["kind"] == "grounded")
    n_r = len(rows) - n_g
    print(f"wrote {len(rows)} examples -> {OUT.relative_to(ROOT)}")
    print(f"  grounded: {n_g}   refusal: {n_r}  ({n_r/len(rows):.0%} refusal rate)")


if __name__ == "__main__":
    build()
