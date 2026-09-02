"""
Phase 3 - build the first real batch of the training dataset.

Unlike a blind top-5-retrieval pipeline, context for each question here was
hand-verified: for every question, the actual source article was found and read in
full (grep/targeted search against data/processed/corpus.jsonl), not assumed from
automatic retrieval alone. Automatic top-5 retrieval (scripts/phase3_retrieve.py)
missed the correct article for several natural-phrasing questions even when the
article exists in the corpus - production retrieval is measured separately in
Phase 2 (~80% hit@5); dataset construction uses a higher bar; the two are not the
same process. Where the corpus (after the phase-3 gap-fill sources) still doesn't
support the question, the example is a refusal - not a guess.

Run: .venv/Scripts/python.exe scripts/phase3_build_dataset.py
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "samples" / "phase3_dataset.jsonl"

SYS = ("You are an assistant for Myanmar workers on E-9/EPS visas in Korea. Answer ONLY from "
       "the provided context. Reply in Burmese. Keep Korean official terms in Korean with a "
       "Burmese gloss. Always cite the source. If the context does not answer the question, "
       "say so and refuse — never invent a rule.")

DISC = "*မှတ်ချက် — ဤအချက်အလက်သည် ရည်ညွှန်းချက်သာဖြစ်ပြီး တရားဝင် အာဏာမရှိပါ။ လက်ရှိစည်းမျဉ်းကို 출입국관리사무소 သို့မဟုတ် 고용센터 တွင် အတည်ပြုပါ။*"
REFUSAL = "ပေးထားသော အချက်အလက်များတွင် ဤမေးခွန်းအတွက် တိကျသော အဖြေ မပါဝင်ပါ။ မှားယွင်းသော အချက်အလက် မပေးလိုပါ။ ကျေးဇူးပြု၍ 고용센터 (အလုပ်အကိုင်စင်တာ) သို့မဟုတ် 출입국관리사무소 တွင် တိုက်ရိုက် စုံစမ်းပါ။"

# Each row: (id, topic, question, context_note, context_text, answer_body, citation)
# context_text is the ACTUAL corpus excerpt read and verified (not paraphrased),
# trimmed to the relevant passage. citation names source id + title for the answer.
ROWS = [
    dict(id="p001", topic="workplace_change", kind="grounded",
        question="အလုပ်ရှင်က အကြောင်းမပြဘဲ ငါ့ကို အလုပ်ထုတ်ရင် အလုပ်ခွင် ပြောင်းလို့ရလား။",
        context="[eps_act_eng] Article 25 (Permission for Change of Business or Place of Business)\n"
                "(1) Where any of the following events occur, a foreign worker may file an application "
                "for change of business or place of business with the head of an employment security office:\n"
                "1. If his or her employer intends to terminate his or her employment contract during the "
                "contract period, or intends to refuse renewal of his or her employment contract after its "
                "expiration, on a justifiable ground;\n"
                "2. Where the foreign worker is unable to continue to work on a ground not attributable to "
                "him or her, such as temporary shutdown, closure of business, cancellation of the employment "
                "permit, or his or her employer's violation of terms and conditions of employment or unfair "
                "treatment;\n"
                "3. Where any other cause or event prescribed by Presidential Decree occurs.",
        answer="ဟုတ်ကဲ့၊ အခြေအနေပေါ်မူတည်၍ ရနိုင်ပါသည်။ 외국인고용법 ပုဒ်မ ၂၅ (Article 25) အရ "
               "အလုပ်ရှင်က စာချုပ်ကို ရပ်စဲခြင်း (သို့) စာချုပ်သစ် ငြင်းဆန်ခြင်း ဖြစ်ပါက "
               "(သို့) အလုပ်ရှင်၏ စည်းကမ်းချိုးဖောက်မှု၊ မတရားဆက်ဆံမှုကြောင့် ဆက်လက်အလုပ်မလုပ်နိုင်တော့ပါက "
               "အလုပ်ခွင် (သို့) လုပ်ငန်းနေရာ ပြောင်းလဲခွင့် လျှောက်ထားနိုင်ပါသည်။ "
               "\"အကြောင်းမပြ\" ဆိုသည့် တိကျသောအခြေအနေမှာ ဤနှစ်မျိုးအနက် မည်သည့်အမျိုးအစားဖြစ်သည်ကို "
               "고용센터 (အလုပ်အကိုင်စင်တာ) တွင် တွေ့ဆုံ စစ်ဆေးအတည်ပြုသင့်ပါသည်။",
        cite="eps_act_eng — Act on the Employment, etc. of Foreign Workers, Article 25"),

    dict(id="p002", topic="workplace_change", kind="grounded",
        question="လုပ်ငန်းရှင်က လုပ်ငန်း ပိတ်လိုက်ရင် ငါ့ဗီဇာ ဘာဖြစ်မလဲ။",
        context="[eps_act_eng] Article 25 (Permission for Change of Business or Place of Business)\n"
                "(1) 2. Where the foreign worker is unable to continue to work in the business or place of "
                "business on a ground not attributable to him or her, such as temporary shutdown, closure "
                "of business, cancellation of the employment permit under Article 19 (1), limitation on the "
                "employment under Article 20 (1), provision of a dormitory in violation of Article 22-2, or "
                "his or her employer's violation of terms and conditions of employment or unfair treatment.",
        answer="လုပ်ငန်းရှင်က လုပ်ငန်းပိတ်လိုက်ခြင်းသည် 외국인고용법 ပုဒ်မ ၂၅ (Article 25 (1) 2) တွင် "
               "အလုပ်ခွင် ပြောင်းလဲခွင့် လျှောက်ထားနိုင်သည့် အကြောင်းရင်းတစ်ခုအဖြစ် ဖော်ပြထားပါသည်။ "
               "ဗီဇာ (နေထိုင်ခွင့်) ချက်ချင်း ပျက်သွားခြင်း မဟုတ်ပါ — 고용센터 (အလုပ်အကိုင်စင်တာ) တွင် "
               "အလုပ်ခွင် ပြောင်းလဲခွင့် လျှောက်ထား၍ အလုပ်သစ် ရှာနိုင်ပါသည်။",
        cite="eps_act_eng — Act on the Employment, etc. of Foreign Workers, Article 25"),

    dict(id="p003", topic="workplace_change", kind="grounded",
        question="အလုပ်ခွင် ပြောင်းဖို့ ဘယ်နေရာမှာ လျှောက်ရမလဲ။",
        context="[hikorea_189] How to apply for permission for changing or adding workplace\n"
                "Principal or the assigned agent must submit the required documents (See the documents by "
                "the status) to the local Immigration Office.\n"
                "Application Period for Permission of changing or adding workplace: Any change or addition "
                "to a foreigner's workplace must be reviewed and granted in advance by the local Immigration.",
        answer="အလုပ်ခွင် ပြောင်းလဲ (သို့) ထပ်တိုးခွင့်ကို ဒေသဆိုင်ရာ 출입국관리사무소 (Local Immigration "
               "Office) တွင် လျှောက်ထားရပါသည်။ ကိုယ်တိုင် (သို့) တာဝန်ပေးအပ်ထားသော ကိုယ်စားလှယ်က "
               "သတ်မှတ်ထားသော စာရွက်စာတမ်းများကို တင်သွင်းနိုင်ပါသည်။ ပြောင်းလဲမှု မပြုမီ ကြိုတင်ခွင့်ပြုချက် "
               "ရယူရန် လိုအပ်ပါသည်။",
        cite="hikorea_189 — Change/Addition of Workplace (HiKorea)"),

    dict(id="p004", topic="workplace_change", kind="grounded",
        question="အလုပ်ရှင်က ရိုက်တာ၊ ဆဲတာ ခံရရင် အလုပ်ခွင် ပြောင်းလို့ရလား။",
        context="[eps_act_eng] Article 25 (1) 2. Where the foreign worker is unable to continue to work in "
                "the business or place of business on a ground not attributable to him or her, such as ... "
                "his or her employer's violation of terms and conditions of employment or unfair treatment.",
        answer="ဟုတ်ကဲ့၊ ရနိုင်ပါသည်။ 외국인고용법 ပုဒ်မ ၂၅ (Article 25 (1) 2) တွင် အလုပ်ရှင်၏ "
               "\"မတရားဆက်ဆံမှု\" (unfair treatment) ကြောင့် ဆက်လက်အလုပ်မလုပ်နိုင်တော့ပါက "
               "အလုပ်ခွင် ပြောင်းလဲခွင့် လျှောက်ထားနိုင်ကြောင်း ဖော်ပြထားပါသည်။ ရုပ်ပိုင်း/စကားဖြင့် "
               "အကြမ်းဖက်ခံရခြင်းသည် ဤအမျိုးအစားတွင် အကျုံးဝင်ပါသည်။ 고용센터 (အလုပ်အကိုင်စင်တာ) "
               "တွင် ချက်ချင်း တိုင်ကြားသင့်ပါသည် — ၎င်းသည် ပြစ်မှုလည်း ဖြစ်နိုင်၍ ရဲအဖွဲ့သို့လည်း "
               "တိုင်ကြားနိုင်ပါသည်။",
        cite="eps_act_eng — Act on the Employment, etc. of Foreign Workers, Article 25"),

    dict(id="p005", topic="insurance", kind="grounded",
        question="ထွက်ခွာချိန် အာမခံငွေ ဘယ်တော့ ထုတ်လို့ရမလဲ။",
        context="[eps_act_eng] Article 13 (Insurance and Trust for Departure Guaranty)\n"
                "(1) Any employer who runs a business or place of business hiring a foreign worker shall "
                "purchase an insurance policy or a trust deed with the foreign worker as the insured or "
                "beneficiary, in preparation for the payment of retirement benefits to the foreign worker "
                "when he or she leaves the business or place of business.",
        answer="출국만기보험 (ထွက်ခွာချိန် အာမခံ) သည် အလုပ်သမားသည် ထိုအလုပ်ခွင်မှ ထွက်ခွာချိန်တွင် "
               "အငြိမ်းစားအကျိုးခံစားခွင့်အဖြစ် ရရှိရန် ရည်ရွယ်ထားခြင်း ဖြစ်ပါသည် — 외국인고용법 ပုဒ်မ ၁၃ "
               "(Article 13) အရ အလုပ်ရှင်က ဝယ်ယူပေးရသော အာမခံဖြစ်ပါသည်။ ထုတ်ယူနိုင်သည့် တိကျသောရက်၊ "
               "လုပ်ငန်းစဉ်အသေးစိတ်ကို ဤအကြောင်းအရာတွင် မဖော်ပြထားသဖြင့် 고용센터 (အလုပ်အကိုင်စင်တာ) "
               "တွင် စစ်ဆေးသင့်ပါသည်။",
        cite="eps_act_eng — Act on the Employment, etc. of Foreign Workers, Article 13"),

    dict(id="p006", topic="insurance", kind="grounded",
        question="အလုပ်ခွင်ထိခိုက်မှု အာမခံ (산재보험) ကို ဘယ်သူက ပေးဆောင်ရမလဲ။",
        context="[insurance_premiums_act_eng] ...the relevant business owner shall bear the industrial "
                "accident insurance premium.",
        answer="산재보험 (အလုပ်ခွင်ထိခိုက်မှု လျော်ကြေးအာမခံ) ၏ ပရီမီယံကို 사업주 (အလုပ်ရှင်) က "
               "အပြည့်အဝ ပေးဆောင်ရပါသည် — အလုပ်သမားထံမှ မနုတ်ယူရပါ။ ဤသည်မှာ 건강보험 "
               "(ကျန်းမာရေးအာမခံ) ကဲ့သို့ နှစ်ဖက်ခွဲဝေမပေးရသော အာမခံအမျိုးအစား ဖြစ်ပါသည်။",
        cite="insurance_premiums_act_eng — Act on the Collection of Insurance Premiums for Employment "
             "Insurance and Industrial Accident Compensation Insurance"),

    dict(id="p007", topic="insurance", kind="refusal",
        question="ကျန်းမာရေးအာမခံ (건강보험) ကြေးကို လစဉ် ဘယ်လောက် ပေးရမလဲ။",
        context="[insurance_premiums_act_eng] Article 16-8 (Notice of Monthly Insurance Premiums)\n"
                "The Health Insurance Service shall notify business owners of the payment of monthly "
                "insurance premiums ten days before the payment deadline in writing, stating: type of the "
                "insurance premium to be collected; amount of the insurance premium to be paid; due date "
                "for and place of payment.",
        answer=REFUSAL,
        cite=None),

    dict(id="p008", topic="insurance", kind="refusal",
        question="အလုပ်အကိုင်အာမခံ (고용보험) ကြေးကို ဘယ်သူတွေ ပေးရမလဲ။",
        context="[insurance_premiums_act_eng] Article 16 (Withholding of Employment Insurance Premiums)\n"
                "A business owner may withhold the equivalent of the employment insurance premium to be "
                "paid by an employee who is an employment insurance policyholder, from his or her "
                "remuneration.",
        answer=REFUSAL,
        cite=None),

    dict(id="p009", topic="wages", kind="grounded",
        question="အပိုအလုပ်ချိန် (야근) အတွက် ငွေ ဘယ်လောက် ပိုရမလဲ။",
        context="[labor_standards_act_eng] Article 56 (Extended, Night and Holiday Work)\n"
                "An employer shall, in addition to the ordinary wages, pay employees at least 50/100 "
                "thereof for extended work (referring to the work during the hours extended pursuant to "
                "Articles 53 and 59 and to the proviso of Article 69).",
        answer="근로기준법 (Labor Standards Act) ပုဒ်မ ၅၆ (Article 56) အရ အပိုအလုပ်ချိန် (연장근로) "
               "အတွက် ပုံမှန်လုပ်ခ၏ **အနည်းဆုံး ၅၀%** ထပ်တိုး၍ ရရှိရပါမည်။ ညအလုပ် (ည ၁၀ နာရီမှ "
               "မနက် ၆ နာရီအတွင်း) နှင့် နားရက်အလုပ်များအတွက်လည်း ထပ်တိုးလုပ်ခ ရှိပါသည်။",
        cite="labor_standards_act_eng — Labor Standards Act, Article 56"),

    dict(id="p010", topic="wages", kind="refusal",
        question="လစာ နှုန်းထား သတ်မှတ်ချက်ကို ဘယ်သူ့ဆီမှာ စစ်ကြည့်လို့ရမလဲ။",
        context="[labor_standards_decree_eng] Article 26 ... where suspension due to any ground attributable "
                "to the employer, the employer shall pay him or her an allowance equivalent to at least "
                "70/100 of the difference calculated by subtracting that part of wages already paid...",
        answer=REFUSAL,
        cite=None),

    dict(id="p011", topic="wages", kind="grounded",
        question="လစာ အချိန်မှန် မရရင် ဘယ်ဌာနကို တိုင်ကြားရမလဲ။",
        context="[wage_arrears_easylaw] You may visit the competent local employment and labor office's "
                "customer service having jurisdiction over the workplace location, receive prior counseling, "
                "and then submit a petition or file a criminal complaint.",
        answer="လစာ အချိန်မှန် မရပါက ဒေသဆိုင်ရာ 고용노동부 (အလုပ်သမားဝန်ကြီးဌာန) ရုံးခွဲသို့ "
               "တိုင်ကြားနိုင်ပါသည်။ ကြိုတင် အကြံဉာဏ် တောင်းခံပြီးနောက် တိုင်ကြားလွှာ "
               "(진정) သို့မဟုတ် ရာဇဝတ်မှု တိုင်ကြားလွှာ (고소) တင်သွင်းနိုင်ပါသည်။",
        cite="wage_arrears_easylaw — Wage Arrears (Unpaid Wages), Easy-to-Find Practical Law"),

    dict(id="p012", topic="reentry", kind="grounded",
        question="၁ နှစ်ထက် ကြာကြာ ကိုရီးယားပြင်ပ သွားချင်ရင် ဘာလုပ်ရမလဲ။",
        context="[immigration_act_eng] Article 30 (Permission on Reentry)\n"
                "(1) If a foreigner who has made a foreigner registration desires to reenter the Republic "
                "of Korea after departure within his sojourn period, the Minister of Justice may permit "
                "such reentry upon his request.\n"
                "(2) The reentry permission shall be classified into the single reentry permission valid "
                "only for one time and the multiple reentry permission valid for twice or more times.",
        answer="출입국관리법 ပုဒ်မ ၃၀ (Article 30) အရ၊ ကိုရီးယားမှ ထွက်ခွာပြီး ပြန်ဝင်ချင်ပါက "
               "**ထွက်ခွာမီ** 재입국허가 (ပြန်လည်ဝင်ရောက်ခွင့်) ကို ဝန်ကြီး (Minister of Justice) ထံမှ "
               "ကြိုတင် လျှောက်ထားရပါသည်။ တစ်ကြိမ်သုံး (single) နှင့် အကြိမ်ကြိမ်သုံး (multiple) "
               "ဟူ၍ ခွဲခြားထားပါသည်။ ၁ နှစ်ထက် ကြာမည့် အစီအစဉ်အတွက် မည်သည့်အမျိုးအစား "
               "သင့်လျော်သည်ကို 출입국관리사무소 တွင် တိုက်ရိုက် တွေ့ဆုံ ဆွေးနွေးသင့်ပါသည်။",
        cite="immigration_act_eng — Immigration Act, Article 30"),

    dict(id="p013", topic="alien_registration", kind="refusal",
        question="နိုင်ငံခြားသား မှတ်ပုံတင်ကတ် ပျောက်သွားရင် ဘယ်လို လုပ်ရမလဲ။",
        context="[hikorea_282] Reporting Changes in Place of Residence — such change shall be reported via "
                "in-person visit to a Si/Gun/Gu or Eup/Myeon/Dong office, or a local immigration office, "
                "within 14 days.",
        answer=REFUSAL,
        cite=None),

    dict(id="p014", topic="stay", kind="grounded",
        question="နေထိုင်ခွင့် သက်တမ်းတိုးဖို့ ဘယ်အချိန်ကတည်းက လျှောက်ရမလဲ။",
        context="[hikorea_181] Application Period for Extension\n"
                "The foreigner shall apply before 4 months from the current expiration date to the "
                "expiration date. In case of applying for the extension after the expiration date, the "
                "individual shall pay the penalty according to Article 25 of Immigration Act.",
        answer="နေထိုင်ခွင့် သက်တမ်းကုန်ဆုံးမည့်ရက်မတိုင်မီ **၄ လအလို** မှစ၍ သက်တမ်းကုန်ဆုံးမည့်ရက်အထိ "
               "လျှောက်ထားနိုင်ပါသည်။ သက်တမ်းကုန်ပြီးမှ လျှောက်ထားပါက 출입국관리법 ပုဒ်မ ၂၅ အရ "
               "ဒဏ်ကြေး ပေးဆောင်ရနိုင်ပါသည်။",
        cite="hikorea_181 — Extension of Sojourn Period (HiKorea)"),

    dict(id="p015", topic="contract_renewal", kind="refusal",
        question="အလုပ်စာချုပ် အသစ် ရေးဖို့ အလုပ်ရှင်က ငြင်းရင် ဘာလုပ်ရမလဲ။",
        context="[insurance_premiums_act_eng] A business owner shall report to COMWEL the total remuneration "
                "paid to employees in the previous year by March 15 each year...",
        answer=REFUSAL,
        cite=None),

    dict(id="p016", topic="contract_renewal", kind="refusal",
        question="စာချုပ် သက်တမ်း ကုန်ခါနီးရင် ဘယ်နှစ်ရက်အလိုမှာ အသိပေးရမလဲ။",
        context="[insurance_premiums_act_eng] ...report the commencement or termination of the businesses "
                "to COMWEL within 14 days...",
        answer=REFUSAL,
        cite=None),

    dict(id="p017", topic="e7_transition", kind="refusal",
        question="E-9 က E-7-4 ဗီဇာ ပြောင်းချင်ရင် ဘယ်လို အရည်အချင်း လိုအပ်သလဲ။",
        context="[hikorea_189] Diagram of the procedure / Permission for Changing or Adding Workplace — "
                "general workplace-change procedure, does not cover visa-status upgrade eligibility.",
        answer=REFUSAL,
        cite=None),

    dict(id="p018", topic="e7_transition", kind="refusal",
        question="ကျွမ်းကျင်လုပ်သား ဗီဇာ ပြောင်းဖို့ ဘယ်လောက်နှစ် အလုပ်လုပ်ဖူးရမလဲ။",
        context="[eps_act_eng] Article 18-3 (Limitation on Employment after Re-Entry) — governs re-entry "
                "employment timing, does not cover E-7-4 status-change eligibility.",
        answer=REFUSAL,
        cite=None),
]


def build():
    rows = []
    for r in ROWS:
        answer = r["answer"]
        if r["kind"] == "grounded":
            answer = answer + f"\n\nရင်းမြစ် — {r['cite']}" + "\n\n" + DISC
        rows.append({"id": r["id"], "kind": r["kind"], "system": SYS,
                     "context": r["context"], "question": r["question"], "answer": answer})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_g = sum(1 for r in rows if r["kind"] == "grounded")
    n_r = len(rows) - n_g
    print(f"wrote {len(rows)} examples -> {OUT.relative_to(ROOT)}")
    print(f"  grounded: {n_g}   refusal: {n_r}  ({n_r/len(rows):.0%} refusal rate)")
    print("  topics:", sorted(set(r["topic"] for r in ROWS)))


if __name__ == "__main__":
    build()
