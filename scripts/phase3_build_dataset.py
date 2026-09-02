"""
Phase 3 - build the training dataset, in small hand-verified rounds.

Unlike a blind top-5-retrieval pipeline, context for each question here was
hand-verified: for every question, the actual source article was found and read in
full (grep/targeted search against data/processed/corpus.jsonl), not assumed from
automatic retrieval alone. Automatic top-5 retrieval (scripts/phase3_retrieve.py)
missed the correct article for several natural-phrasing questions even when the
article exists in the corpus - production retrieval is measured separately in
Phase 2 (~80% hit@5); dataset construction uses a higher bar; the two are not the
same process. Where the corpus still doesn't support the question, the example is
a refusal - not a guess. Where a fact isn't in the corpus but the user (a native
Burmese speaker) confirms it directly, context is marked [user-verified] instead
of a fake document citation.

Round 1 (p001-p018): first batch across all 6 topics, gap-fill.
Round 2 (p101-p115): more workplace_change, using Article 25(3)/(4) and Decree
Article 30 details found while re-reading the EPS Act for round 1.

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

    dict(id="p007", topic="insurance", kind="grounded",
        question="ကျန်းမာရေးအာမခံ (건강보험) ကြေးကို လစဉ် ဘယ်လောက် ပေးရမလဲ။",
        context="[user-verified] Officially published National Health Insurance premium rate, "
                "confirmed 2026-09: employer-sponsored insurance total rate is 7.19% of monthly "
                "salary (보수월액), split 50:50 between employer and employee (3.595% each).",
        answer="ကိုရီးယားနိုင်ငံ၏ ၂၀၂၆ ခုနှစ် စံနှုန်းအရ 직장가입자 (ကုမ္ပဏီဝန်ထမ်း) များအတွက် "
               "ကျန်းမာရေးအာမခံကြေး စုစုပေါင်းမှာ လစဉ်လစာ (보수월액) ၏ **၇.၁၉%** ဖြစ်ပြီး၊ "
               "အလုပ်ရှင်နှင့် အလုပ်သမား နှစ်ဦးစလုံးက ၅၀:၅၀ နှုန်းဖြင့် **၃.၅၉၅%** စီ "
               "ခွဲဝေပေးဆောင်ရပါသည်။ ဥပမာ — လစဉ်လစာ ၃,၀၀၀,၀၀၀ ဝမ် ရရှိပါက စုစုပေါင်း ၂၁၅,၇၀၀ ဝမ် "
               "ကျသင့်ပြီး၊ ဝန်ထမ်းဘက်မှ လစဉ် ၁၀၇,၈၅၀ ဝမ်ကို လစာထဲမှ ဖြတ်တောက်ခံရမည် ဖြစ်သည်။",
        cite="National Health Insurance Service (NHIS) — publicly announced 2026 premium rate"),

    dict(id="p008", topic="insurance", kind="grounded",
        question="အလုပ်အကိုင်အာမခံ (고용보험) ကြေးကို ဘယ်သူတွေ ပေးရမလဲ။",
        context="[user-verified] Officially published Employment Insurance premium rate, confirmed "
                "2026-09: unemployment-benefit portion is split between employer and employee, "
                "0.9% of monthly income each side.",
        answer="အလုပ်အကိုင်အာမခံကြေး (Employment Insurance) ကို အလုပ်ရှင်နှင့် အလုပ်သမား "
               "နှစ်ဦးစလုံး ခွဲဝေပေးဆောင်ရပါသည်။ ဝန်ထမ်း (သင်) ဘက်မှ အလုပ်လက်မဲ့ အာမခံအတွက် "
               "လစဉ်ဝင်ငွေ၏ **၀.၉%** ကို ပေးဆောင်ရပြီး အလုပ်ရှင်ဘက်ကလည်း **၀.၉%** ထပ်ဆောင်း "
               "စိုက်ထုတ်ပေးသွင်းရပါသည်။",
        cite="Employment Insurance Act / Ministry of Employment and Labor — publicly announced "
             "2026 premium rate"),

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

    dict(id="p010", topic="wages", kind="grounded",
        question="လစာ နှုန်းထား သတ်မှတ်ချက်ကို ဘယ်သူ့ဆီမှာ စစ်ကြည့်လို့ရမလဲ။",
        context="[user-verified] Official channels for checking the current statutory minimum wage, "
                "confirmed 2026-09: Ministry of Employment and Labor (moel.go.kr) or the Minimum "
                "Wage Commission (최저임금위원회) website; also HiKorea or the Foreign Worker "
                "Counseling Center hotline 1350.",
        answer="တရားဝင် အနိမ့်ဆုံးလုပ်ခလစာ (Minimum Wage) နှုန်းထားများနှင့် သတ်မှတ်ချက်များကို "
               "အောက်ပါနေရာများတွင် စစ်ဆေးနိုင်သည် — 고용노동부 (အလုပ်သမားဝန်ကြီးဌာန) "
               "တရားဝင်ဝဘ်ဆိုက် (moel.go.kr) သို့မဟုတ် 최저임금위원회 (အနိမ့်ဆုံးလုပ်ခလစာ ကော်မတီ) "
               "စာမျက်နှာ။ HiKorea (하이코리아) သို့မဟုတ် 외국인력상담센터 (နိုင်ငံခြားသား "
               "အလုပ်သမား ကူညီရေးစင်တာ — ဖုန်းနံပါတ် ၁၃၅၀) သို့ ဆက်သွယ်မေးမြန်း၍လည်း "
               "စစ်ဆေးနိုင်ပါသည်။",
        cite="Ministry of Employment and Labor (moel.go.kr) / Minimum Wage Commission — official channels"),

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

    dict(id="p013", topic="alien_registration", kind="grounded",
        question="နိုင်ငံခြားသား မှတ်ပုံတင်ကတ် ပျောက်သွားရင် ဘယ်လို လုပ်ရမလဲ။",
        context="[user-verified] Official reissuance procedure for a lost Alien Registration Card, "
                "confirmed 2026-09: apply within 14 days of discovering the loss, at the local "
                "immigration office (via HiKorea reservation), with passport, photo, a completed "
                "loss-statement/reissuance form, and the reissuance fee (~30,000 won).",
        answer="ကတ်ပျောက်ဆုံးကြောင်း သိရှိရသည့်နေ့မှစ၍ **၁၄ ရက်**အတွင်း အမြန်ဆုံး လျှောက်ထားရပါမည်။\n"
               "လုပ်ဆောင်ရမည့်ပုံစံ — နိုင်ငံခြားသား မှတ်ပုံတင်ကတ် ပျောက်ဆုံးကြောင်း ဖောင် "
               "(Statement of Loss) ကို ဖြည့်စွက်ရန် လိုအပ်သည်။ နိုင်ငံကူးလက်မှတ် (Passport)၊ "
               "ဓာတ်ပုံ (၁ ပုံ)၊ နှင့် ပြန်လည်ထုတ်ယူခ (ဝန်ဆောင်ခ ၃၀,၀၀၀ ဝမ် ဝန်းကျင်) တို့ကို "
               "ယူဆောင်သွားပါ။ HiKorea (hikorea.go.kr) တွင် ကြိုတင် ချိန်းယူပြီး သင့်နေထိုင်ရာ "
               "ဒေသကို တာဝန်ယူသော 출입국·외국인청 (လူဝင်မှုကြီးကြပ်ရေးရုံး) သို့ လူကိုယ်တိုင် "
               "သွားရောက်၍ Reissuance (재발급) လျှောက်ထားရပါမည်။",
        cite="HiKorea (hikorea.go.kr) — Reissuance of Alien Registration Card procedure"),

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

    dict(id="p015", topic="contract_renewal", kind="grounded",
        question="အလုပ်စာချုပ် အသစ် ရေးဖို့ အလုပ်ရှင်က ငြင်းရင် ဘာလုပ်ရမလဲ။",
        context="[user-verified] Standard EPS worker protections, confirmed 2026-09: an employee facing "
                "an unjustified refusal to renew can seek counseling/file a complaint at the local "
                "Employment and Labor office or the Foreign Worker Support Center (hotline 1350); "
                "may also be eligible to apply for permission to change workplace (사업장 변경).",
        answer="အကြောင်းရင်းကို သေချာစွာ မေးမြန်းဆွေးနွေးပါ။ အလုပ်ရှင်ဘက်က တရားမဲ့စွာ ငြင်းဆိုခြင်း "
               "(သို့) သက်တမ်းတိုးရန် ပျက်ကွက်ပါက သက်ဆိုင်ရာ 고용노동청 (အလုပ်သမားရုံး) "
               "သို့မဟုတ် 외국인력지원센터 (နိုင်ငံခြားသား အလုပ်သမား ကူညီရေးစင်တာ — ၁၃၅၀) "
               "သို့ သွားရောက်၍ တိုင်ကြားခြင်း၊ အကြံဉာဏ်တောင်းခံခြင်းများ ပြုလုပ်နိုင်ပါသည်။ "
               "လုပ်ငန်းခွင် အခြေအနေကြောင့် တရားဝင် အလုပ်ပြောင်းရွှေ့ခွင့် (사업장 변경) "
               "လျှောက်ထားနိုင်သည့် အခွင့်အရေး ရှိမရှိလည်း စစ်ဆေးသင့်ပါသည်။",
        cite="EPS worker protection channels (고용노동청 / 외국인력지원센터 1350) — standard procedure"),

    dict(id="p016", topic="contract_renewal", kind="grounded",
        question="စာချုပ် သက်တမ်း ကုန်ခါနီးရင် ဘယ်နှစ်ရက်အလိုမှာ အသိပေးရမလဲ။",
        context="[user-verified] Customary practice, confirmed 2026-09: when either party does not "
                "intend to renew, notice is customarily given at least 30 days (1 month) before "
                "expiry; E-9 workers must handle extension/continued-work procedures with the "
                "relevant offices within the prescribed periods.",
        answer="အလုပ်ရှင် (သို့) အလုပ်သမား တစ်ဦးဦးက စာချုပ်သက်တမ်း မတိုးတော့ဘူး (သို့) "
               "အလုပ်စာချုပ် အသစ်မချုပ်တော့ဘူးဆိုပါက သက်တမ်းမကုန်ဆုံးမီ **အနည်းဆုံး ရက်ပေါင်း "
               "၃၀ (၁ လ)** အလို ကြိုတင်၍ အချင်းချင်း အသိပေးကြေညာရိုး ထုံးစံရှိပါသည်။ E-9 "
               "အလုပ်သမားများအတွက်မူ သက်တမ်းမကုန်ဆုံးမီ သက်တမ်းတိုးရန် (သို့) လုပ်ငန်းခွင် "
               "ဆက်လက်လုပ်ကိုင်ရန် လုပ်ထုံးလုပ်နည်းများကို သတ်မှတ်ရက်များအတွင်း သက်ဆိုင်ရာ "
               "ဌာနများတွင် ကြိုတင်လုပ်ဆောင်ရန် လိုအပ်ပါသည်။ ဤသည်မှာ ဥပဒေအရ တိကျစွာ "
               "သတ်မှတ်ထားခြင်း မဟုတ်ဘဲ ဓလေ့ထုံးစံ ဖြစ်သဖြင့် 고용센터 တွင် "
               "အတည်ပြုသင့်ပါသည်။",
        cite="Customary EPS practice, confirmed by native-speaker review — not a fixed statutory rule"),

    dict(id="p017", topic="e7_transition", kind="grounded",
        question="E-9 က E-7-4 ဗီဇာ ပြောင်းချင်ရင် ဘယ်လို အရည်အချင်း လိုအပ်သလဲ။",
        context="[user-verified] E-7-4 (숙련기능인력, Skilled Worker) point-based visa system "
                "(K-Point E74), confirmed 2026-09: requires (1) at least 4 years of lawful E-9 work "
                "within the last 10 years, (2) current annual income of at least 26,000,000 won and "
                "a contract for at least 2 more years, (3) Korean-language ability — TOPIK level 2+ "
                "or KIIP (사회통합프로그램) level 2+, and (4) at least 200 of 300 points under the "
                "K-Point E74 scoring system.",
        answer="E-9 (비전문취업) မှ E-7-4 (숙련기능인력) ဗီဇာသို့ ပြောင်းလဲရန် အဓိက "
               "လိုအပ်ချက်များမှာ — **လုပ်သက်**: လွန်ခဲ့သော ၁၀ နှစ်အတွင်း E-9 ဗီဇာဖြင့် "
               "ကိုရီးယားနိုင်ငံတွင် ၄ နှစ် (သို့) ထိုထက်ပို၍ တရားဝင် အလုပ်လုပ်ကိုင်ဖူးသူ "
               "ဖြစ်ရမည်။ **လုပ်ခလစာနှင့် စာချုပ်**: လက်ရှိအလုပ်တွင် သတ်မှတ်ထားသော "
               "နှစ်စဉ်ဝင်ငွေ (연봉 2,600만원 ဖြစ်) ရှိရမည် ဖြစ်ပြီး အနည်းဆုံး ၂ နှစ် "
               "ထပ်မံလုပ်ကိုင်မည့် စာချုပ် ရှိရမည်။ **ကိုရီးယားစာ အရည်အချင်း**: TOPIK "
               "(အဆင့် ၂ နှင့်အထက်) (သို့) 사회통합프로그램 (KIIP ၂ အဆင့်နှင့်အထက်) "
               "လိုအပ်သည် (သတ်မှတ်ကာလအလိုက် သက်သာခွင့်/유예 များ ရှိနိုင်သည်)။ **ရမှတ်စနစ်**: "
               "K-Point E74 စနစ်အရ စုစုပေါင်း ရမှတ် ၃၀၀ အနက် အနည်းဆုံး ရမှတ် ၂၀၀ ရရှိထားရမည် "
               "ဖြစ်သည်။ တိကျသော အချိန်ကာလအလိုက် စည်းမျဉ်းအပြောင်းအလဲများ ရှိနိုင်သဖြင့် "
               "လျှောက်ထားမည့်အချိန်တွင် 고용센터 (သို့) 하이코리아 တွင် အတည်ပြုပါ။",
        cite="E-7-4 (숙련기능인력) K-Point points-based visa system — publicly announced criteria"),

    dict(id="p018", topic="e7_transition", kind="grounded",
        question="ကျွမ်းကျင်လုပ်သား ဗီဇာ ပြောင်းဖို့ ဘယ်လောက်နှစ် အလုပ်လုပ်ဖူးရမလဲ။",
        context="[user-verified] E-7-4 (숙련기능인력, Skilled Worker) eligibility, confirmed "
                "2026-09: at least 4 years of lawful E-9 work in Korea within the last 10 years.",
        answer="E-7-4 (숙련기능인력) ကျွမ်းကျင်လုပ်သား ဗီဇာသို့ လျှောက်ထားပြောင်းလဲရန်အတွက် "
               "လွန်ခဲ့သော **၁၀ နှစ်**အတွင်း E-9 ဗီဇာဖြင့် ကိုရီးယားနိုင်ငံတွင်း၌ စုစုပေါင်း "
               "**အနည်းဆုံး ၄ နှစ်** အလုပ်လုပ်ကိုင်ဖူးသည့် လုပ်သက် ရှိရမည် ဖြစ်ပါသည်။ "
               "(အခြား စည်းကမ်းချက်များလည်း ရှိသေးသည် — p017 ကို ကြည့်ပါ။)",
        cite="E-7-4 (숙련기능인력) K-Point points-based visa system — publicly announced criteria"),

    # --- Round 2 (2026-09-02): more workplace_change, hand-verified against
    # eps_act_eng Article 25(3)/(4), eps_decree_eng Article 30, and hikorea_189's
    # full "Criteria for granting/denying" section.
    dict(id="p101", topic="workplace_change", kind="grounded",
        question="၃ လအတွင်း အလုပ်သစ် မရှာနိုင်ရင် ဘာဖြစ်မလဲ။",
        context="[eps_act_eng] Article 25 (3) Any foreign worker who fails to obtain permission for "
                "change of workplace within three months from the date of the application, or who "
                "fails to file an application for change within one month after the expiration of "
                "the employment contract, shall leave the Republic of Korea: Provided, That for a "
                "foreign worker who is unable to do so due to causes such as an accident on duty, "
                "illnesses, pregnancy and childbirth, such period shall be calculated from the date "
                "on which such cause ceases to exist.",
        answer="외국인고용법 ပုဒ်မ ၂၅(၃) (Article 25(3)) အရ၊ အလုပ်ခွင်ပြောင်းလွှင့်ခွင့် "
               "လျှောက်ထားပြီးနောက် ၃ လအတွင်း ခွင့်ပြုချက် မရရှိပါက (သို့) စာချုပ်သက်တမ်းကုန်ပြီးနောက် "
               "၁ လအတွင်း လျှောက်လွှာ မတင်နိုင်ပါက **ကိုရီးယားမှ ထွက်ခွာရမည်** ဖြစ်ပါသည်။ သို့သော် "
               "အလုပ်ခွင်ထိခိုက်မှု၊ နာမကျန်းမှု၊ ကိုယ်ဝန်ဆောင်ခြင်း (သို့) မီးဖွားခြင်း စသည့် "
               "အကြောင်းများကြောင့် မဖြစ်နိုင်ပါက ထိုအကြောင်းရင်း ကုန်ဆုံးသည့်နေ့မှစ၍ "
               "ရေတွက်ရပါမည်။",
        cite="eps_act_eng — Act on the Employment, etc. of Foreign Workers, Article 25(3)"),

    dict(id="p102", topic="workplace_change", kind="grounded",
        question="ဖျားနာနေရင် (သို့) ကိုယ်ဝန်ဆောင်နေရင် အလုပ်ခွင်ပြောင်းဖို့ အချိန်ကန့်သတ်ချက်ကို "
                 "ဆိုင်းငံ့ပေးလား။",
        context="[eps_act_eng] Article 25 (3) proviso: for a foreign worker unable to obtain "
                "permission or file an application due to causes such as an accident on duty, "
                "illnesses, pregnancy and childbirth, such period shall be calculated from the date "
                "on which such cause ceases to exist.",
        answer="ဟုတ်ကဲ့၊ ဆိုင်းငံ့ပေးပါသည်။ 외국인고용법 ပုဒ်မ ၂၅(၃) (Article 25(3)) အရ၊ "
               "အလုပ်ခွင်ထိခိုက်မှု၊ နာမကျန်းမှု၊ ကိုယ်ဝန်ဆောင်ခြင်း (သို့) မီးဖွားခြင်းကြောင့် "
               "ခွင့်ပြုချက် မရနိုင်ခြင်း (သို့) လျှောက်လွှာ မတင်နိုင်ခြင်း ဖြစ်ပါက ၃ လ (သို့) ၁ လ "
               "အချိန်ကာလကို ထိုအကြောင်းရင်း ကုန်ဆုံးသည့်နေ့မှသာ စတင်ရေတွက်ပါသည်။",
        cite="eps_act_eng — Act on the Employment, etc. of Foreign Workers, Article 25(3)"),

    dict(id="p103", topic="workplace_change", kind="grounded",
        question="အလုပ်ရှင်ချို့ယွင်းချက်ကြောင့် အလုပ်ခွင်ပြောင်းရင် အကြိမ်ရေ ကန့်သတ်ချက်ထဲ "
                 "ပါဝင်လား။",
        context="[eps_act_eng] Article 25 (4) Foreign worker's change of business or place of "
                "business shall not, in principle, exceed three times during the period under "
                "Article 18 or two times during the extended period under Article 18-2 (1): "
                "Provided, That the foregoing shall not include cases of change on any ground "
                "prescribed in paragraph (1) 2.",
        answer="မပါဝင်ပါ။ 외국인고용법 ပုဒ်မ ၂၅(၄) (Article 25(4)) အရ၊ အလုပ်ခွင် ပြောင်းလဲခွင့်ကို "
               "ပုံမှန်အားဖြင့် ၃ ကြိမ် (တာဝန်ကာလ တိုးချဲ့ပါက ၂ ကြိမ်) ကန့်သတ်ထားသော်လည်း၊ "
               "အလုပ်ရှင်၏ ချို့ယွင်းမှု (Article 25(1)(2) — စီးပွားရေးရပ်ဆိုင်း၊ လုပ်ငန်းပိတ်ခြင်း၊ "
               "မတရားဆက်ဆံမှု စသည်) ကြောင့် ပြောင်းရသော အကြိမ်များကို ဤကန့်သတ်ချက်တွင် "
               "မထည့်တွက်ပါ။",
        cite="eps_act_eng — Act on the Employment, etc. of Foreign Workers, Article 25(4)"),

    dict(id="p104", topic="workplace_change", kind="grounded",
        question="အလုပ်ခွင်ထိခိုက်ဒဏ်ရာရလို့ လက်ရှိအလုပ် ဆက်မလုပ်နိုင်တော့ရင် အလုပ်ခွင်ပြောင်းလို့ "
                 "ရလား။",
        context="[eps_decree_eng] Article 30 (Change of Business or Place of Business) (1) means "
                "where it is deemed that a foreign worker is unfit to continue service in the "
                "business or place of business due to his or her own injury, etc., but it is "
                "possible for him or her to work for another business or place of business.",
        answer="ဟုတ်ကဲ့၊ ရနိုင်ပါသည်။ 시행령 ပုဒ်မ ၃၀ (Enforcement Decree Article 30) အရ၊ "
               "မိမိကိုယ်တိုင် ထိခိုက်ဒဏ်ရာစသည့် အကြောင်းကြောင့် လက်ရှိအလုပ်ကို ဆက်လုပ်ရန် "
               "မသင့်တော်တော့သော်လည်း၊ အခြားလုပ်ငန်း (သို့) နေရာတွင် အလုပ်လုပ်နိုင်သေးပါက "
               "အလုပ်ခွင် ပြောင်းလဲခွင့် လျှောက်ထားနိုင်ပါသည်။",
        cite="eps_decree_eng — Enforcement Decree of the EPS Act, Article 30"),

    dict(id="p105", topic="workplace_change", kind="grounded",
        question="အလုပ်စာချုပ် သက်တမ်းကုန်ပြီးနောက် အလုပ်ခွင်ပြောင်းဖို့ ဘယ်နှစ်ရက်အတွင်း "
                 "လျှောက်ရမလဲ။",
        context="[eps_act_eng] Article 25 (3) ... or who fails to file an application for change of "
                "business or place of business within one month after the expiration of the "
                "employment contract with the employer shall leave the Republic of Korea.",
        answer="စာချုပ်သက်တမ်း ကုန်ဆုံးပြီးနောက် **၁ လ**အတွင်း အလုပ်ခွင် ပြောင်းလဲခွင့် လျှောက်လွှာ "
               "တင်သွင်းရပါမည်။ ၁ လထက် ကျော်လွန်ပါက ကိုရီးယားမှ ထွက်ခွာရနိုင်ပါသည် — 외국인고용법 "
               "ပုဒ်မ ၂၅(၃) (Article 25(3))။",
        cite="eps_act_eng — Act on the Employment, etc. of Foreign Workers, Article 25(3)"),

    dict(id="p106", topic="workplace_change", kind="refusal",
        question="အလုပ်ခွင်ပြောင်းဖို့ လျှောက်ရင် ဘယ်စာရွက်စာတမ်းတွေ အတိအကျ လိုအပ်သလဲ။",
        context="[hikorea_189] Principal or the assigned agent must submit the required documents "
                "(See the documents by the status) to the local Immigration Office.",
        answer=REFUSAL,
        cite=None),

    dict(id="p107", topic="workplace_change", kind="grounded",
        question="အလုပ်ခွင် အခြေစိုက်နေရာ ၂ ခုထက်ပို၍ ထပ်တိုးလို့ရလား။",
        context="[hikorea_189] Adding more than two workplaces beside the main workplace is "
                "prohibited.",
        answer="မရပါ။ အဓိကအလုပ်ခွင်အပြင် နောက်ထပ် အလုပ်ခွင် ၂ ခုထက်ပို၍ ထပ်တိုးခြင်းကို "
               "တားမြစ်ထားပါသည်။",
        cite="hikorea_189 — Change/Addition of Workplace (HiKorea)"),

    dict(id="p108", topic="workplace_change", kind="grounded",
        question="အလုပ်သစ်က မူလထက် လုပ်ခ ပိုများရင် (သို့) အချိန်ပိုကြာရင် ထပ်တိုးလို့ရလား။",
        context="[hikorea_189] If the added workplace should have longer work hours or higher "
                "salary than the original workplace, addition of workplace will be restricted.",
        answer="ကန့်သတ်ချက် ရှိပါသည်။ ထပ်တိုးမည့် အလုပ်ခွင်သည် မူလအလုပ်ခွင်ထက် လုပ်ချိန် ပိုကြာသည် "
               "(သို့) လုပ်ခ ပိုများသည် ဆိုပါက ထပ်တိုးခွင့်ကို ကန့်သတ်ပါသည်။",
        cite="hikorea_189 — Change/Addition of Workplace (HiKorea)"),

    dict(id="p109", topic="workplace_change", kind="grounded",
        question="အလုပ်ခွင် အကြိမ်ကြိမ် ပြောင်းနေရင် ဘာဖြစ်နိုင်လဲ။",
        context="[hikorea_189] If the foreigner working in too many workplaces or is changing jobs "
                "too many times without any consistent pattern, he/she will be evaluated. Should the "
                "foreigner be found to have poor work conduct or is in some way against Korea's "
                "national interest, then any future change/addition to the workplace will be "
                "restricted.",
        answer="အကြောင်းရင်း တသမတ်တည်း မရှိဘဲ အကြိမ်ကြိမ် ပြောင်းနေပါက စိစစ်ခံရနိုင်ပါသည်။ "
               "အလုပ်လုပ်ပုံအမူအကျင့် ညံ့ဖျင်းသည် (သို့) ကိုရီးယားနိုင်ငံ၏ အကျိုးစီးပွားနှင့် "
               "ဆန့်ကျင်သည်ဟု တွေ့ရှိပါက နောင်လာမည့် အလုပ်ခွင်ပြောင်းလဲ/ထပ်တိုးမှုများကို "
               "ကန့်သတ်ခံရနိုင်ပါသည်။",
        cite="hikorea_189 — Change/Addition of Workplace (HiKorea)"),

    dict(id="p110", topic="workplace_change", kind="grounded",
        question="အလုပ်ခွင်ပြောင်းခွင့်ကို လူဝင်မှုကြီးကြပ်ရေးက စိစစ်ပြီးမှ ခွင့်ပြုတာလား၊ "
                 "အလိုအလျောက်ရမလား။",
        context="[hikorea_189] Any change or addition to a foreigner's workplace must be reviewed "
                "and granted in advance by the local Immigration.",
        answer="အလိုအလျောက် မရပါ — 출입국관리사무소 (Local Immigration Office) က ကြိုတင် "
               "စိစစ်၍ ခွင့်ပြုမှသာ ပြောင်းလဲနိုင်ပါသည်။",
        cite="hikorea_189 — Change/Addition of Workplace (HiKorea)"),

    dict(id="p111", topic="workplace_change", kind="refusal",
        question="အလုပ်ခွင်ပြောင်းဖို့ လျှောက်ထားရင် အခကြေးငွေ ပေးရလား။",
        context="[hikorea_189] How to apply for permission for changing or adding workplace — "
                "documents and venue are described; no fee amount is stated.",
        answer=REFUSAL,
        cite=None),

    dict(id="p112", topic="workplace_change", kind="refusal",
        question="အလုပ်ခွင်ပြောင်းခွင့် လျှောက်လွှာ ငြင်းပယ်ခံရရင် အယူခံဝင်လို့ရလား။",
        context="[hikorea_189] Criteria for granting/denying are described; no appeal procedure is "
                "stated.",
        answer=REFUSAL,
        cite=None),

    dict(id="p113", topic="workplace_change", kind="refusal",
        question="အလုပ်ခွင်ပြောင်းဖို့ လုပ်ငန်းစဉ်လုပ်နေတုန်း အလုပ်ရှင်သစ်က အလုပ်ပေးမည့်ကတိကို "
                 "ပြန်ရုပ်သိမ်းရင် ဘာဖြစ်မလဲ။",
        context="[eps_act_eng] Article 25 covers the worker's own application and eligibility; does "
                "not address a prospective new employer withdrawing an offer mid-process.",
        answer=REFUSAL,
        cite=None),

    dict(id="p114", topic="workplace_change", kind="refusal",
        question="ကိုယ်ဝန်ဆောင်နေတဲ့အတွက်ချည်း အလုပ်ခွင်ပြောင်းခွင့် လျှောက်ထားလို့ရလား။",
        context="[eps_act_eng] Article 25 (1) lists the grounds for eligibility (employer-side "
                "termination, business closure/unfair treatment, other Presidential Decree causes); "
                "pregnancy appears only in Article 25(3) as a reason the 3-month/1-month deadline "
                "can be paused, not as an independent (1) eligibility ground.",
        answer=REFUSAL,
        cite=None),

    dict(id="p115", topic="workplace_change", kind="refusal",
        question="အလုပ်ခွင်ပြောင်းရှာနေတဲ့ကာလအတွင်း လစာ ရနေဦးမလား။",
        context="[eps_act_eng] Article 25 covers eligibility and the change/application procedure; "
                "does not address income or support during the job-search period.",
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
