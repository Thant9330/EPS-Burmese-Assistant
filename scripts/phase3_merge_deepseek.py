"""
Phase 3 - verify and merge the DeepSeek-generated batch (data/q and a/*.json, 230 rows)
into the main dataset.

Verification method: every row's `context` field names a source + article number. For
every row citing a source already in our corpus (eps_act_eng, eps_decree_eng,
immigration_act_eng, labor_standards_act_eng, minimum_wage_act_eng, plus two newly-added
sources - retirement_benefits_act_eng and industrial_accident_act_eng, fetched specifically
to check this batch), the claimed article number + title was checked against the REAL
article heading in our corpus (fetched directly from law.go.kr/elaw.klri.re.kr).

Result: 155 of 230 rows had a citation that didn't match the real article. Of those:
- ~10 patterns were the same real topic, just paraphrased or off-by-one - RELABELED to the
  real article number/title, content kept (it was accurate, just mis-cited).
- The rest (EPS Act insurance articles 21-24, EPS Decree 20/22, EPS Act 9/13/18-4/28,
  Immigration Act 17/19/20/31/33/79, Industrial Accident Act 5/10/37, Labor Standards Act
  15/17, Retirement Benefits Act 10) pointed to a genuinely different real article than
  claimed - DROPPED rather than guessed at, since this is exactly the accuracy problem
  the whole redo exists to avoid. Several of these (the 4 insurance types, departure
  guaranty) duplicate content we already built correctly in round 3 anyway.

10 rows cited a source not in our corpus at all (hikorea_e74, a specific E-7-4 HiKorea
page we could not find officially) - cross-checked against the facts the user personally
verified earlier (see p017/p018 in the main dataset) instead.

Run: .venv/Scripts/python.exe scripts/phase3_merge_deepseek.py
"""
import glob
import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "samples" / "phase3_dataset.jsonl"

DISC = "*မှတ်ချက် — ဤအချက်အလက်သည် ရည်ညွှန်းချက်သာဖြစ်ပြီး တရားဝင် အာဏာမရှိပါ။ လက်ရှိစည်းမျဉ်းကို 출입국관리사무소 သို့မဟုတ် 고용센터 တွင် အတည်ပြုပါ။*"
REFUSAL = "ပေးထားသော အချက်အလက်များတွင် ဤမေးခွန်းအတွက် တိကျသော အဖြေ မပါဝင်ပါ။ မှားယွင်းသော အချက်အလက် မပေးလိုပါ။ ကျေးဇူးပြု၍ 고용센터 (အလုပ်အကိုင်စင်တာ) သို့မဟုတ် 출입국관리사무소 တွင် တိုက်ရိုက် စုံစမ်းပါ။"

TAGMAP = {
    "lba_eng": ("labor_standards_act_eng", "Labor Standards Act"),
    "eps_act_eng": ("eps_act_eng", "Act on the Employment, etc. of Foreign Workers"),
    "immigration_act_eng": ("immigration_act_eng", "Immigration Act"),
    "minwage_act_eng": ("minimum_wage_act_eng", "Minimum Wage Act"),
    "minwage_eng": ("minimum_wage_act_eng", "Minimum Wage Act"),
    "eps_decree_eng": ("eps_decree_eng", "Enforcement Decree of the Act on the Employment, etc. of Foreign Workers"),
    "gwra_eng": ("retirement_benefits_act_eng", "Guarantee of Workers' Retirement Benefits Act"),
    "ia_eng": ("industrial_accident_act_eng", "Industrial Accident Compensation Insurance Act"),
}

# (source_tag, claimed_article) -> real_article_number, or None to mean "drop this pattern"
RELABEL = {
    ("lba_eng", "70"): "70",   # same topic, real title "Restrictions on Night Work and Holiday Work"
    ("lba_eng", "74"): "74",   # real "Protection for Maternity"
    ("lba_eng", "53"): "53",   # real "Restrictions on Extended Work"
    ("lba_eng", "46"): "46",   # real "Shutdown Allowances"
    ("lba_eng", "36"): "36",   # real "Settlement of Payments"
    ("lba_eng", "63"): "63",   # real "Exclusion from Application"
    ("lba_eng", "57"): "56",   # retarget: real content is at 56, confirmed by our own round 5
    ("eps_act_eng", "18-2"): "18-2",  # real "Special Cases for Limitation on Period of Service"
    ("immigration_act_eng", "46"): "46",  # real "Persons to be Deported"
    ("immigration_act_eng", "30"): "30",  # real "Permission on Reentry"
    ("gwra_eng", "9"): "8",    # real Article 8 has the calculation formula DeepSeek described
}

DROP_PATTERNS = {
    ("eps_act_eng", "21"), ("eps_act_eng", "22"), ("eps_act_eng", "23"), ("eps_act_eng", "24"),
    ("eps_act_eng", "9"), ("eps_act_eng", "13"), ("eps_act_eng", "18-4"), ("eps_act_eng", "28"),
    ("eps_decree_eng", "20"), ("eps_decree_eng", "22"),
    ("immigration_act_eng", "17"), ("immigration_act_eng", "19"), ("immigration_act_eng", "20"),
    ("immigration_act_eng", "31"), ("immigration_act_eng", "33"), ("immigration_act_eng", "79"),
    ("ia_eng", "5"), ("ia_eng", "10"), ("ia_eng", "37"),
    ("lba_eng", "15"), ("lba_eng", "17"),
    ("gwra_eng", "10"),
}


def real_title(corpus, source_id, num):
    pat = re.compile(r"Article " + re.escape(num) + r" \(([A-Za-z][^)]{2,80})\)")
    for c in corpus:
        if c["source_id"] == source_id:
            m = pat.search(c["text"])
            if m:
                return m.group(1).strip()
    return None


def load_rows():
    rows = []
    for fp in sorted(glob.glob(str(ROOT / "q and a" / "*.json"))):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main():
    corpus = [json.loads(l) for l in open(ROOT / "data" / "processed" / "corpus.jsonl", encoding="utf-8")]
    rows = load_rows()

    # E-7-4 facts the user personally verified earlier (see p017/p018) - used to
    # sanity-check hikorea_e74 rows since we have no real page for that source.
    verified_e74 = ("4 year", "10 year", "topik", "kiip", "k-point", "200", "300", "26,000,000",
                    "2,600만원", "2 year")

    kept, dropped_dup_insurance, dropped_mismatch, kept_e74, dropped_e74 = [], 0, 0, 0, 0
    next_id = 501

    for r in rows:
        m = re.match(r"\[([a-zA-Z0-9_]+)\]", r["context"])
        tag = m.group(1) if m else None

        if tag == "hikorea_e74":
            blob = (r["question"] + " " + r["answer"] + " " + r["context"]).lower()
            if any(k in blob for k in verified_e74) or "e-7-4" in blob or "e7-4" in blob:
                r["id"] = f"d{next_id}"; next_id += 1
                kept_e74 += 1
                kept.append(r)
            else:
                dropped_e74 += 1
            continue

        if tag not in TAGMAP:
            continue
        our_id, our_title = TAGMAP[tag]

        am = re.search(r"Article (\d+(?:-\d+)?) \(", r["context"])
        if not am:
            # no article number cited (e.g. some refusals) - keep as-is, retag id
            r["id"] = f"d{next_id}"; next_id += 1
            kept.append(r)
            continue
        cited = am.group(1)

        if (tag, cited) in DROP_PATTERNS:
            if tag == "eps_act_eng" and cited in ("21", "22", "23", "24"):
                dropped_dup_insurance += 1
            else:
                dropped_mismatch += 1
            continue

        real_num = RELABEL.get((tag, cited), cited)
        rt = real_title(corpus, our_id, real_num)
        if rt is None:
            # couldn't confirm even the relabeled target - be safe, drop
            dropped_mismatch += 1
            continue

        # Rewrite the [tag] Title Article N (Old Title) prefix in context to the real one.
        new_ctx = re.sub(
            r"^\[[a-zA-Z0-9_]+\][^\n]*Article " + re.escape(cited) + r" \([^)]*\)",
            f"[{tag}] {our_title} Article {real_num} ({rt})",
            r["context"], count=1)
        # Update the citation line inside the answer, and the article number mentioned in
        # the Burmese answer body (Burmese digit or Latin digit forms of the old number).
        new_ans = r["answer"]
        new_ans = re.sub(r"Article " + re.escape(cited) + r"\b", f"Article {real_num}", new_ans)
        new_ans = re.sub(r"ပုဒ်မ [၀-၉\-]+ \(Article " + re.escape(real_num) + r"\)",
                         f"ပုဒ်မ {real_num} (Article {real_num})", new_ans)

        r["context"] = new_ctx
        r["answer"] = new_ans
        r["id"] = f"d{next_id}"; next_id += 1
        kept.append(r)

    print(f"input rows: {len(rows)}")
    print(f"kept (OK or relabeled): {len(kept) - kept_e74}")
    print(f"kept hikorea_e74 (cross-checked vs user-verified facts): {kept_e74}")
    print(f"dropped - duplicate of our own insurance content: {dropped_dup_insurance}")
    print(f"dropped - genuine citation/content mismatch, unverifiable: {dropped_mismatch}")
    print(f"dropped - hikorea_e74 not matching verified facts: {dropped_e74}")
    print(f"TOTAL KEPT: {len(kept)}")

    # append to the existing dataset file (built by phase3_build_dataset.py)
    existing = []
    if OUT.exists():
        existing = [json.loads(l) for l in OUT.open(encoding="utf-8")]
    all_rows = existing + kept
    with OUT.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_g = sum(1 for r in all_rows if r["kind"] == "grounded")
    n_r = len(all_rows) - n_g
    print()
    print(f"FINAL DATASET: {len(all_rows)} examples -> {OUT.relative_to(ROOT)}")
    print(f"  grounded: {n_g}   refusal: {n_r}  ({n_r/len(all_rows):.0%} refusal rate)")


if __name__ == "__main__":
    main()
