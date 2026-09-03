"""Build phase3_dataset_v2 from the v1 dataset.

Three fixes, all motivated by measurements recorded in PHASE4_6_TRAINING_RESULTS.md:

1. Strip the fixed 131-char disclaimer suffix. It is byte-identical on 467/471
   grounded answers (33% of the average answer) and carries no information, so
   training on it spends a third of the gradient signal copying constant text.
   The app appends it in code after generation instead.

2. Fix CJK contamination. 9 rows in the `s` batch wrote Korean terms with Chinese
   ideographs (外国人 for 외국인, 信託 for 신탁). The source contexts contain zero
   CJK, so this was introduced when the answers were written.

3. Replace the single repeated refusal string. 48 of 49 training refusals were
   byte-identical, which teaches "copy this string" rather than "notice I don't
   know". Variants are owner-written (data/refusal_variants.txt) and the referral
   agency is routed by question topic.

Run with --dry-run to review the refusal agency routing without writing files.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/samples/phase3_dataset.jsonl"
VARIANTS = ROOT / "data/refusal_variants.txt"
OUT_FULL = ROOT / "data/samples/phase3_dataset_v2.jsonl"
OUT_TRAIN = ROOT / "data/train/phase3_train_v2.jsonl"
OUT_HOLDOUT = ROOT / "data/eval/phase4_holdout_v2.jsonl"
OLD_HOLDOUT = ROOT / "data/eval/phase4_holdout.jsonl"

# The disclaimer, stripped from training answers and re-added by the app at runtime.
DISCLAIMER = (
    "*မှတ်ချက် — ဤအချက်အလက်သည် ရည်ညွှန်းချက်သာဖြစ်ပြီး တရားဝင် အာဏာမရှိပါ။ "
    "လက်ရှိစည်းမျဉ်းကို 출입국관리사무소 သို့မဟုတ် 고용센터 တွင် အတည်ပြုပါ။*"
)

CJK_FIXES = {
    "外国人고용법시행령": "외국인고용법 시행령",
    "外国人고용법": "외국인고용법",
    "外国人고용허가": "외국인 고용허가",
    "外国人": "외국인",
    "信託": "신탁",
}

IMMIGRATION_OFFICE = "출입국관리사무소"
EMPLOYMENT_CENTRE = "고용센터"

# A refusal about visas / residence / re-entry must send the worker to the
# immigration office; everything else (wages, contracts, insurance, pensions)
# goes to the employment centre.
VISA_PATTERNS = [
    "ဗီဇာ",            # visa
    "နေထိုင်ခွင့်",      # residence permit
    "ပြန်လည်ဝင်ရောက်",  # re-entry
    "출입국",
    "체류",
    "E-7", "E-9", "D-2", "F-2", "F-4", "F-5",
    "re-entry", "sojourn",
    "နိုင်ငံကူးလက်မှတ်",  # passport
    "အမြဲတမ်းနေထိုင်",   # permanent residence
]


def load_variants():
    groups, current = {}, None
    for line in VARIANTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            groups[current] = []
        elif "|" in line and current:
            _, text = line.split("|", 1)
            groups[current].append(text.strip())
    return groups


# Keyword routing gets these two wrong; owner-confirmed corrections.
#   s071 asks about a lost 외국인등록증, which the immigration office issues.
#   d678 asks what documents an employer needs to hire an E-9 worker, which runs
#   through the Employment Permit System at the employment centre, not immigration.
AGENCY_OVERRIDES = {
    "s071": IMMIGRATION_OFFICE,
    "d678": EMPLOYMENT_CENTRE,
}


def route_agency(row):
    if row["id"] in AGENCY_OVERRIDES:
        return AGENCY_OVERRIDES[row["id"]]
    hay = f"{row['question']} {row['context'][:400]}"
    return IMMIGRATION_OFFICE if any(p in hay for p in VISA_PATTERNS) else EMPLOYMENT_CENTRE


def strip_disclaimer(answer):
    a = answer.rstrip()
    if DISCLAIMER in a:
        a = a.split(DISCLAIMER)[0]
    return a.rstrip()


def fix_cjk(text):
    for wrong, right in CJK_FIXES.items():
        text = text.replace(wrong, right)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the refusal routing table and exit without writing")
    args = ap.parse_args()

    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = load_variants()
    # Interleave registers so consecutive refusals do not share a style.
    ordered = []
    for i in range(max(len(v) for v in groups.values())):
        for name in ("formal", "short", "conversational"):
            if i < len(groups[name]):
                ordered.append((name, groups[name][i]))

    # v1 reused 39 ids for two entirely different rows each (e.g. d702 is both
    # "fired without notice" and "normal weekly hours"), which makes any id-keyed
    # operation unreliable. Questions ARE unique, so keep the original id for the
    # first occurrence and suffix later ones rather than renumbering everything.
    seen_ids = {}
    for r in rows:
        n = seen_ids.get(r["id"], 0)
        seen_ids[r["id"]] = n + 1
        r["orig_id"] = r["id"]
        if n:
            r["id"] = f"{r['id']}_{chr(ord('b') + n - 1)}"

    refusal_i = 0
    routing = []
    out = []
    for r in rows:
        row = dict(r)
        row["answer"] = fix_cjk(row["answer"])
        row["context"] = fix_cjk(row["context"])

        if row["kind"] == "refusal":
            agency = route_agency(row)
            register, template = ordered[refusal_i % len(ordered)]
            row["answer"] = template.replace("{AGENCY}", agency)
            routing.append((row["id"], agency, register, row["question"][:58]))
            refusal_i += 1
        else:
            row["answer"] = strip_disclaimer(row["answer"])
        out.append(row)

    print(f"rows: {len(out)}  refusals rewritten: {refusal_i}")
    print(f"variants available: {len(ordered)}")
    n_imm = sum(1 for _, a, _, _ in routing if a == IMMIGRATION_OFFICE)
    print(f"routed to {IMMIGRATION_OFFICE}: {n_imm}   to {EMPLOYMENT_CENTRE}: {len(routing)-n_imm}")
    print()
    print(f"{'id':6s} {'agency':22s} {'register':16s} question")
    print("-" * 100)
    for rid, agency, register, q in routing:
        print(f"{rid:6s} {agency:22s} {register:16s} {q}")

    if args.dry_run:
        print("\n[dry run] nothing written.")
        return

    # Preserve the original train/holdout split exactly, so v1 and v2 results are
    # comparable on the same held-out questions. Keyed on the question text, not
    # the id, because v1 ids are not unique (see the de-duplication above).
    holdout_qs = {json.loads(l)["question"].strip() for l in
                  OLD_HOLDOUT.read_text(encoding="utf-8").splitlines() if l.strip()}
    train = [r for r in out if r["question"].strip() not in holdout_qs]
    holdout = [r for r in out if r["question"].strip() in holdout_qs]
    assert len(holdout) == len(holdout_qs), (
        f"split mismatch: {len(holdout)} rows matched {len(holdout_qs)} holdout questions")
    assert len({r['id'] for r in out}) == len(out), "ids still not unique after dedup"

    for path, data in ((OUT_FULL, out), (OUT_TRAIN, train), (OUT_HOLDOUT, holdout)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(data):4d} -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
