"""
Phase 3 - reusable citation checker for a new external-AI batch.

Drop new JSON/JSONL files into "q and a/" and run this BEFORE merging them into the
dataset. It checks every row's claimed [source_tag] Article N (Title) against the real
article heading in our corpus (data/processed/corpus.jsonl) and sorts rows into three
buckets so a human (or a follow-up merge pass) only has to look at the ones that actually
need a judgment call:

  OK          - claimed title matches (or is a close paraphrase of) the real one -> safe to keep as-is
  MISMATCH    - source is one we have, but claimed article number/title doesn't match
                the real heading -> needs a judgment call: relabel to the real article,
                or drop, per row (see phase3_merge_deepseek.py for the pattern to follow)
  UNKNOWN_SRC - source tag isn't in TAGMAP at all -> either add the real source to
                data/sources.json and re-run phase1_build_corpus.py first, or these rows
                can't be verified and should be treated with suspicion

This does NOT modify the dataset file. It only reports. Use phase3_merge_deepseek.py (or
copy/adapt it) to actually apply relabel/drop decisions and append to
data/samples/phase3_dataset.jsonl, same as the first batch.

Run: .venv/Scripts/python.exe scripts/phase3_verify_batch.py [path-to-folder-or-file ...]
     (defaults to everything in "q and a/" if no path given)
"""
import glob
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent

# Extend this as new sources get added to data/sources.json / the corpus.
TAGMAP = {
    "lba_eng": "labor_standards_act_eng",
    "eps_act_eng": "eps_act_eng",
    "immigration_act_eng": "immigration_act_eng",
    "minwage_act_eng": "minimum_wage_act_eng",
    "minwage_eng": "minimum_wage_act_eng",
    "eps_decree_eng": "eps_decree_eng",
    "gwra_eng": "retirement_benefits_act_eng",
    "ia_eng": "industrial_accident_act_eng",
    "insurance_premiums_act_eng": "insurance_premiums_act_eng",
    "labor_standards_decree_eng": "labor_standards_decree_eng",
    "industrial_accident_decree_eng": "industrial_accident_decree_eng",
    "wage_arrears_easylaw": "wage_arrears_easylaw",
    "health_insurance_premium_easylaw": "health_insurance_premium_easylaw",
    "retirement_benefits_act_eng": "retirement_benefits_act_eng",
    "industrial_accident_act_eng": "industrial_accident_act_eng",
}


def load_rows(paths):
    rows = []
    for p in paths:
        p = Path(p)
        with p.open(encoding="utf-8") as f:
            content = f.read()
        try:
            data = json.loads(content)
            rows.extend(data if isinstance(data, list) else [data])
        except json.JSONDecodeError:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def real_titles_for(corpus, source_id, num):
    titles = set()
    pat = re.compile(r"Article " + re.escape(num) + r" \(([A-Za-z][^)]{2,80})\)")
    for c in corpus:
        if c["source_id"] == source_id:
            titles.update(m.group(1).strip() for m in pat.finditer(c["text"]))
    return titles


def close(claimed, real_titles):
    claimed = claimed.strip().lower()
    for real in real_titles:
        real2 = real.lower()
        if claimed == real2:
            return True
        cw, rw = set(claimed.split()), set(real2.split())
        if cw and rw and len(cw & rw) / max(1, min(len(cw), len(rw))) > 0.5:
            return True
    return False


def main():
    args = sys.argv[1:]
    if args:
        paths = []
        for a in args:
            paths.extend(glob.glob(a) if any(c in a for c in "*?[") else [a])
    else:
        paths = sorted(glob.glob(str(ROOT / "q and a" / "*.json"))) + \
                sorted(glob.glob(str(ROOT / "q and a" / "*.jsonl")))

    if not paths:
        print("No input files found. Pass a path/glob, or drop files into 'q and a/'.")
        return

    corpus = [json.loads(l) for l in (ROOT / "data" / "processed" / "corpus.jsonl").open(encoding="utf-8")]
    rows = load_rows(paths)
    print(f"loaded {len(rows)} rows from {len(paths)} file(s)")

    buckets = Counter()
    mismatches, unknown = [], Counter()

    for r in rows:
        m = re.match(r"\[([a-zA-Z0-9_]+)\]", r.get("context", ""))
        tag = m.group(1) if m else None
        if tag not in TAGMAP:
            buckets["UNKNOWN_SRC"] += 1
            unknown[tag] += 1
            continue
        our_id = TAGMAP[tag]
        am = re.search(r"Article (\d+(?:-\d+)?) \(([A-Za-z][^)]{2,80})\)", r["context"])
        if not am:
            buckets["OK_NO_ARTICLE_CITED"] += 1
            continue
        cited_num, claimed_title = am.group(1), am.group(2)
        titles = real_titles_for(corpus, our_id, cited_num)
        if not titles:
            buckets["MISMATCH"] += 1
            mismatches.append((r.get("id", "?"), tag, cited_num, claimed_title, "NUMBER NOT FOUND"))
        elif close(claimed_title, titles):
            buckets["OK"] += 1
        else:
            buckets["MISMATCH"] += 1
            mismatches.append((r.get("id", "?"), tag, cited_num, claimed_title, "|".join(titles)))

    print()
    print("=== summary ===")
    for k, v in buckets.most_common():
        print(f"  {k}: {v}")

    if unknown:
        print()
        print("=== unknown source tags (add to TAGMAP + data/sources.json if worth pursuing) ===")
        for tag, n in unknown.most_common():
            print(f"  {tag}: {n} rows")

    if mismatches:
        print()
        print(f"=== {len(mismatches)} MISMATCH rows - unique (tag, article) patterns needing a judgment call ===")
        seen = set()
        for row_id, tag, num, claimed, real in mismatches:
            key = (tag, num)
            if key in seen:
                continue
            seen.add(key)
            print(f"  {tag}:{num}  claimed='{claimed}'  real='{real}'")
        print(f"\n({len(mismatches)} total rows affected by these {len(seen)} patterns)")

    print()
    print("Next step: for each MISMATCH pattern, decide relabel (same topic, fix number/title) "
          "or drop (different topic) - see scripts/phase3_merge_deepseek.py for the pattern to "
          "follow, adapting RELABEL/DROP_PATTERNS to this batch's findings.")


if __name__ == "__main__":
    main()
