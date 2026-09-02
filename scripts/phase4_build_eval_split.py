"""
Phase 4 - build the held-out eval split from the Phase 3 dataset.

Stratified sample: pulls ~50 held-out rows keeping (a) the grounded/refusal ratio close
to the full dataset's, and (b) coverage spread across source tags so no single law
dominates the held-out set. Everything not held out becomes the training file.

Run: .venv/Scripts/python.exe scripts/phase4_build_eval_split.py
"""
import io
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "samples" / "phase3_dataset.jsonl"
HOLDOUT_OUT = ROOT / "data" / "eval" / "phase4_holdout.jsonl"
TRAIN_OUT = ROOT / "data" / "train" / "phase3_train.jsonl"

SEED = 20260902
HOLDOUT_N = 50


def tag_of(row):
    m = re.match(r"\[([a-zA-Z0-9_]+)\]", row.get("context", ""))
    return m.group(1) if m else "NONE"


def main():
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    rng = random.Random(SEED)

    by_tag = defaultdict(list)
    for r in rows:
        by_tag[tag_of(r)].append(r)
    for tag in by_tag:
        rng.shuffle(by_tag[tag])

    total = len(rows)
    n_refusal = sum(1 for r in rows if r["kind"] == "refusal")
    refusal_frac = n_refusal / total
    target_refusal = round(HOLDOUT_N * refusal_frac)
    target_grounded = HOLDOUT_N - target_refusal

    # Round-robin across tags (largest tag first drained slowest) so coverage spreads
    # instead of concentrating in the biggest source (lba_eng).
    tags_sorted = sorted(by_tag, key=lambda t: len(by_tag[t]))
    holdout = []
    picked_grounded = 0
    picked_refusal = 0
    progress = True
    while progress and (picked_grounded < target_grounded or picked_refusal < target_refusal):
        progress = False
        for tag in tags_sorted:
            bucket = by_tag[tag]
            for i, r in enumerate(bucket):
                if r["kind"] == "refusal" and picked_refusal < target_refusal:
                    holdout.append(bucket.pop(i))
                    picked_refusal += 1
                    progress = True
                    break
                if r["kind"] == "grounded" and picked_grounded < target_grounded:
                    holdout.append(bucket.pop(i))
                    picked_grounded += 1
                    progress = True
                    break
            if picked_grounded >= target_grounded and picked_refusal >= target_refusal:
                break

    holdout_ids = {r["id"] for r in holdout}
    train = [r for r in rows if r["id"] not in holdout_ids]

    HOLDOUT_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with HOLDOUT_OUT.open("w", encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with TRAIN_OUT.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"total rows: {total}  (grounded={total - n_refusal}, refusal={n_refusal})")
    print(f"holdout: {len(holdout)}  (grounded={picked_grounded}, refusal={picked_refusal})")
    print(f"train:   {len(train)}")
    print()
    print("holdout tag coverage:")
    ht = defaultdict(int)
    for r in holdout:
        ht[tag_of(r)] += 1
    for t, c in sorted(ht.items(), key=lambda kv: -kv[1]):
        print(f"  {c:3d}  {t}")
    print()
    print(f"wrote {HOLDOUT_OUT.relative_to(ROOT)}")
    print(f"wrote {TRAIN_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
