"""
Phase 2 - Retrieval smoke test.

The architecture assumes: corpus + retrieval in English, generation in Burmese.
That assumption rests on one open question - can BGE-M3 match a BURMESE query against
ENGLISH chunks directly, or must we translate the query to English first?

This measures it instead of guessing. Two conditions, same corpus, same encoder:
  A) embed the Burmese question directly
  B) embed a human English translation (the upper bound any MT pipeline could reach)

Relevance labels are WEAK: a chunk is gold if it contains any of the question's
gold_terms. That is a proxy for human judgement - read the numbers as a signal about
the LANGUAGE GAP (A vs B on identical labels), not as absolute retrieval quality.

Run:  .venv/Scripts/python.exe scripts/phase2_retrieval_smoketest.py
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"
QUESTIONS = ROOT / "data" / "eval" / "phase2_questions.json"
OUT = ROOT / "data" / "eval" / "phase2_results.json"
MODEL = "BAAI/bge-m3"
K = 5


def gold_mask(chunks, terms):
    """Weak label: chunk is relevant if it contains any gold phrase (case-insensitive)."""
    low = [c["text"].lower() for c in chunks]
    return np.array([any(t.lower() in txt for t in terms) for txt in low])


def main():
    chunks = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    print(f"corpus: {len(chunks)} chunks | questions: {len(qs)}")

    print(f"loading {MODEL} (first run downloads ~2.3GB) ...", flush=True)
    model = SentenceTransformer(MODEL)

    print("embedding corpus ...", flush=True)
    C = model.encode([c["text"] for c in chunks], batch_size=8,
                     normalize_embeddings=True, show_progress_bar=False)

    print("embedding queries ...", flush=True)
    Qmy = model.encode([q["my"] for q in qs], normalize_embeddings=True)
    Qen = model.encode([q["en"] for q in qs], normalize_embeddings=True)

    results, agg = [], {"my": {"hit": 0, "mrr": 0.0, "prec": 0.0},
                        "en": {"hit": 0, "mrr": 0.0, "prec": 0.0}}
    skipped = 0

    for i, q in enumerate(qs):
        gold = gold_mask(chunks, q["gold_terms"])
        n_gold = int(gold.sum())
        if n_gold == 0:
            skipped += 1
            results.append({**{k: q[k] for k in ("id", "topic")},
                            "n_gold": 0, "note": "no chunk matches gold_terms - unlabelable"})
            continue

        row = {"id": q["id"], "topic": q["topic"], "n_gold": n_gold}
        for lang, Q in (("my", Qmy), ("en", Qen)):
            sims = C @ Q[i]
            top = np.argsort(-sims)[:K]
            hits = gold[top]
            hit = bool(hits.any())
            rank = int(np.argmax(hits)) + 1 if hit else 0
            prec = float(hits.sum()) / K
            row[lang] = {"hit@5": hit, "first_rank": rank, "prec@5": round(prec, 2),
                         "top_sources": [chunks[j]["source_id"] for j in top[:3]]}
            agg[lang]["hit"] += int(hit)
            agg[lang]["mrr"] += (1.0 / rank) if rank else 0.0
            agg[lang]["prec"] += prec
        results.append(row)

    n = len(qs) - skipped
    print(f"\nlabelable questions: {n}/{len(qs)}  (skipped {skipped} with no gold match)\n")
    print(f"{'id':<5} {'topic':<20} {'gold':>4} | {'MY hit':>6} {'rank':>4} | {'EN hit':>6} {'rank':>4}")
    print("-" * 74)
    for r in results:
        if r["n_gold"] == 0:
            print(f"{r['id']:<5} {r['topic']:<20} {0:>4} | {'--':>6} {'--':>4} | {'--':>6} {'--':>4}  (unlabelable)")
            continue
        m, e = r["my"], r["en"]
        print(f"{r['id']:<5} {r['topic']:<20} {r['n_gold']:>4} | "
              f"{'YES' if m['hit@5'] else 'no':>6} {m['first_rank'] or '-':>4} | "
              f"{'YES' if e['hit@5'] else 'no':>6} {e['first_rank'] or '-':>4}")

    print(f"\n{'condition':<34} {'hit@5':>7} {'MRR':>7} {'prec@5':>7}")
    print("-" * 60)
    for lang, label in (("my", "A) Burmese query -> EN corpus"),
                        ("en", "B) English query -> EN corpus")):
        a = agg[lang]
        print(f"{label:<34} {a['hit']/n:>6.1%} {a['mrr']/n:>7.3f} {a['prec']/n:>7.3f}")

    gap = (agg["en"]["hit"] - agg["my"]["hit"]) / n
    print(f"\nlanguage gap (B - A) on hit@5: {gap:+.1%}")
    OUT.write_text(json.dumps({"model": MODEL, "k": K, "n_labelable": n,
                               "aggregate": {k: {kk: (vv / n) for kk, vv in v.items()}
                                             for k, v in agg.items()},
                               "per_question": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"saved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
