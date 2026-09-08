"""
Recompute the headline results in README.md from the committed evaluation data.

The point of this script is that nobody has to take the README's numbers on trust. It reads
data/eval/full50_base_vs_tuned.json - the raw generations, base and fine-tuned, on the same
50 held-out questions with identical prompts and decoding - and re-derives every figure in
the results table.

Metric definitions, stated plainly because they are judgement calls:

  korean   of held-out rows whose GOLD answer contains Korean text, how many model outputs
           share at least one Korean term with it. Measures whether official terms survive.
  cites    of grounded rows, how many outputs carry a source marker - either a bracketed
           tag like (hikorea_189) or the trained "ရင်းမြစ် —" source line.
  refuses  of refusal rows, how many outputs decline rather than answer. Burmese negation
           is မ + verb; this is a crude proxy and undercounts politely-worded refusals that
           carry no negation, so read it as a floor.
  article  of rows where BOTH gold and output cite an Article number, how many agree. Note
           the denominators differ between models: the fine-tune cites far more often, so
           it is judged on more rows.
  stopped  how many generations emitted an end-of-turn token rather than hitting the
           300-token cap.

Run:  python scripts/verify_results.py
"""
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL50 = ROOT / "data/eval/full50_base_vs_tuned.json"

HANGUL = re.compile(r"[가-힣]+")
SRC_TAG = re.compile(r"[\(\[]\s*(?:[a-z][a-z0-9_]*(?:_eng)?|hikorea_\d+)[^\)\]]*[\)\]]")
SRC_LINE = re.compile(r"ရင်းမြစ်\s*—")
DECLINE = re.compile(r"မ(ပါ|ပေး|ဖော်ပြ|ပြော|သိ|ရှိ|တွေ့|ဆို|ပါဝင်|ပါရှိ)")
ARTICLE = re.compile(r"Article\s*(\d+)")


def cites(text):
    return bool(SRC_TAG.search(text) or SRC_LINE.search(text))


def score(rows):
    grounded = [r for r in rows if r["kind"] == "grounded"]
    refusal = [r for r in rows if r["kind"] == "refusal"]
    korean = [r for r in grounded if HANGUL.findall(r["answer"])]
    with_article = [r for r in grounded if ARTICLE.findall(r["answer"])]
    cited = [r for r in with_article if ARTICLE.findall(r["out"])]
    agreed = [r for r in cited
              if set(ARTICLE.findall(r["answer"])) & set(ARTICLE.findall(r["out"]))]
    kept = [r for r in korean
            if set(HANGUL.findall(r["answer"])) & set(HANGUL.findall(r["out"]))]
    return {
        "keeps Korean terms": f"{len(kept)}/{len(korean)}",
        "cites a source": f"{sum(cites(r['out']) for r in grounded)}/{len(grounded)}",
        "refuses correctly": f"{sum(bool(DECLINE.search(r['out'])) for r in refusal)}/{len(refusal)}",
        "article correct": f"{len(agreed)}/{len(cited)}",
        "stopped naturally": f"{sum(r['stopped'] for r in rows)}/{len(rows)}",
        "mean output chars": sum(len(r["out"]) for r in rows) // len(rows),
    }


def main():
    data = json.loads(FULL50.read_text(encoding="utf-8"))
    base, tuned = data["base"], data["tuned"]
    assert len(base) == len(tuned) == 50, "expected 50 rows per condition"

    # Guard against the failure that bit this project twice: an evaluation that silently
    # compares a model against itself. Identical outputs are a bug signal, not a result.
    identical = sum(1 for a, b in zip(base, tuned) if a["out"].strip() == b["out"].strip())
    assert identical < 45, (
        f"{identical}/50 outputs identical - the two conditions are the same model")

    b, t = score(base), score(tuned)
    width = max(len(k) for k in b)
    print(f"{'metric':{width}}  {'BASE':>12}  {'FINE-TUNED':>12}")
    print("-" * (width + 28))
    for k in b:
        print(f"{k:{width}}  {str(b[k]):>12}  {str(t[k]):>12}")
    print(f"\nidentical outputs: {identical}/50 (must be low - see assertion above)")
    print(f"source: {FULL50.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
