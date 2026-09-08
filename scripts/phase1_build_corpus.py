"""
Phase 1 - Corpus builder.

Fetches authoritative Korean immigration/labour sources listed in data/sources.json,
extracts clean text, chunks it with the SAME tokenizer chosen in Phase 0, and writes
JSONL with full provenance on every chunk.

Two extractors:
  statute       - law.go.kr / KLRI pages: strip chrome + known UI noise lines.
  hikorea_info  - HiKorea info pages share one large sidebar nav, so they are fetched
                  as a GROUP and any line appearing on every page is treated as
                  boilerplate. What survives is that page's unique content.

Provenance is not optional: answers must cite, and the model must be able to say
"this is an unofficial translation". license_note rides on every chunk for that reason.

Run:  .venv/Scripts/python.exe scripts/phase1_build_corpus.py
Out:  data/processed/corpus.jsonl   (gitignored - we do not commit scraped gov content)
"""
import io
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "sources.json"
OUT_DIR = ROOT / "data" / "processed"
OUT = OUT_DIR / "corpus.jsonl"

TOKENIZER = "aisingapore/gemma2-9b-cpt-sea-lionv3-instruct"  # locked in Phase 0
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
DELAY_S = 1.5  # be polite to government servers

UA = {"User-Agent": "Mozilla/5.0 (compatible; EPS-Burmese-Assistant/0.1; "
                    "non-commercial research; +https://github.com/MYOTHANTZIN-THANT/EPS-Burmese-Assistant)"}

NOISE = [
    "\uce74\uce74\uc624\ud1a1", "\ud398\uc774\uc2a4\ubd81", "\ud2b8\uc704\ud130", "\ub77c\uc778",
    "\uc8fc\uc18c\ubcf5\uc0ac", "\ud654\uba74\ub0b4\uac80\uc0c9", "\uc810\uc790\ubdf0\uc5b4",
    "\uc74c\uc131\uc9c0\uc6d0", "\uc6d0\ubb38\ub2e4\uc6b4\ub85c\ub4dc",
    "Copy URL", "Expand characters", "shrink characters", "Times New Roman",
    "Lucida Console", "Download the Statute", "Copying a law link",
]


def fetch(url):
    r = requests.get(url, headers=UA, timeout=40)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding
    return r.text


def extract_statute(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        t.decompose()
    lines = []
    for raw in soup.get_text("\n").splitlines():
        line = raw.strip()
        if not line or len(line) < 2 or any(n in line for n in NOISE):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def page_lines(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return [l.strip() for l in soup.get_text("\n").splitlines() if l.strip()]


def strip_boilerplate(pages):
    """A line present on EVERY page in the group is shared nav, not content.
    Needs >=3 pages, otherwise genuine shared content would be eaten."""
    if len(pages) < 3:
        return {k: "\n".join(v) for k, v in pages.items()}, 0
    freq = Counter()
    for lines in pages.values():
        freq.update(set(lines))
    boiler = {l for l, c in freq.items() if c == len(pages)}
    return ({k: "\n".join(l for l in v if l not in boiler) for k, v in pages.items()},
            len(boiler))


def chunk(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    for start in range(0, len(ids), step):
        window = ids[start:start + CHUNK_TOKENS]
        if len(window) < 40:
            continue
        yield tok.decode(window), len(window)
        if start + CHUNK_TOKENS >= len(ids):
            break


def main():
    srcs = json.loads(SOURCES.read_text(encoding="utf-8"))["sources"]
    print(f"loading tokenizer {TOKENIZER} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    statutes = [s for s in srcs if s.get("extractor", "statute") == "statute"]
    hik = [s for s in srcs if s.get("extractor") == "hikorea_info"]
    texts, report = {}, []

    for src in statutes:
        try:
            texts[src["id"]] = extract_statute(fetch(src["url"]))
        except Exception as e:
            report.append((src["id"], "FETCH FAIL", 0, 0, f"{type(e).__name__}: {e}"))
            print(f"[fail] {src['id']}: {e}", flush=True)
        time.sleep(DELAY_S)

    if hik:
        print(f"fetching {len(hik)} HiKorea pages as a group ...", flush=True)
        raw = {}
        for src in hik:
            try:
                raw[src["id"]] = page_lines(fetch(src["url"]))
            except Exception as e:
                report.append((src["id"], "FETCH FAIL", 0, 0, f"{type(e).__name__}: {e}"))
                print(f"[fail] {src['id']}: {e}", flush=True)
            time.sleep(DELAY_S)
        stripped, n_boiler = strip_boilerplate(raw)
        print(f"  removed {n_boiler} shared nav lines from each page", flush=True)
        texts.update(stripped)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_id = {s["id"]: s for s in srcs}
    n_chunks = 0

    with OUT.open("w", encoding="utf-8") as fh:
        for sid, text in texts.items():
            src = by_id[sid]
            floor = src.get("min_chars", 2000)
            if len(text) < floor:
                report.append((sid, "TOO SHORT", len(text), 0, f"below min_chars={floor}"))
                print(f"[warn] {sid}: only {len(text)} chars - skipped", flush=True)
                continue
            got = 0
            for i, (body, ntok) in enumerate(chunk(tok, text)):
                fh.write(json.dumps({
                    "chunk_id": f"{sid}::{i:04d}", "text": body, "token_count": ntok,
                    "source_id": sid, "title": src["title"], "source_url": src["url"],
                    "lang": src["lang"], "authority": src["authority"],
                    "doc_type": src["doc_type"], "topics": src["topics"],
                    "license_note": src["license_note"],
                    "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }, ensure_ascii=False) + "\n")
                got += 1
            n_chunks += got
            report.append((sid, "ok", len(text), got, ""))
            print(f"[ok]   {sid}: {len(text)} chars -> {got} chunks", flush=True)

    print(f"\n{'source':<30} {'status':<11} {'chars':>7} {'chunks':>7}  note")
    print("-" * 92)
    for r in sorted(report, key=lambda x: (x[1] != "ok", x[0])):
        print(f"{r[0]:<30} {r[1]:<11} {r[2]:>7} {r[3]:>7}  {r[4][:30]}")
    print(f"\nTOTAL: {n_chunks} chunks from {sum(1 for r in report if r[1]=='ok')} sources -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
