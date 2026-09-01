"""
Phase 1 - Corpus builder.

Fetches authoritative Korean immigration/labour sources listed in data/sources.json,
extracts clean text, chunks it with the SAME tokenizer chosen in Phase 0, and writes
JSONL with full provenance on every chunk.

Provenance is not optional here: answers must cite, and the model must be able to say
"this is an unofficial translation". license_note rides on every chunk for that reason.

Run:  .venv/Scripts/python.exe scripts/phase1_build_corpus.py
Out:  data/processed/corpus.jsonl   (gitignored - we do not commit scraped gov content)
"""
import io
import json
import re
import sys
import time
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
                    "non-commercial research; +https://github.com/Thant9330/EPS-Burmese-Assistant)"}

# Navigation/UI noise that law.go.kr wraps around the statute body.
NOISE = [
    "카카오톡", "페이스북", "트위터", "라인", "주소복사", "조례위임조문", "위임조례",
    "화면내검색", "조문 선택", "조문선택", "점자뷰어", "음성지원", "원문다운로드",
    "Copy URL", "Expand characters", "shrink characters", "Times New Roman",
    "Lucida Console", "Download the Statute", "Copying a law link",
]


def fetch(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=40)
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding
    return r.text


def extract(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        t.decompose()
    lines = []
    for raw in soup.get_text("\n").splitlines():
        line = raw.strip()
        if not line or len(line) < 2:
            continue
        if any(n in line for n in NOISE):
            continue
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text)


def chunk(tok, text: str):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    for start in range(0, len(ids), step):
        window = ids[start:start + CHUNK_TOKENS]
        if len(window) < 40:      # drop trailing scraps
            continue
        yield tok.decode(window), len(window)
        if start + CHUNK_TOKENS >= len(ids):
            break


def main():
    manifest = json.loads(SOURCES.read_text(encoding="utf-8"))
    print(f"loading tokenizer {TOKENIZER} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    report = []

    with OUT.open("w", encoding="utf-8") as fh:
        for src in manifest["sources"]:
            try:
                html = fetch(src["url"])
                text = extract(html)
            except Exception as e:
                report.append((src["id"], "FETCH FAIL", 0, 0, f"{type(e).__name__}: {e}"))
                print(f"[fail] {src['id']}: {e}", flush=True)
                continue

            if len(text) < 2000:
                report.append((src["id"], "TOO SHORT", len(text), 0,
                               "likely JS-rendered shell, not real content"))
                print(f"[warn] {src['id']}: only {len(text)} chars - skipped", flush=True)
                time.sleep(DELAY_S)
                continue

            got = 0
            for i, (body, ntok) in enumerate(chunk(tok, text)):
                rec = {
                    "chunk_id": f"{src['id']}::{i:04d}",
                    "text": body,
                    "token_count": ntok,
                    "source_id": src["id"],
                    "title": src["title"],
                    "source_url": src["url"],
                    "lang": src["lang"],
                    "authority": src["authority"],
                    "doc_type": src["doc_type"],
                    "topics": src["topics"],
                    "license_note": src["license_note"],
                    "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                got += 1
            n_chunks += got
            report.append((src["id"], "ok", len(text), got, ""))
            print(f"[ok]   {src['id']}: {len(text)} chars -> {got} chunks", flush=True)
            time.sleep(DELAY_S)

    print(f"\n{'source':<22} {'status':<11} {'chars':>7} {'chunks':>7}  note")
    print("-" * 84)
    for r in report:
        print(f"{r[0]:<22} {r[1]:<11} {r[2]:>7} {r[3]:>7}  {r[4][:34]}")
    print(f"\nTOTAL: {n_chunks} chunks -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
