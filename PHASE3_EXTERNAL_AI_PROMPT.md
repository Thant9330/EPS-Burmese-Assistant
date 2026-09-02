# Prompt for building more Phase 3 dataset examples with another AI

Copy everything in the box below into the other AI. Give it real source text to work from
(paste in text from the official pages listed, or let it search/fetch them if it can browse) —
without real source text it will guess, which is exactly what we're avoiding.

**How many:** the project's overall target is 800-1500 examples; we have 260 so far. Don't
force the other AI toward an exact number — real source material runs out before fake volume
does, and that's fine (that's the same wall we hit ourselves). A realistic ask for one pass:
**150-300 more**, in batches of 30-50. If it starts repeating itself or thinning out on real
source text, stop that pass rather than padding with guesses — bring what's genuinely
grounded, and we do another pass later if we still need more.

**Important — learned from the last batch:** checking the last 300-row batch against the
real law text found 155 rows citing an article number that didn't say what the row claimed
(e.g. it invented a clean "Articles 21-24 = the four insurance types" mapping that doesn't
exist — the real ones are scattered at 13, 15, 23). The Burmese language quality was fine;
the specific article numbers were the problem. **Tell the other AI explicitly: if you are
not highly confident of the exact article number, either look it up for real (fetch/search)
or leave it out of the citation and describe the source more generally** (e.g. "the EPS Act's
provision on X" without a specific number) rather than guessing a plausible-sounding one.
A wrong specific number is worse than no number — I can always add the real one once verified.

**Also avoid exact duplicate questions** if you're running multiple separate sessions/batches
— the same question showed up 3 times in the last batch (once even with a wrong answer,
contradicting the other two). If unsure whether a question was already asked, favor a
different phrasing.

When you bring the results back, paste the raw JSONL and I'll check it against the format,
merge it in, and flag anything that looks unsupported by its own context before it goes in the file.

---

## COPY FROM HERE

You are building training data for a Burmese-language legal assistant chatbot, for Myanmar
migrant workers on E-9/EPS work visas in South Korea. Every example must be built from REAL
Korean government source text — never from general knowledge, never guessed.

### Output format

One JSON object per line (JSONL), UTF-8, exactly these 5 fields:

```json
{"id": "q001", "kind": "grounded", "system": "You are an assistant for Myanmar workers on E-9/EPS visas in Korea. Answer ONLY from the provided context. Reply in Burmese. Keep Korean official terms in Korean with a Burmese gloss. Always cite the source. If the context does not answer the question, say so and refuse — never invent a rule.", "context": "...", "question": "...", "answer": "..."}
```

- `id` — any short unique string per row (e.g. `q001`, `q002`, ...)
- `kind` — `"grounded"` if the context actually answers the question, `"refusal"` if it doesn't
- `system` — always exactly the sentence shown above, unchanged, every row
- `context` — the real excerpt(s) you're grounding the answer in. Format: `[source_id] English excerpt text`. Use the ACTUAL text from the source (verbatim or lightly trimmed), not a paraphrase of what you think it says.
- `question` — natural Burmese, the way a worker would actually ask it (mix short/blunt and longer/polite phrasings)
- `answer` — Burmese, following the rules below

### The answer rules (this is the part that matters most)

1. **Only use facts that are literally in the `context` field.** If you're not sure the source says it, don't include it.
2. **Korean official terms**: Korean script first, Burmese gloss in parentheses right after. Example: `출국만기보험 (ထွက်ခွာချိန် အာမခံ)`, `고용센터 (အလုပ်အကိုင်စင်တာ)`.
3. **Every grounded answer ends with these two lines, in this order:**
   ```
   \n\nရင်းမြစ် — <source_id> — <source title>, <article/section if applicable>\n\n*မှတ်ချက် — ဤအချက်အလက်သည် ရည်ညွှန်းချက်သာဖြစ်ပြီး တရားဝင် အာဏာမရှိပါ။ လက်ရှိစည်းမျဉ်းကို 출입국관리사무소 သို့မဟုတ် 고용센터 တွင် အတည်ပြုပါ။*
   ```
4. **If the context doesn't actually answer the question** — use `kind: "refusal"` and set `answer` to exactly this (no citation/disclaimer lines needed on refusals):
   ```
   ပေးထားသော အချက်အလက်များတွင် ဤမေးခွန်းအတွက် တိကျသော အဖြေ မပါဝင်ပါ။ မှားယွင်းသော အချက်အလက် မပေးလိုပါ။ ကျေးဇူးပြု၍ 고용센터 (အလုပ်အကိုင်စင်တာ) သို့မဟုတ် 출입국관리사무소 တွင် တိုက်ရိုက် စုံစမ်းပါ။
   ```
5. **Never mix languages carelessly** — the answer body is Burmese; Korean terms/article numbers are the only Korean/English allowed inline.
6. **About 15-20% of rows should end up as genuine refusals** — pick some questions where you deliberately give context that doesn't cover it (a related-but-different provision, or a document that's silent on this specific point), to teach the model to say "I don't know" instead of inventing an answer. Don't force answers where the source is genuinely silent — that's the whole point.
7. **Only cite a specific article number you're actually confident about.** If you can look it up (browsing/search), do that and quote the real text. If you can't verify it, either omit the specific number (describe the source more generally) or mark the row as `refusal` instead of guessing — a wrong citation is worse than an admitted gap.

### Real examples (already in our dataset — match this pattern exactly)

**Grounded:**
```json
{"id": "p001", "kind": "grounded", "system": "You are an assistant for Myanmar workers on E-9/EPS visas in Korea. Answer ONLY from the provided context. Reply in Burmese. Keep Korean official terms in Korean with a Burmese gloss. Always cite the source. If the context does not answer the question, say so and refuse — never invent a rule.", "context": "[eps_act_eng] Article 25 (Permission for Change of Business or Place of Business)\n(1) Where any of the following events occur, a foreign worker may file an application for change of business or place of business with the head of an employment security office:\n1. If his or her employer intends to terminate his or her employment contract during the contract period, or intends to refuse renewal of his or her employment contract after its expiration, on a justifiable ground;\n2. Where the foreign worker is unable to continue to work on a ground not attributable to him or her, such as temporary shutdown, closure of business, cancellation of the employment permit, or his or her employer's violation of terms and conditions of employment or unfair treatment;\n3. Where any other cause or event prescribed by Presidential Decree occurs.", "question": "အလုပ်ရှင်က အကြောင်းမပြဘဲ ငါ့ကို အလုပ်ထုတ်ရင် အလုပ်ခွင် ပြောင်းလို့ရလား။", "answer": "ဟုတ်ကဲ့၊ အခြေအနေပေါ်မူတည်၍ ရနိုင်ပါသည်။ 외국인고용법 ပုဒ်မ ၂၅ (Article 25) အရ အလုပ်ရှင်က စာချုပ်ကို ရပ်စဲခြင်း (သို့) စာချုပ်သစ် ငြင်းဆန်ခြင်း ဖြစ်ပါက (သို့) အလုပ်ရှင်၏ စည်းကမ်းချိုးဖောက်မှု၊ မတရားဆက်ဆံမှုကြောင့် ဆက်လက်အလုပ်မလုပ်နိုင်တော့ပါက အလုပ်ခွင် (သို့) လုပ်ငန်းနေရာ ပြောင်းလဲခွင့် လျှောက်ထားနိုင်ပါသည်။ \"အကြောင်းမပြ\" ဆိုသည့် တိကျသောအခြေအနေမှာ ဤနှစ်မျိုးအနက် မည်သည့်အမျိုးအစားဖြစ်သည်ကို 고용센터 (အလုပ်အကိုင်စင်တာ) တွင် တွေ့ဆုံ စစ်ဆေးအတည်ပြုသင့်ပါသည်။\n\nရင်းမြစ် — eps_act_eng — Act on the Employment, etc. of Foreign Workers, Article 25\n\n*မှတ်ချက် — ဤအချက်အလက်သည် ရည်ညွှန်းချက်သာဖြစ်ပြီး တရားဝင် အာဏာမရှိပါ။ လက်ရှိစည်းမျဉ်းကို 출입국관리사무소 သို့မဟုတ် 고용센터 တွင် အတည်ပြုပါ။*"}
```

**Refusal:**
```json
{"id": "p106", "kind": "refusal", "system": "You are an assistant for Myanmar workers on E-9/EPS visas in Korea. Answer ONLY from the provided context. Reply in Burmese. Keep Korean official terms in Korean with a Burmese gloss. Always cite the source. If the context does not answer the question, say so and refuse — never invent a rule.", "context": "[hikorea_189] Principal or the assigned agent must submit the required documents (See the documents by the status) to the local Immigration Office.", "question": "အလုပ်ခွင်ပြောင်းဖို့ လျှောက်ရင် ဘယ်စာရွက်စာတမ်းတွေ အတိအကျ လိုအပ်သလဲ။", "answer": "ပေးထားသော အချက်အလက်များတွင် ဤမေးခွန်းအတွက် တိကျသော အဖြေ မပါဝင်ပါ။ မှားယွင်းသော အချက်အလက် မပေးလိုပါ။ ကျေးဇူးပြု၍ 고용센터 (အလုပ်အကိုင်စင်တာ) သို့မဟုတ် 출입국관리사무소 တွင် တိုက်ရိုက် စုံစမ်းပါ။"}
```

### Sources to use (real, official, in English)

Pull real excerpts from these — search/fetch them if you can browse, or ask me to paste in text
for the ones you can't reach:

- **Act on the Employment, etc. of Foreign Workers** (the "EPS Act") — https://www.law.go.kr/LSW/engLsInfoR.do?lsiSeq=231477
- **Immigration Act** — law.go.kr English (search "Immigration Act")
- **Labor Standards Act** — https://www.law.go.kr/LSW/engLsInfoR.do?lsiSeq=232199
- **Guarantee of Workers' Retirement Benefits Act** — https://www.law.go.kr/LSW/engLsInfoR.do?lsiSeq=86562
- **Industrial Accident Compensation Insurance Act** — https://www.law.go.kr/LSW/engLsInfoR.do?lsiSeq=198265
- **Minimum Wage Act** — elaw.klri.re.kr
- **Enforcement Decree of the Labor Standards Act** — elaw.klri.re.kr
- **Enforcement Decree of the Act on the Employment of Foreign Workers** — elaw.klri.re.kr
- **Act on the Collection of Insurance Premiums for Employment Insurance and Industrial Accident Compensation Insurance** — elaw.klri.re.kr
- **HiKorea** (hikorea.go.kr) English info pages — procedural guidance (workplace change, visa extension, re-entry permits, reporting obligations, deportation, etc.)
- **easylaw.go.kr** (Ministry of Government Legislation's plain-language guides) — good for practical how-to questions (wage complaints, insurance rates)

When a source above has a direct URL, use that exact page — it's already verified to exist and
be in English. For the elaw.klri.re.kr / HiKorea ones without a URL listed, search for them;
if you find one, quote the real text rather than the number/title you remember.

### Topics — what's already well-covered vs. what's still needed

**Already have good coverage (260 examples built so far) — don't just repeat these facts, but
different phrasings of the same facts are fine and useful:**
workplace change eligibility/procedure/limits, the 4 insurance types (departure guaranty — EPS
Act Article 13; wage-delay guaranty and personal injury — Article 23; return-home expense —
Article 15) and their penalties, annual leave (Labor Standards Act Article 60), unfair dismissal
protection (Articles 23/28), overtime pay rate (Article 56, 50%), dismissal advance notice
(Article 26, 30 days), alien registration reporting, stay categories, status change, re-entry
permit rules, deportation basics, severance pay (Guarantee of Workers' Retirement Benefits Act
Articles 8-9), industrial accident insurance benefit categories and COMWEL as the administering
body (Industrial Accident Compensation Insurance Act Articles 10, 36), E-9 → E-7-4 basics
(4-year work history, TOPIK/KIIP, K-Point scoring).

**Still thin or missing — prioritize these:**
- **Contract renewal** — what happens at contract expiry, renewal process, notice periods (still almost nothing found here — if you can find real sources, that's high value)
- **Exact minimum wage figures** (current year's won/hour rate — we only have the mechanism, not the number)
- **Lost/stolen Alien Registration Card reissuance** — couldn't find an official hikorea.go.kr page for this specifically
- **E-9 → E-7-4 official source page** — we have the facts (confirmed by a native-speaker reviewer) but never found the actual official page to cite; if you can find one, that upgrades those rows from "reviewer-verified" to properly cited
- More **industrial accident insurance** claims filing process (how a worker actually files, not just who administers it)
- More **workplace safety** topics if you can find an Occupational Safety and Health Act source

### A few ground rules

- If you genuinely can't find real source text for a topic, don't fabricate it — either skip it or make it a refusal example with an honest note in the context about what's missing.
- Batches of 30-50 rows are easier for me to check than one giant dump — but bring however much you have.
- Keep questions realistic — how a worker with limited Korean and limited legal literacy would actually phrase things, not textbook legal English translated into Burmese.

## COPY TO HERE
