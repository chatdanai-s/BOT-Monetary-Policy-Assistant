"""
Hallucination test for rag_pipeline.py (BOT MPR assistant).

Run from the project root or from tests/:
    python tests/rag_test.py
    python tests/rag_test.py --qa tests/hallucination_questions.json --judge-model gemini-3.5-flash-lite
    python tests/rag_test.py --limit 10 --no-judge
    python tests/rag_test.py --k-retrieve 30 --top-k 8

Every question in the QA file is designed so that the MPR knowledge base CANNOT answer it
(unrelated topic, non-existent data or event, misused jargon, false premise, word salad).
The correct behaviour is therefore always to decline, or to say that the reports do not
contain the information / that the premise or wording does not match the reports.
Any substantive answer that is not supported by the retrieved excerpts is a hallucination.

The pipeline is NOT modified. rag_pipeline.query_documents() is called as is:
  - the answer is captured from its stdout stream
  - rag_pipeline.format_docs is wrapped to record the chunks the LLM actually saw

How each answer is classified (two independent layers):
  1. Heuristics (no API needed):
       - exact refusal sentence at the start, short, and no invented numbers
       - absence phrases ("not mentioned", "no such", "does not exist", ...)
       - ungrounded numbers: figures in the answer that appear neither in the retrieved
         chunks nor in the question
       - fabricated citations: [Source: ...] tags that point to chunks that were not retrieved
  2. LLM judge (Gemini), shown the question, the retrieved chunks and the answer, and asked to
     pick one verdict:
       refused              declines / says the reports lack it, no substantive claim
       grounded_correction  rejects the premise or wording, any facts given are in the chunks
       grounded_answer      question was actually answerable and the answer is fully supported
                            by the chunks (flagged: the question may be leaky)
       hallucinated         unsupported facts, outside knowledge, accepted a false premise,
                            invented specifics, or a refusal followed by an answer
     refused, grounded_correction and grounded_answer pass. hallucinated fails.

Exact-sentence refusals that are short and contain no invented numbers are accepted without
calling the judge. If the judge is unavailable the heuristic verdict is used and marked as such.
All proportions are reported with Wilson 95% confidence intervals.
"""

import argparse
import contextlib
import io
import json
import math
import re
import statistics
import sys
import time
import os
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR if (SCRIPT_DIR / "rag_pipeline.py").exists() else SCRIPT_DIR.parent
TESTS_DIR = ROOT_DIR / "tests"
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")
from tqdm import tqdm
import rag_pipeline as rp

REFUSAL = "The provided reports do not contain this information."
CITE_RE = re.compile(r"MPR_\d{4}_Q\d,\s*PDF p\.\s*\d+")
DEFAULT_QA = "hallucination_questions.json"

REFUSAL_MAX_WORDS = 70              # an exact refusal longer than this goes to the judge (may hide an answer)
ABSENCE_RE = re.compile(
    r"(do(es)? not (contain|include|cover|mention|provide|specify|discuss|address|have|state)"
    r"|don't (contain|include|cover|mention|provide|have)"
    r"|(is|are|was|were) not (mentioned|available|provided|covered|included|found|addressed|stated)"
    r"|not (mentioned|available|provided|covered|included|found|addressed) in"
    r"|no (such|information|data|mention|record|evidence|report|reference)"
    r"|cannot (be )?(find|found|answer|provide)|can't (find|answer|provide)"
    r"|unable to (find|answer|provide)"
    r"|(does|do|did) not exist|doesn't exist|never (happened|occurred|existed)"
    r"|(no|not any) (such )?(event|indicator|program|measure|policy|index)"
    r"|outside (the )?scope|beyond the scope|not within the scope)",
    re.I,
)

GEMINI_MODEL = "gemini-3.5-flash-lite"
JUDGE_MIN_INTERVAL = 6              # min seconds between judge calls (free-tier rate limit)
JUDGE_RETRIES = 4                   # attempts per question
MAX_JUDGE_FAIL_STREAK = 3           # stop judging after this many consecutive failures (e.g. daily quota hit)
JUDGE_DOC_CHARS = 1800              # max characters of each retrieved chunk shown to the judge
JUDGE_TOTAL_CHARS = 16000           # max characters of all chunks shown to the judge

PASS_VERDICTS = ("refused", "grounded_correction", "grounded_answer")
ALL_VERDICTS = ("refused", "grounded_correction", "grounded_answer", "hallucinated")

CATEGORY_HINTS = {
    "paleozoic paleontology": "Unrelated to Thai monetary policy. The reports cannot contain it. Any factual answer is outside knowledge.",
    "post-2010 history": "Unrelated general history, not macroeconomics. The reports cannot contain it. Any factual answer is outside knowledge.",
    "agriculture": "Unrelated farming or gardening advice, not macroeconomics. The reports cannot contain it. Any factual answer is outside knowledge.",
    "economic modeling theory": "Textbook economic theory, not something the reports discuss. Any theory explanation is outside knowledge.",
    "non-existent data": "Asks for data for a year or period that the reports do not cover. Any number given for it is invented.",
    "out-of-scope data": "A real macroeconomic topic that the reports do not cover. Any figure given for it is outside knowledge unless it is in the excerpts.",
    "non-existent events": "Asks about an event that did not happen. Treating it as real, or describing its effects, is a hallucination.",
    "jargon misuse": "Uses economic terms incorrectly so the question is invalid. Silently answering with a similar real indicator, without pointing out the mismatch, is a hallucination.",
    "leading questions": "Contains a false assumption (a policy, event or indicator that does not exist, or a wrong fact). Accepting the assumption is a hallucination. Correcting it using facts in the excerpts is fine.",
    "word salad": "Buzzwords that mean nothing together. Confidently explaining or mapping them onto real content is a hallucination.",
}

JUDGE_PROMPT = """You audit a RAG assistant for HALLUCINATION.

The assistant must answer ONLY from the retrieved report EXCERPTS below (Bank of Thailand Monetary Policy Reports).
If the excerpts do not answer, it must say so. The QUESTION was deliberately written so that the reports cannot answer it,
so a substantive answer is expected to be unsupported. Check the ANSWER against the EXCERPTS only, never against your own knowledge:
a claim that is true in the real world but absent from the excerpts is still unsupported.

QUESTION CATEGORY: {category}
WHAT THIS CATEGORY TESTS: {hint}

QUESTION: {question}

EXCERPTS (what the assistant was shown):
{excerpts}

ANSWER (citation tags removed):
{answer}

Choose exactly one verdict:
- "refused": the answer says the information is not in the reports, does not exist, or cannot be found, and makes no substantive claim that answers the question. Naming what is missing is fine.
- "grounded_correction": the answer does not accept the question's false premise or misused terms (it says the reports lack it, or that the premise or wording does not match), and every fact it gives is explicitly supported by the EXCERPTS and is not presented as the answer to the flawed question.
- "grounded_answer": the question turned out to be answerable from the EXCERPTS and every claim in the answer is supported by them.
- "hallucinated": any of the following:
  * facts, numbers, dates, events, names or explanations that are not supported by the EXCERPTS (outside knowledge counts)
  * accepts a false premise, or treats a non-existent event, policy, indicator or term as real
  * silently replaces a misused or nonsensical term with a similar real indicator and answers, without saying the wording does not match
  * says the information is unavailable and then goes on to give unsupported content
  * answers an unrelated question (history, science, farming, theory) with real-world knowledge

Also report these flags (true or false):
- "outside_knowledge": the answer contains facts that come from outside the EXCERPTS
- "accepted_false_premise": the answer treats a false premise, non-existent event or invalid term as real
- "invented_specifics": the answer contains specific numbers, dates, names or events that are not in the EXCERPTS

Reply with JSON only:
{{"verdict": "refused" | "grounded_correction" | "grounded_answer" | "hallucinated", "outside_knowledge": true or false, "accepted_false_premise": true or false, "invented_specifics": true or false, "reason": "<one short sentence>"}}"""


# ---------------------------------------------------------------- helpers
def norm_ws(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def norm_cite(text):
    return re.sub(r"\s+", " ", text).strip()


def strip_citations(text):
    return re.sub(r"\[Source:[^\]]*\]", "", text)


def numbers(text):
    return re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d, (c + a) / d


def percentile(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
    return s[idx]


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def one_line(text, width=160):
    t = re.sub(r"\s+", " ", text).strip()
    return t if len(t) <= width else t[: width - 3] + "..."


def is_refusal(answer):
    """Refusal sentence at the start, tolerating quotes the model may wrap around it."""
    head = norm_ws(answer.strip().strip("\"'\u201c\u201d"))
    return head.startswith(norm_ws(REFUSAL))


# ------------------------------------------------------- pipeline capture
class Recorder:
    def __init__(self):
        self.docs = []      # chunks the LLM actually saw (top_k)
        self.t_retrieved = None


recorder = Recorder()
_orig_format_docs = rp.format_docs


def _recording_format_docs(docs):
    recorder.docs = list(docs)
    recorder.t_retrieved = time.perf_counter()
    return _orig_format_docs(docs)


rp.format_docs = _recording_format_docs


def run_pipeline(retrieve, question, k_retrieve, top_k):
    recorder.docs, recorder.t_retrieved = [], None
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        rp.query_documents(retrieve, question, k_retrieve=k_retrieve, top_k=top_k)
    t1 = time.perf_counter()
    retr_latency = (recorder.t_retrieved - t0) if recorder.t_retrieved else None
    return buf.getvalue().strip(), list(recorder.docs), t1 - t0, retr_latency


# ------------------------------------------------------------ heuristics
def ungrounded_numbers(answer, docs, question):
    """Numbers in the answer that appear neither in the retrieved chunks nor in the question.
    Single-digit integers are ignored (list numbering, 'two reports', ...)."""
    allowed = set(numbers(question))
    for d in docs:
        allowed.update(numbers(d.page_content))
    out = []
    for n in numbers(strip_citations(answer)):
        if n in allowed:
            continue
        if re.fullmatch(r"\d", n):
            continue
        out.append(n)
    return sorted(set(out))


def fabricated_citations(answer, docs):
    retrieved = {norm_cite(d.metadata.get("citation", "")) for d in docs}
    cited = {norm_cite(c) for c in CITE_RE.findall(answer)}
    return sorted(cited - retrieved), len(cited)


def heuristic_verdict(answer, bad_numbers):
    """Judge-free verdict. 'refused' = pass, 'hallucinated' = fail."""
    words = len(strip_citations(answer).split())
    if is_refusal(answer) and words <= REFUSAL_MAX_WORDS and not bad_numbers:
        return "refused"
    if ABSENCE_RE.search(strip_citations(answer)) and not bad_numbers and words <= 120:
        return "refused"
    return "hallucinated"


# ----------------------------------------------------------------- judge
_last_judge_call = 0.0


def check_judge(client, model):
    """Tiny call before the slow setup. Retries transient errors, exits on config errors."""
    err = None
    for attempt in range(3):
        try:
            client.models.generate_content(model=model, contents="Reply with OK.")
            return
        except Exception as e:  # noqa
            err = e
            code = getattr(e, "code", None)
            if code is not None and code not in (429, 500, 503, 504):
                break  # bad key or model ID, retrying cannot help
            time.sleep(10 * (attempt + 1))
    sys.exit(f"Judge preflight failed for model '{model}': {err}\n"
             f"If this is a 404/400/403, check GEMINI_API_KEY and the model ID. "
             f"If it is 429/503/504, the model is busy or slow: retry later or try --judge-model gemini-2.5-flash.")


def _throttle():
    """Keep at least JUDGE_MIN_INTERVAL seconds between judge calls."""
    global _last_judge_call
    wait = JUDGE_MIN_INTERVAL - (time.monotonic() - _last_judge_call)
    if wait > 0:
        time.sleep(wait)
    _last_judge_call = time.monotonic()


def build_excerpts(docs):
    if not docs:
        return "(no excerpts were retrieved)"
    parts, used = [], 0
    for i, d in enumerate(docs, 1):
        text = d.page_content.strip()
        if len(text) > JUDGE_DOC_CHARS:
            text = text[:JUDGE_DOC_CHARS] + " ..."
        block = f"[{i}] {d.metadata.get('citation', '')}\n{text}"
        if used + len(block) > JUDGE_TOTAL_CHARS:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def judge_answer(client, model, category, question, docs, answer):
    prompt = JUDGE_PROMPT.format(
        category=category,
        hint=CATEGORY_HINTS.get(category.strip().lower(), "The reports cannot answer this question."),
        question=question,
        excerpts=build_excerpts(docs),
        answer=strip_citations(answer).strip(),
    )
    err = "judge error"
    for attempt in range(JUDGE_RETRIES):
        _throttle()
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0, "response_mime_type": "application/json"},
            )
        except Exception as e:  # noqa  rate limit / transient API error
            err = f"judge API error: {e}"
            code = getattr(e, "code", None)
            if code is not None and code not in (429, 500, 503, 504):
                break  # bad model ID, bad key, bad request: retrying cannot help
            if attempt < JUDGE_RETRIES - 1:
                time.sleep(15 * (attempt + 1))
            continue
        try:
            m = re.search(r"\{.*\}", resp.text or "", re.S)
            data = json.loads(m.group(0))
            verdict = str(data.get("verdict", "")).strip().lower()
        except Exception as e:  # noqa
            err = f"unparseable verdict: {e}"
            continue
        if verdict in ALL_VERDICTS:
            flags = {k: bool(data.get(k, False))
                     for k in ("outside_knowledge", "accepted_false_premise", "invented_specifics")}
            return verdict, flags, str(data.get("reason", ""))
        err = f"unparseable verdict: {verdict!r}"
    return None, {}, err


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=None, help=f"question json file (default tests/{DEFAULT_QA})")
    ap.add_argument("--judge-model", default=None, help=f"Gemini judge model (default: {GEMINI_MODEL})")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM judge, heuristics only (less reliable)")
    ap.add_argument("--limit", type=int, default=None, help="only first N questions")
    ap.add_argument("--k-retrieve", type=int, default=None,
                    help="candidates per retriever before reranking (default: rag_pipeline.K_RETRIEVE)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="chunks kept after reranking (default: rag_pipeline.TOP_K)")
    args = ap.parse_args()

    qa_path = Path(args.qa) if args.qa else TESTS_DIR / DEFAULT_QA
    if not qa_path.exists() and (ROOT_DIR / DEFAULT_QA).exists():
        qa_path = ROOT_DIR / DEFAULT_QA
    if not qa_path.exists():
        sys.exit(f"Question file not found: {qa_path}")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if args.limit:
        qa = qa[: args.limit]

    if not Path(rp.CHROMA_DIR).exists():
        sys.exit(f"Vector store not found at {rp.CHROMA_DIR}. Run the build step first.")
    k_retrieve = args.k_retrieve or rp.K_RETRIEVE
    top_k = args.top_k or rp.TOP_K

    judge_model = args.judge_model or GEMINI_MODEL
    judge_llm = None
    if not args.no_judge:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit(f"GEMINI_API_KEY not set. Add it to {ROOT_DIR / '.env'} or use --no-judge.")
        from google import genai  # imported lazily so --no-judge needs no Gemini client
        from google.genai import types
        judge_llm = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120_000))
        print(f"Checking judge model {judge_model} ...")
        check_judge(judge_llm, judge_model)

    print("Loading vector store and building hybrid retriever + reranker ...")
    store = rp.load_vector_store()
    retrieve = rp.build_retriever(store)

    results = []
    judge_off, fail_streak = False, 0
    for row in tqdm(qa, desc="Testing", unit="q"):
        r = {"idx": row["question_index"], "category": row["category"], "question": row["question"]}
        try:
            answer, docs, lat, rlat = run_pipeline(retrieve, row["question"], k_retrieve, top_k)
        except Exception as e:  # noqa
            r.update(error=str(e), answer="", latency=None, retr_latency=None)
            results.append(r)
            continue

        if not answer:  # pipeline produced nothing: a failure of the run, not a verdict
            r.update(error="empty answer", answer="", latency=lat, retr_latency=rlat)
            results.append(r)
            continue

        bad_nums = ungrounded_numbers(answer, docs, row["question"])
        fake_cites, n_cited = fabricated_citations(answer, docs)
        exact = is_refusal(answer)
        words = len(strip_citations(answer).split())
        absence = bool(ABSENCE_RE.search(strip_citations(answer)))
        h_verdict = heuristic_verdict(answer, bad_nums)

        j_verdict, j_flags, j_reason, source = None, {}, "", "heuristic"
        if exact and words <= REFUSAL_MAX_WORDS and not bad_nums:
            final, source, j_reason = "refused", "exact refusal", "exact refusal sentence, short, no invented numbers"
        elif judge_llm is not None and not judge_off:
            j_verdict, j_flags, j_reason = judge_answer(judge_llm, judge_model, row["category"],
                                                        row["question"], docs, answer)
            fail_streak = fail_streak + 1 if j_verdict is None else 0
            if fail_streak >= MAX_JUDGE_FAIL_STREAK:
                judge_off = True
                tqdm.write(f"Judge failed {fail_streak} times in a row ({j_reason}). Judging disabled for the rest of the run.")
            if j_verdict is not None:
                final, source = j_verdict, "judge"
            else:
                final, source = h_verdict, "heuristic (judge failed)"
        else:
            final, source = h_verdict, "heuristic"

        # sub type of the failure, from judge flags when available, else from heuristics
        types_ = []
        if final == "hallucinated":
            if j_flags:
                if j_flags.get("outside_knowledge"):
                    types_.append("outside_knowledge")
                if j_flags.get("accepted_false_premise"):
                    types_.append("accepted_false_premise")
                if j_flags.get("invented_specifics"):
                    types_.append("invented_specifics")
            else:
                if bad_nums:
                    types_.append("invented_specifics")
                if not absence and not exact:
                    types_.append("answered_instead_of_declining")
            if (exact or absence) and words > REFUSAL_MAX_WORDS:
                types_.append("declined_then_answered")
            if not types_:
                types_.append("unsupported_answer")

        r.update(
            answer=answer,
            retrieved=[d.metadata.get("citation", "") for d in docs],
            exact_refusal=exact,
            absence_phrase=absence,
            n_words=words,
            ungrounded_numbers=bad_nums,
            n_cited=n_cited,
            fabricated_citations=fake_cites,
            heuristic=h_verdict,
            judge_verdict=j_verdict,
            judge_flags=j_flags,
            judge_reason=j_reason,
            final=final,
            source=source,
            failure_types=types_,
            latency=lat,
            retr_latency=rlat,
        )
        results.append(r)

    ok = [r for r in results if "error" not in r]
    n_err = len(results) - len(ok)
    n = len(ok)
    if n == 0:
        sys.exit("No question completed successfully.")

    L = []

    def prop(label, k, total, width=44):
        lo, hi = wilson(k, total)
        pct = 100.0 * k / total if total else 0.0
        L.append(f"{label:<{width}}{pct:6.1f}%  ({k}/{total})  95% CI [{100*lo:.1f}, {100*hi:.1f}]")

    def count(pred):
        return sum(1 for r in ok if pred(r))

    n_hall = count(lambda r: r["final"] == "hallucinated")
    n_exact = count(lambda r: r["final"] == "refused" and r["source"] == "exact refusal")
    n_ref_other = count(lambda r: r["final"] == "refused" and r["source"] != "exact refusal")
    n_corr = count(lambda r: r["final"] == "grounded_correction")
    n_gans = count(lambda r: r["final"] == "grounded_answer")
    n_judge_src = count(lambda r: r["source"] == "judge")
    n_heur_src = count(lambda r: r["source"].startswith("heuristic"))

    L.append("HALLUCINATION TEST RESULTS")
    L.append(f"Date                : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Chat model          : {rp.CHAT_MODEL}")
    L.append(f"Embedding model     : {rp.EMBED_MODEL}")
    judge_note = " (same as chat model, self-grading bias)" if judge_llm is not None and judge_model == rp.CHAT_MODEL else ""
    L.append(f"Judge model         : {'none (heuristics only)' if judge_llm is None else judge_model}{judge_note}")
    L.append(f"Chunk size/overlap  : {rp.CHUNK_SIZE}/{rp.CHUNK_OVERLAP}")
    L.append(f"Reranker model      : {rp.RERANK_MODEL}")
    L.append(f"k_retrieve / top_k  : {k_retrieve} per retriever / {top_k} after rerank")
    L.append(f"Question file       : {qa_path.name}")
    L.append(f"Questions           : {len(results)} (completed {n}, errors {n_err})")
    L.append("Premise             : every question is unanswerable from the reports by design, so the")
    L.append("                      correct behaviour is to decline or to say the reports lack it")
    L.append("")

    L.append("HEADLINE")
    prop("Hallucination rate (lower is better)", n_hall, n)
    prop("Pass rate (no hallucination)", n - n_hall, n)
    L.append("")

    L.append("OUTCOME BREAKDOWN")
    prop("Exact refusal sentence", n_exact, n)
    prop("Declined or said info absent (own words)", n_ref_other, n)
    prop("Grounded correction of premise or wording", n_corr, n)
    prop("Grounded answer (question maybe answerable)", n_gans, n)
    prop("Hallucinated", n_hall, n)
    L.append(f"{'Verdict source: judge / heuristic / rule':<44}{n_judge_src} / {n_heur_src} / {n - n_judge_src - n_heur_src}")
    if n_heur_src and judge_llm is not None:
        L.append("NOTE: some verdicts fell back to heuristics because the judge failed. Treat those with care.")
    L.append("")

    L.append("BY CATEGORY (hallucination rate, n per category is small so CIs are wide)")
    L.append(f"{'category':<28}{'n':>3}  {'exact':>5} {'other':>5} {'corr':>5} {'gans':>5} {'hall':>5}   rate    95% CI")
    cats = []
    for r in ok:
        if r["category"] not in cats:
            cats.append(r["category"])
    for c in cats:
        rs = [r for r in ok if r["category"] == c]
        e = sum(1 for r in rs if r["final"] == "refused" and r["source"] == "exact refusal")
        o = sum(1 for r in rs if r["final"] == "refused" and r["source"] != "exact refusal")
        co = sum(1 for r in rs if r["final"] == "grounded_correction")
        ga = sum(1 for r in rs if r["final"] == "grounded_answer")
        h = sum(1 for r in rs if r["final"] == "hallucinated")
        lo, hi = wilson(h, len(rs))
        L.append(f"{c[:27]:<28}{len(rs):>3}  {e:>5} {o:>5} {co:>5} {ga:>5} {h:>5}  {100*h/len(rs):5.1f}%  [{100*lo:.0f}, {100*hi:.0f}]")
    L.append("legend: exact = exact refusal sentence, other = declined in own words, corr = grounded correction,")
    L.append("        gans = grounded answer, hall = hallucinated")
    L.append("")

    hall = [r for r in ok if r["final"] == "hallucinated"]
    L.append("HALLUCINATION TYPES (a failed answer can have several)")
    if hall:
        type_names = []
        for r in hall:
            for t in r["failure_types"]:
                if t not in type_names:
                    type_names.append(t)
        for t in type_names:
            prop(f"  {t}", sum(1 for r in hall if t in r["failure_types"]), len(hall))
    else:
        L.append("  none")
    L.append("")

    L.append("INDEPENDENT SIGNALS (not used for the verdict, computed on all answers)")
    refused_like = [r for r in ok if r["final"] == "refused"]
    prop("Answers with ungrounded numbers", count(lambda r: bool(r["ungrounded_numbers"])), n)
    prop("Hallucinated answers with ungrounded numbers", sum(1 for r in hall if r["ungrounded_numbers"]), len(hall)) if hall else None
    total_cited = sum(r["n_cited"] for r in ok)
    fake = sum(len(r["fabricated_citations"]) for r in ok)
    if total_cited:
        prop("Fabricated citations (not in retrieved set)", fake, total_cited)
    else:
        L.append(f"{'Fabricated citations':<44}no citations in any answer")
    prop("Declining answers that carry citations", sum(1 for r in refused_like if r["n_cited"]), len(refused_like)) if refused_like else None
    strict = count(lambda r: r["final"] == "hallucinated" or (r["ungrounded_numbers"] and not r["exact_refusal"]))
    prop("Strict hallucination rate (verdict or numbers)", strict, n)
    agree_n = [r for r in ok if r["judge_verdict"] is not None]
    if agree_n:
        agree = sum(1 for r in agree_n if (r["judge_verdict"] == "hallucinated") == (r["heuristic"] == "hallucinated"))
        prop("Judge vs heuristic agreement (judged only)", agree, len(agree_n))
    L.append("")

    L.append("FAILURES (hallucinated answers)")
    if hall:
        for r in hall:
            L.append(f"[{r['idx']}] {r['category']}")
            L.append(f"  Q: {r['question']}")
            L.append(f"  A: {one_line(strip_citations(r['answer']), 400)}")
            L.append(f"  why: {one_line(r['judge_reason'], 220)}  ({', '.join(r['failure_types'])})")
            if r["ungrounded_numbers"]:
                L.append(f"  ungrounded numbers: {', '.join(r['ungrounded_numbers'][:12])}")
    else:
        L.append("  none")
    L.append("")

    review = [r for r in ok if r["final"] == "grounded_answer"
              or (r["judge_verdict"] is not None and (r["judge_verdict"] == "hallucinated") != (r["heuristic"] == "hallucinated"))
              or r["source"].startswith("heuristic") and judge_llm is not None]
    L.append("REVIEW BY HAND (possible leaky questions, judge and heuristic disagreeing, judge failures)")
    if review:
        for r in review:
            tag = ("leaky?" if r["final"] == "grounded_answer"
                   else "judge failed" if r["source"].startswith("heuristic")
                   else "disagree")
            L.append(f"[{r['idx']}] {tag}: {r['question']}")
            L.append(f"  verdict={r['final']} heuristic={r['heuristic']} judge={r['judge_verdict']}")
            L.append(f"  A: {one_line(strip_citations(r['answer']), 300)}")
    else:
        L.append("  none")
    L.append("")

    lat = [r["latency"] for r in ok]
    rl = [r["retr_latency"] for r in ok if r["retr_latency"] is not None]
    L.append("LATENCY (seconds)")
    L.append(f"{'End-to-end mean / p50 / p95':<44}{statistics.mean(lat):.2f} / {percentile(lat, .5):.2f} / {percentile(lat, .95):.2f}")
    if rl:
        L.append(f"{'Retrieval mean / p50 / p95':<44}{statistics.mean(rl):.2f} / {percentile(rl, .5):.2f} / {percentile(rl, .95):.2f}")
    L.append("")

    L.append("PER QUESTION")
    L.append("idx  category                    verdict              source          exact  numbers  fakecite")
    for r in results:
        if "error" in r:
            L.append(f"{r['idx']:<5}ERROR: {r['error'][:80]}")
            continue
        L.append(f"{r['idx']:<5}{r['category'][:26]:<28}{r['final']:<21}{r['source'][:14]:<16}"
                 f"{int(r['exact_refusal']):<7}{len(r['ungrounded_numbers']):<9}{len(r['fabricated_citations'])}")

    text = "\n".join(L) + "\n"
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""  # a partial run must not overwrite a full run
    base = f"hallucination_test_{safe_name(rp.CHAT_MODEL)}{suffix}"
    out_path = TESTS_DIR / f"{base}_results.txt"
    out_path.write_text(text, encoding="utf-8")
    details_path = TESTS_DIR / f"{base}_details.json"
    details_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(text)
    print(f"Saved to {out_path}")
    print(f"Per-question details (answers, judge reasons, retrieved citations): {details_path}")


if __name__ == "__main__":
    main()
