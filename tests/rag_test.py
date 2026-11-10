"""
RAG evaluation for rag_pipeline.py (BOT MPR assistant).

Run from the project root or from tests/:
    python tests/rag_test.py
    python tests/rag_test.py --qa tests/qa_pairs.json --judge-model gemini-3.5-flash-lite
    python tests/rag_test.py --limit 5 --no-judge
    python tests/rag_test.py --k-retrieve 30 --top-k 8

The judge is a Gemini model; put GEMINI_API_KEY in .env next to rag_pipeline.py.
The pipeline is NOT modified. The hybrid retriever (BM25 + dense + cross-encoder
reranker) is built once with rag_pipeline.build_retriever(), then
rag_pipeline.query_documents() is called as is:
  - the answer is captured from its stdout stream
  - the retriever is wrapped so the FULL reranked candidate list is recorded
    (best first); only the first top_k chunks are passed on to the LLM, exactly
    as the pipeline would. This also gives candidate-pool recall, which shows
    whether a miss comes from retrieval (pool) or from the reranker cut (top_k)
  - rag_pipeline.format_docs is wrapped to record the chunks the LLM saw

Retrieval metrics (gold = cited_chunk + citation in the QA json):
  Hit@k (= Recall@k, single gold chunk per question), MRR@top_k, nDCG@k,
  each at exact-chunk level and at page level (citation only), plus
  candidate-pool recall before the top_k cut.
Answer metrics:
  LLM-judge correctness vs the reference answer, token F1, ROUGE-L F1,
  numeric recall, citation recall, citation precision / hallucination rate,
  refusal rate, format compliance, end-to-end success, latency.
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
BASE_KS = (1, 3, 5)
JACCARD_MATCH = 0.90

GEMINI_MODEL = "gemini-3.5-flash-lite"
JUDGE_MIN_INTERVAL = 6              # min seconds between judge calls (free-tier rate limit)
JUDGE_RETRIES = 4                   # attempts per question
MAX_JUDGE_FAIL_STREAK = 3           # stop judging after this many consecutive failures (e.g. daily quota hit)

JUDGE_PROMPT = """Grade a CANDIDATE answer against a REFERENCE answer.

QUESTION: {question}

REFERENCE: {reference}

CANDIDATE: {candidate}

Verdict "correct" only if the candidate gives the same main answer as the reference, with the same key numbers,
periods and direction of change, and does not contradict it. Extra non-conflicting detail is fine.
Ignore citations and formatting. A candidate saying the information is unavailable is "incorrect".

Reply with JSON only: {{"verdict": "correct" or "incorrect", "reason": "<one short sentence>"}}"""


# ---------------------------------------------------------------- helpers
def norm_ws(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def norm_cite(text):
    return re.sub(r"\s+", " ", text).strip()


def word_set(text):
    return set(re.findall(r"\w+", text.lower()))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def strip_citations(text):
    return re.sub(r"\[Source:[^\]]*\]", "", text)


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", strip_citations(text).lower())


def numbers(text):
    return re.findall(r"\d+(?:\.\d+)?", strip_citations(text))


def token_f1(ref, cand):
    r, c = tokens(ref), tokens(cand)
    if not r or not c:
        return 0.0
    rc, cc = {}, {}
    for t in r:
        rc[t] = rc.get(t, 0) + 1
    for t in c:
        cc[t] = cc.get(t, 0) + 1
    overlap = sum(min(v, cc.get(k, 0)) for k, v in rc.items())
    if overlap == 0:
        return 0.0
    p, rcl = overlap / len(c), overlap / len(r)
    return 2 * p * rcl / (p + rcl)


def rouge_l_f1(ref, cand):
    r, c = tokens(ref), tokens(cand)
    if not r or not c:
        return 0.0
    prev = [0] * (len(c) + 1)
    for i in range(1, len(r) + 1):
        cur = [0] * (len(c) + 1)
        for j in range(1, len(c) + 1):
            if r[i - 1] == c[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, rcl = lcs / len(c), lcs / len(r)
    return 2 * p * rcl / (p + rcl)


def numeric_recall(ref, cand):
    rn = numbers(ref)
    if not rn:
        return None
    cn = set(numbers(cand))
    return sum(1 for n in rn if n in cn) / len(rn)


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


def is_refusal(answer):
    """Refusal sentence at the start, tolerating quotes the model may wrap around it."""
    head = norm_ws(answer.strip().strip("\"'\u201c\u201d"))
    return head.startswith(norm_ws(REFUSAL))


# ------------------------------------------------------- pipeline capture
class Recorder:
    def __init__(self):
        self.docs = []      # chunks the LLM actually saw (top_k)
        self.pool = []      # full reranked candidate list (best first)
        self.t_retrieved = None


recorder = Recorder()
_orig_format_docs = rp.format_docs


def _recording_format_docs(docs):
    recorder.docs = list(docs)
    recorder.t_retrieved = time.perf_counter()
    return _orig_format_docs(docs)


rp.format_docs = _recording_format_docs


def make_pool_retriever(retrieve):
    """Wrap retrieve() so the whole reranked list is recorded, but only top_k is returned."""
    def wrapped(question, k_retrieve, top_k):
        pool = retrieve(question, k_retrieve, 10 ** 6)  # rerank order is identical for any top_k
        recorder.pool = list(pool)
        return pool[:top_k]
    return wrapped


def run_pipeline(retrieve, question, k_retrieve, top_k):
    recorder.docs, recorder.pool, recorder.t_retrieved = [], [], None
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        rp.query_documents(retrieve, question, k_retrieve=k_retrieve, top_k=top_k)
    t1 = time.perf_counter()
    retr_latency = (recorder.t_retrieved - t0) if recorder.t_retrieved else None
    return (buf.getvalue().strip(), list(recorder.docs), list(recorder.pool),
            t1 - t0, retr_latency)


# ------------------------------------------------------------- retrieval
def find_ranks(docs, gold_chunk, gold_cite):
    """Return (chunk_rank, page_rank), 1-based, or None if not retrieved."""
    gold_norm = norm_ws(gold_chunk)
    gold_words = word_set(gold_chunk)
    gold_cite = norm_cite(gold_cite)
    chunk_rank, page_rank = None, None
    for i, d in enumerate(docs, 1):
        cite = norm_cite(d.metadata.get("citation", ""))
        if cite != gold_cite:
            continue
        if page_rank is None:
            page_rank = i
        text = d.page_content
        if norm_ws(text) == gold_norm or jaccard(word_set(text), gold_words) >= JACCARD_MATCH:
            chunk_rank = i
            break
    return chunk_rank, page_rank


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


def judge_answer(client, model, question, reference, candidate):
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=strip_citations(reference).strip(),
        candidate=strip_citations(candidate).strip(),
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
        if verdict in ("correct", "incorrect"):
            return (1 if verdict == "correct" else 0), str(data.get("reason", ""))
        err = f"unparseable verdict: {verdict!r}"
    return None, err


# --------------------------------------------------------------- format
def format_ok(answer):
    if not answer:
        return False
    if answer.lstrip().startswith(("-", "*", "\u2022")):
        return False
    lines = [l for l in answer.strip().splitlines() if l.strip()]
    # tolerate markdown emphasis such as **Summary:**
    if not lines or not lines[-1].strip().lstrip("*_#> ").lower().startswith("summary"):
        return False
    return len(answer.split()) <= 500


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=None, help="QA json file (default tests/qa_pairs.json)")
    ap.add_argument("--judge-model", default=None, help=f"Gemini judge model (default: {GEMINI_MODEL})")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM judge")
    ap.add_argument("--limit", type=int, default=None, help="only first N questions")
    ap.add_argument("--k-retrieve", type=int, default=None,
                    help="candidates per retriever before reranking (default: rag_pipeline.K_RETRIEVE)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="chunks kept after reranking (default: rag_pipeline.TOP_K)")
    args = ap.parse_args()

    qa_path = Path(args.qa) if args.qa else TESTS_DIR / "qa_pairs.json"
    if not qa_path.exists() and (ROOT_DIR / "qa_pairs.json").exists():
        qa_path = ROOT_DIR / "qa_pairs.json"
    if not qa_path.exists():
        sys.exit(f"QA file not found: {qa_path}")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if args.limit:
        qa = qa[: args.limit]

    if not Path(rp.CHROMA_DIR).exists():
        sys.exit(f"Vector store not found at {rp.CHROMA_DIR}. Run the build step first.")
    k_retrieve = args.k_retrieve or rp.K_RETRIEVE
    top_k = args.top_k or rp.TOP_K
    KS = tuple(sorted({k for k in BASE_KS if k <= top_k} | {top_k}))

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
    retrieve = make_pool_retriever(rp.build_retriever(store))

    results = []
    judge_off, fail_streak = False, 0
    for row in tqdm(qa, desc="Evaluating", unit="q"):
        r = {"idx": row["question_index"], "question": row["question"]}
        try:
            answer, docs, pool, lat, rlat = run_pipeline(retrieve, row["question"], k_retrieve, top_k)
        except Exception as e:  # noqa
            r.update(error=str(e), answer="", docs=[], latency=None, retr_latency=None)
            results.append(r)
            continue

        # ranks are over the full reranked pool; values above top_k mean "retrieved but cut"
        chunk_rank, page_rank = find_ranks(pool, row["cited_chunk"], row["citation"])
        gold_cite = norm_cite(row["citation"])
        cited = {norm_cite(c) for c in CITE_RE.findall(answer)}
        retrieved_cites = {norm_cite(d.metadata.get("citation", "")) for d in docs}
        refused = is_refusal(answer)

        if judge_llm is None:
            correct, reason = None, "judge skipped"
        elif refused:
            correct, reason = 0, "refusal"
        elif judge_off:
            correct, reason = None, "judge disabled after repeated failures"
        else:
            correct, reason = judge_answer(judge_llm, judge_model, row["question"], row["answer"], answer)
            fail_streak = fail_streak + 1 if correct is None else 0
            if fail_streak >= MAX_JUDGE_FAIL_STREAK:
                judge_off = True
                tqdm.write(f"Judge failed {fail_streak} times in a row ({reason}). Judging disabled for the rest of the run.")

        r.update(
            answer=answer,
            n_retrieved=len(docs),
            pool_size=len(pool),
            retrieved=[d.metadata.get("citation", "") for d in docs],
            reference=row["answer"],
            gold_citation=row["citation"],
            chunk_rank=chunk_rank,
            page_rank=page_rank,
            refused=refused,
            correct=correct,
            judge_reason=reason,
            token_f1=token_f1(row["answer"], answer),
            rouge_l=rouge_l_f1(row["answer"], answer),
            num_recall=numeric_recall(row["answer"], answer),
            n_cited=len(cited),
            cite_recall=1 if gold_cite in cited else 0,
            cite_valid=len(cited & retrieved_cites),
            fmt_ok=format_ok(answer),
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

    def prop(label, k, total):
        lo, hi = wilson(k, total)
        pct = 100.0 * k / total if total else 0.0
        L.append(f"{label:<38}{pct:6.1f}%  ({k}/{total})  95% CI [{100*lo:.1f}, {100*hi:.1f}]")

    L.append("RAG TEST RESULTS")
    L.append(f"Date                : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Chat model          : {rp.CHAT_MODEL}")
    L.append(f"Embedding model     : {rp.EMBED_MODEL}")
    judge_note = " (same as chat model, self-grading bias)" if judge_llm is not None and judge_model == rp.CHAT_MODEL else ""
    L.append(f"Judge model         : {'none' if judge_llm is None else judge_model}{judge_note}")
    L.append(f"Chunk size/overlap  : {rp.CHUNK_SIZE}/{rp.CHUNK_OVERLAP}")
    L.append("Retriever           : hybrid BM25 + dense (Ensemble, weights 0.5/0.5) + cross-encoder rerank")
    L.append(f"Reranker model      : {rp.RERANK_MODEL}")
    L.append(f"k_retrieve / top_k  : {k_retrieve} per retriever / {top_k} after rerank")
    L.append(f"QA file             : {qa_path.name}")
    L.append(f"Questions           : {len(results)} (completed {n}, errors {n_err})")
    L.append(f"Chunk match rule    : same citation and (normalized text equal or word Jaccard >= {JACCARD_MATCH})")
    L.append("")

    L.append("RETRIEVAL (exact gold chunk)")
    for k in KS:
        prop(f"Hit@{k} (= Recall@{k})", sum(1 for r in ok if r["chunk_rank"] and r["chunk_rank"] <= k), n)
    mrr = sum(1.0 / r["chunk_rank"] for r in ok if r["chunk_rank"] and r["chunk_rank"] <= top_k) / n
    L.append(f"{'MRR@' + str(top_k):<38}{mrr:8.4f}")
    for k in KS:
        nd = sum(1.0 / math.log2(r["chunk_rank"] + 1) for r in ok if r["chunk_rank"] and r["chunk_rank"] <= k) / n
        L.append(f"{'nDCG@' + str(k):<38}{nd:8.4f}")
    avg_pool = statistics.mean(r["pool_size"] for r in ok)
    prop(f"Candidate pool recall (avg {avg_pool:.0f} cands)", sum(1 for r in ok if r["chunk_rank"]), n)
    L.append("")

    L.append("RETRIEVAL (gold page, citation only)")
    for k in KS:
        prop(f"Page Hit@{k}", sum(1 for r in ok if r["page_rank"] and r["page_rank"] <= k), n)
    pmrr = sum(1.0 / r["page_rank"] for r in ok if r["page_rank"] and r["page_rank"] <= top_k) / n
    L.append(f"{'Page MRR@' + str(top_k):<38}{pmrr:8.4f}")
    prop("Candidate pool page recall", sum(1 for r in ok if r["page_rank"]), n)
    L.append("")

    judged = [r for r in ok if r["correct"] is not None]
    L.append("ANSWER QUALITY")
    if judged:
        prop("Answer correctness (LLM judge)", sum(r["correct"] for r in judged), len(judged))
        L.append(f"{'Judge failures (excluded)':<38}{len(ok) - len(judged):8d}")
        hit = [r for r in judged if r["chunk_rank"] and r["chunk_rank"] <= top_k]
        miss = [r for r in judged if not (r["chunk_rank"] and r["chunk_rank"] <= top_k)]
        if hit:
            prop("Correctness | gold chunk retrieved", sum(r["correct"] for r in hit), len(hit))
        if miss:
            prop("Correctness | gold chunk missed", sum(r["correct"] for r in miss), len(miss))
        e2e = sum(1 for r in judged if r["chunk_rank"] and r["chunk_rank"] <= top_k and r["correct"])
        prop("End-to-end success (hit and correct)", e2e, len(judged))
    else:
        L.append("Answer correctness (LLM judge)    skipped")
    L.append(f"{'Token F1 vs reference (mean)':<38}{statistics.mean(r['token_f1'] for r in ok):8.4f}")
    L.append(f"{'ROUGE-L F1 vs reference (mean)':<38}{statistics.mean(r['rouge_l'] for r in ok):8.4f}")
    nr = [r["num_recall"] for r in ok if r["num_recall"] is not None]
    if nr:
        L.append(f"{'Numeric recall vs reference (mean)':<38}{statistics.mean(nr):8.4f}")
    prop("Refusal rate", sum(1 for r in ok if r["refused"]), n)
    L.append("")

    L.append("CITATIONS AND FORMAT")
    prop("Gold citation present in answer", sum(r["cite_recall"] for r in ok), n)
    total_cited = sum(r["n_cited"] for r in ok)
    valid = sum(r["cite_valid"] for r in ok)
    prop("Citation precision (in retrieved set)", valid, total_cited)
    prop("Hallucinated citation rate", total_cited - valid, total_cited)
    answered = [r for r in ok if not r["refused"]]  # refusals have no Summary line by design
    if answered:
        prop("Format compliance (non-refusals)", sum(1 for r in answered if r["fmt_ok"]), len(answered))
    L.append("")

    lat = [r["latency"] for r in ok]
    rl = [r["retr_latency"] for r in ok if r["retr_latency"] is not None]
    L.append("LATENCY (seconds)")
    L.append(f"{'End-to-end mean / p50 / p95':<38}{statistics.mean(lat):.2f} / {percentile(lat, .5):.2f} / {percentile(lat, .95):.2f}")
    if rl:
        L.append(f"{'Retrieval mean / p50 / p95':<38}{statistics.mean(rl):.2f} / {percentile(rl, .5):.2f} / {percentile(rl, .95):.2f}")
    L.append("")

    L.append("PER QUESTION")
    L.append("idx  chunk_rank  page_rank  judge  tokF1  gold_cite  refused")
    for r in results:
        if "error" in r:
            L.append(f"{r['idx']:<5}ERROR: {r['error'][:80]}")
            continue
        cr = r["chunk_rank"] if r["chunk_rank"] else "-"
        pr = r["page_rank"] if r["page_rank"] else "-"
        jd = "-" if r["correct"] is None else r["correct"]
        L.append(f"{r['idx']:<5}{str(cr):<12}{str(pr):<11}{str(jd):<7}{r['token_f1']:<7.3f}{r['cite_recall']:<11}{int(r['refused'])}")

    text = "\n".join(L) + "\n"
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""  # a partial run must not overwrite a full run
    base = f"rag_test_{safe_name(rp.CHAT_MODEL)}{suffix}"
    out_path = TESTS_DIR / f"{base}_results.txt"
    out_path.write_text(text, encoding="utf-8")
    details_path = TESTS_DIR / f"{base}_details.json"
    details_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(text)
    print(f"Saved to {out_path}")
    print(f"Per-question details (answers, judge reasons, retrieved citations): {details_path}")


if __name__ == "__main__":
    main()