"""
RAG evaluation for rag_pipeline.py (BOT MPR assistant).

Run from the project root or from tests/:
    python tests/rag_test.py
    python tests/rag_test.py --qa tests/qa_pairs.json --judge-model qwen3:4b-instruct
    python tests/rag_test.py --limit 5 --no-judge
    python tests/rag_test.py --k-retrieve 30 --top-k 8

The pipeline is NOT modified. The hybrid retriever (BM25 + dense + cross-encoder
reranker) is built once with rag_pipeline.build_retriever(), then
rag_pipeline.query_documents() is called as is:
  - the answer is captured from its stdout stream
  - the retrieved chunks are captured by wrapping rag_pipeline.format_docs,
    which receives the final reranked chunks (best first, top_k of them)

Retrieval metrics (gold = cited_chunk + citation in the QA json):
  Hit@k (= Recall@k, single gold chunk per question), MRR, nDCG@k,
  each at exact-chunk level and at page level (citation only).
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
from datetime import datetime
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TESTS_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from tqdm import tqdm
from langchain_ollama import ChatOllama

import rag_pipeline as rp

REFUSAL = "The provided reports do not contain this information."
CITE_RE = re.compile(r"MPR_\d{4}_Q\d,\s*PDF p\.\s*\d+")
BASE_KS = (1, 3, 5, 8)
JACCARD_MATCH = 0.90

JUDGE_PROMPT = """You are a strict grader for a question answering system.
Compare the CANDIDATE answer with the REFERENCE answer for the QUESTION.

Mark "correct" only if ALL are true:
1. The candidate gives the same direct answer as the reference.
2. Every key number, date, period and direction of change in the reference
   that is needed to answer the question appears in the candidate with the same value.
3. The candidate does not contradict the reference.
Extra detail that does not contradict the reference is allowed.
Ignore citation tags and formatting.
If the candidate says the information is not available, it is "incorrect".

QUESTION:
{question}

REFERENCE:
{reference}

CANDIDATE:
{candidate}

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


# ------------------------------------------------------- pipeline capture
class Recorder:
    def __init__(self):
        self.docs = []
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
def judge_answer(judge_llm, question, reference, candidate):
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=strip_citations(reference).strip(),
        candidate=strip_citations(candidate).strip(),
    )
    try:
        out = judge_llm.invoke(prompt).content
        m = re.search(r"\{.*\}", out, re.S)
        data = json.loads(m.group(0))
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict in ("correct", "incorrect"):
            return (1 if verdict == "correct" else 0), str(data.get("reason", ""))
    except Exception as e:  # noqa
        return None, f"judge error: {e}"
    return None, "unparseable verdict"


# --------------------------------------------------------------- format
def format_ok(answer):
    if not answer:
        return False
    if answer.lstrip().startswith(("-", "*", "\u2022")):
        return False
    lines = [l for l in answer.strip().splitlines() if l.strip()]
    if not lines or not lines[-1].strip().lower().startswith("summary:"):
        return False
    return len(answer.split()) <= 500


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=None, help="QA json file (default tests/qa_pairs.json)")
    ap.add_argument("--judge-model", default=None, help="Ollama judge model (default: CHAT_MODEL)")
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

    print("Loading vector store and building hybrid retriever + reranker ...")
    store = rp.load_vector_store()
    retrieve = rp.build_retriever(store)

    judge_model = args.judge_model or rp.CHAT_MODEL
    judge_llm = None
    if not args.no_judge:
        judge_llm = ChatOllama(model=judge_model, temperature=0, format="json")

    results = []
    for row in tqdm(qa, desc="Evaluating", unit="q"):
        r = {"idx": row["question_index"], "question": row["question"]}
        try:
            answer, docs, lat, rlat = run_pipeline(retrieve, row["question"], k_retrieve, top_k)
        except Exception as e:  # noqa
            r.update(error=str(e), answer="", docs=[], latency=None, retr_latency=None)
            results.append(r)
            continue

        chunk_rank, page_rank = find_ranks(docs, row["cited_chunk"], row["citation"])
        gold_cite = norm_cite(row["citation"])
        cited = {norm_cite(c) for c in CITE_RE.findall(answer)}
        retrieved_cites = {norm_cite(d.metadata.get("citation", "")) for d in docs}
        refused = answer.startswith(REFUSAL)

        if refused:
            correct, reason = 0, "refusal"
        elif judge_llm is not None:
            correct, reason = judge_answer(judge_llm, row["question"], row["answer"], answer)
        else:
            correct, reason = None, "judge skipped"

        r.update(
            answer=answer,
            n_retrieved=len(docs),
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
    L.append(f"Judge model         : {'none' if judge_llm is None else judge_model}")
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
    mrr = sum(1.0 / r["chunk_rank"] for r in ok if r["chunk_rank"]) / n
    L.append(f"{'MRR':<38}{mrr:8.4f}")
    for k in KS:
        nd = sum(1.0 / math.log2(r["chunk_rank"] + 1) for r in ok if r["chunk_rank"] and r["chunk_rank"] <= k) / n
        L.append(f"{'nDCG@' + str(k):<38}{nd:8.4f}")
    L.append("")

    L.append("RETRIEVAL (gold page, citation only)")
    for k in KS:
        prop(f"Page Hit@{k}", sum(1 for r in ok if r["page_rank"] and r["page_rank"] <= k), n)
    pmrr = sum(1.0 / r["page_rank"] for r in ok if r["page_rank"]) / n
    L.append(f"{'Page MRR':<38}{pmrr:8.4f}")
    L.append("")

    judged = [r for r in ok if r["correct"] is not None]
    L.append("ANSWER QUALITY")
    if judged:
        prop("Answer correctness (LLM judge)", sum(r["correct"] for r in judged), len(judged))
        L.append(f"{'Judge failures (excluded)':<38}{len(ok) - len(judged):8d}")
        hit = [r for r in judged if r["chunk_rank"]]
        miss = [r for r in judged if not r["chunk_rank"]]
        if hit:
            prop("Correctness | gold chunk retrieved", sum(r["correct"] for r in hit), len(hit))
        if miss:
            prop("Correctness | gold chunk missed", sum(r["correct"] for r in miss), len(miss))
        e2e = sum(1 for r in judged if r["chunk_rank"] and r["correct"])
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
    prop("Format compliance", sum(1 for r in ok if r["fmt_ok"]), n)
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
    out_path = TESTS_DIR / f"rag_test_{safe_name(rp.CHAT_MODEL)}_results.txt"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
