"""
Multi-source synthesis evaluation for rag_pipeline.py (BOT MPR assistant).

Every question in the QA file needs 2 or 3 gold chunks to be answered fully
(citation_1..3 / cited_chunk_1..3). The test measures whether the system
(1) retrieves ALL the gold chunks, (2) uses and cites ALL of them, and
(3) combines them into one accurate, faithful, non-mixed-up answer.

Run from the project root or from tests/:
    python tests/rag_test_multisource.py
    python tests/rag_test_multisource.py --qa tests/qa_pairs_multisource.json --judge-model gemini-3.5-flash-lite
    python tests/rag_test_multisource.py --limit 5 --no-judge
    python tests/rag_test_multisource.py --no-faithfulness
    python tests/rag_test_multisource.py --k-retrieve 30 --top-k 8

The judge is a Gemini model; put GEMINI_API_KEY in .env next to rag_pipeline.py.
The pipeline is NOT modified. The hybrid retriever (BM25 + dense + cross-encoder
reranker) is built once with rag_pipeline.build_retriever(), then
rag_pipeline.query_documents() is called as is:
  - the answer is captured from its stdout stream
  - the retriever is wrapped so the FULL reranked candidate list is recorded
    (best first); only the first top_k chunks are passed on to the LLM, exactly
    as the pipeline would. Pool metrics show whether a miss comes from
    retrieval (pool) or from the reranker cut (top_k)
  - rag_pipeline.format_docs is wrapped to record the chunks the LLM saw

METRICS (standard retrieval / RAG evaluation practice, extended to several gold chunks)
Retrieval, multi-gold, exact chunk and page level:
  Recall@k (macro mean over questions), Any-hit@k, FULL-COVERAGE@k (all gold
  chunks in the top k, the key multi-hop retrieval metric, as in HotpotQA
  supporting-fact and MultiHop-RAG evaluations), Precision@k, MRR (first gold),
  MAP@top_k (RAGAS-style context precision), nDCG@k, plus the same coverage
  before the top_k cut (candidate pool).
Source use in the answer (deterministic):
  cited-gold recall, ALL-gold-cited rate, multi-source citation rate,
  citation validity / hallucinated citations, per-line citation coverage,
  retrieval-to-citation conversion (all gold retrieved AND all gold cited).
Answer quality (Gemini judge, reference-based):
  accuracy, per-source completeness, full completeness, integration, cross-source
  misattribution rate, MULTI-SOURCE SUCCESS (accurate and complete and no mix-up).
Grounding (Gemini judge, context-based, RAGAS faithfulness and ALCE citation precision):
  faithfulness (supported claims / claims), fully-faithful rate, citation support rate.
Lexical (reference-free of judge): token F1, ROUGE-L F1, numeric recall.
Also: refusal / partial-refusal rate, format compliance, failure funnel,
breakdown by number of required sources, latency.
All proportions use Wilson 95% confidence intervals; all means use a seeded
bootstrap 95% confidence interval.
"""

import argparse
import contextlib
import io
import json
import math
import random
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
BASE_KS = (2, 3, 5, 8)
JACCARD_MATCH = 0.90
QA_DEFAULT = "qa_pairs_multisource.json"

GEMINI_MODEL = "gemini-3.5-flash-lite"
JUDGE_MIN_INTERVAL = 6              # min seconds between judge calls (free-tier rate limit)
JUDGE_RETRIES = 4                   # attempts per judge call
MAX_JUDGE_FAIL_STREAK = 3           # stop judging after this many consecutive failures (e.g. daily quota hit)
BOOTSTRAP_N = 2000

# ------------------------------------------------------------ judge prompts
SYNTHESIS_JUDGE_PROMPT = """You are grading a retrieval-augmented system on a MULTI-SOURCE SYNTHESIS question.
The question can only be answered completely by combining facts from {n_sources} different sources
(different reports, pages or passages). Grade the CANDIDATE against the REFERENCE.

QUESTION:
{question}

REQUIRED SOURCES ({n_sources}):
{sources}

REFERENCE ANSWER (ideal answer built from all required sources; each [Source: ...] tag shows which source
supports that statement):
{reference}

CANDIDATE ANSWER (citation tags removed):
{candidate}

Judge meaning, not wording. Ignore formatting and citations. The candidate may contain extra correct detail
that is not in the reference; that is fine unless it contradicts the reference.

Decide four things:

1. "accurate": true only if every statement in the candidate about the same report, period and metric as the
   reference agrees with the reference: same numbers, units, periods, direction of change (up, down, unchanged),
   YoY vs QoQ, and forecast vs actual. false if any key number, period or direction is wrong or contradicts the
   reference, or if the candidate says the information is unavailable.

2. "covered": a list with exactly {n_sources} booleans, one per required source in the order listed above.
   true if the candidate clearly states the key fact(s) the reference takes from that source (also true if
   worded differently). false if that source's facts are missing, only hinted at, or too vague to be checked.

3. "misattributed": true if the candidate assigns a fact, figure or period from one report or period to a
   different one (for example mixing the figures of two quarters, or presenting one source's value as another's),
   or merges figures from different sources into a single wrong figure. Otherwise false.

4. "integrated": true if the candidate answers the question as one coherent answer that relates the sources
   (for example compares them, traces a change over time, or combines complementary facts). false if it answers
   from only one source, or presents the sources as unrelated disconnected lists without addressing the question.

Reply with JSON only:
{{"accurate": true or false, "covered": [true or false, ...], "misattributed": true or false, "integrated": true or false, "reason": "<one short sentence>"}}"""

FAITHFULNESS_JUDGE_PROMPT = """You are checking whether an ANSWER is fully supported by the CONTEXT it was written from.
The CONTEXT is a set of retrieved passages, each labeled with a [Source: ...] tag. The ANSWER uses the same tags
to cite its statements.

CONTEXT:
{context}

ANSWER:
{answer}

Do two checks.

A. Claims. Split the ANSWER into atomic factual claims (each number, period, direction of change, or stated
   cause is one claim; the "Summary:" line counts too). A claim is "supported" if the CONTEXT states it or
   directly entails it. A claim that needs outside knowledge, goes beyond the CONTEXT, or states a cause or
   implication the CONTEXT does not state is unsupported. Ignore the citation tags for this check.

B. Citations. Count the statements in the ANSWER that end with a [Source: ...] tag. For each, check the passage
   with that exact tag. The statement is "wrongly cited" if that passage does not support it (for example the fact
   is actually in a different passage, or in none).

Reply with JSON only:
{{"claims": <integer>, "supported": <integer>, "unsupported": ["<claim>", ...up to 5], "cited_statements": <integer>, "wrongly_cited": <integer>}}"""


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


def bootstrap_ci(values, n_boot=BOOTSTRAP_N, seed=13):
    """Seeded percentile bootstrap 95% CI of the mean."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    m = len(values)
    means = sorted(sum(values[rng.randrange(m)] for _ in range(m)) / m for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))]


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


def has_partial_refusal(answer):
    """Refusal sentence appears later in the answer: the system answered only part and named what is missing."""
    return norm_ws(REFUSAL) in norm_ws(answer) and not is_refusal(answer)


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


# --------------------------------------------------------------- QA file
def load_golds(row):
    """Gold (citation, chunk) pairs of one question. Empty citation columns are skipped."""
    golds = []
    for k in (1, 2, 3):
        cite = (row.get(f"citation_{k}") or "").strip()
        chunk = row.get(f"cited_chunk_{k}") or ""
        if cite and chunk:
            golds.append((norm_cite(cite), chunk))
    return golds


# ------------------------------------------------------------- retrieval
def match_golds(docs, golds):
    """
    For each gold chunk return its 1-based rank in docs, or None.
    A doc matches a gold chunk if it has the same citation and equal normalized text
    or word Jaccard >= JACCARD_MATCH. Each doc can satisfy only one gold chunk.
    """
    ranks, used = [], set()
    for cite, chunk in golds:
        gold_norm, gold_words = norm_ws(chunk), word_set(chunk)
        found = None
        for i, d in enumerate(docs, 1):
            if i in used or norm_cite(d.metadata.get("citation", "")) != cite:
                continue
            text = d.page_content
            if norm_ws(text) == gold_norm or jaccard(word_set(text), gold_words) >= JACCARD_MATCH:
                found = i
                break
        ranks.append(found)
        if found:
            used.add(found)
    return ranks


def page_ranks(docs, golds):
    """Rank of the first doc on each distinct gold page (citation only), None if absent."""
    out = {}
    for cite in dict.fromkeys(c for c, _ in golds):
        out[cite] = next((i for i, d in enumerate(docs, 1)
                          if norm_cite(d.metadata.get("citation", "")) == cite), None)
    return list(out.values())


def recall_at(ranks, k):
    return sum(1 for r in ranks if r and r <= k) / len(ranks)


def full_at(ranks, k):
    return 1 if all(r and r <= k for r in ranks) else 0


def any_at(ranks, k):
    return 1 if any(r and r <= k for r in ranks) else 0


def precision_at(ranks, k):
    return sum(1 for r in ranks if r and r <= k) / k


def ndcg_at(ranks, k):
    dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r and r <= k)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(ranks), k) + 1))
    return dcg / ideal if ideal else 0.0


def mrr_at(ranks, k):
    hits = [r for r in ranks if r and r <= k]
    return 1.0 / min(hits) if hits else 0.0


def ap_at(ranks, k):
    """Average precision over the top k (RAGAS-style context precision with gold chunks as relevant)."""
    hits = sorted(r for r in ranks if r and r <= k)
    if not hits:
        return 0.0
    return sum((j + 1) / r for j, r in enumerate(hits)) / len(ranks)


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


def judge_json(client, model, prompt, validate):
    """Call the judge, parse JSON, run validate(data) -> cleaned dict or raise ValueError. Returns (data, err)."""
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
            return validate(json.loads(m.group(0))), None
        except Exception as e:  # noqa
            err = f"unparseable verdict: {e}"
    return None, err


def judge_synthesis(client, model, question, golds, reference, candidate):
    n = len(golds)
    sources = "\n".join(f"{i}. {c}" for i, (c, _) in enumerate(golds, 1))
    prompt = SYNTHESIS_JUDGE_PROMPT.format(
        n_sources=n, question=question, sources=sources,
        reference=reference.strip(), candidate=strip_citations(candidate).strip())

    def validate(d):
        cov = d.get("covered")
        if not isinstance(cov, list) or len(cov) != n or not all(isinstance(x, bool) for x in cov):
            raise ValueError(f"'covered' must be {n} booleans")
        for key in ("accurate", "misattributed", "integrated"):
            if not isinstance(d.get(key), bool):
                raise ValueError(f"'{key}' must be a boolean")
        return {"accurate": d["accurate"], "covered": cov, "misattributed": d["misattributed"],
                "integrated": d["integrated"], "reason": str(d.get("reason", ""))}

    return judge_json(client, model, prompt, validate)


def judge_faithfulness(client, model, docs, candidate):
    context = "\n\n".join(f"[Source: {d.metadata.get('citation', '')}]\n{d.page_content}" for d in docs)
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=context, answer=candidate.strip())

    def validate(d):
        claims, supported = int(d["claims"]), int(d["supported"])
        cited, wrong = int(d.get("cited_statements", 0)), int(d.get("wrongly_cited", 0))
        if claims < 0 or not 0 <= supported <= claims or cited < 0 or not 0 <= wrong <= cited:
            raise ValueError("inconsistent counts")
        return {"claims": claims, "supported": supported, "cited_statements": cited,
                "wrongly_cited": wrong, "unsupported": [str(x) for x in (d.get("unsupported") or [])][:5]}

    return judge_json(client, model, prompt, validate)


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


def line_cite_coverage(answer):
    """Share of non-empty answer lines (sentences/bullets/Summary) that carry a [Source: ...] tag."""
    lines = [l for l in answer.splitlines() if len(re.findall(r"\w+", l)) >= 4]
    if not lines:
        return None
    return sum(1 for l in lines if CITE_RE.search(l)) / len(lines)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=None, help=f"QA json file (default tests/{QA_DEFAULT})")
    ap.add_argument("--judge-model", default=None, help=f"Gemini judge model (default: {GEMINI_MODEL})")
    ap.add_argument("--no-judge", action="store_true", help="skip all LLM judging")
    ap.add_argument("--no-faithfulness", action="store_true",
                    help="skip the faithfulness / citation-support judge call (halves judge calls)")
    ap.add_argument("--limit", type=int, default=None, help="only first N questions")
    ap.add_argument("--k-retrieve", type=int, default=None,
                    help="candidates per retriever before reranking (default: rag_pipeline.K_RETRIEVE)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="chunks kept after reranking (default: rag_pipeline.TOP_K)")
    args = ap.parse_args()

    qa_path = Path(args.qa) if args.qa else TESTS_DIR / QA_DEFAULT
    if not qa_path.exists() and (ROOT_DIR / QA_DEFAULT).exists():
        qa_path = ROOT_DIR / QA_DEFAULT
    if not qa_path.exists():
        sys.exit(f"QA file not found: {qa_path}")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa = [row for row in qa if len(load_golds(row)) >= 2]
    if not qa:
        sys.exit("No multi-source questions (2+ cited chunks) found in the QA file.")
    if args.limit:
        qa = qa[: args.limit]

    if not Path(rp.CHROMA_DIR).exists():
        sys.exit(f"Vector store not found at {rp.CHROMA_DIR}. Run the build step first.")
    k_retrieve = args.k_retrieve or rp.K_RETRIEVE
    top_k = args.top_k or rp.TOP_K
    max_gold = max(len(load_golds(r)) for r in qa)
    KS = tuple(sorted({k for k in BASE_KS if k <= top_k} | {top_k}))
    warn_topk = top_k < max_gold

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
    do_faith = judge_llm is not None and not args.no_faithfulness

    print("Loading vector store and building hybrid retriever + reranker ...")
    store = rp.load_vector_store()
    retrieve = make_pool_retriever(rp.build_retriever(store))

    results = []
    judge_off, fail_streak = False, 0
    for row in tqdm(qa, desc="Evaluating", unit="q"):
        golds = load_golds(row)
        gold_cites = list(dict.fromkeys(c for c, _ in golds))
        r = {"idx": row["question_index"], "question": row["question"], "n_gold": len(golds)}
        try:
            answer, docs, pool, lat, rlat = run_pipeline(retrieve, row["question"], k_retrieve, top_k)
        except Exception as e:  # noqa
            r.update(error=str(e), answer="", latency=None, retr_latency=None)
            results.append(r)
            continue

        # ranks are over the full reranked pool; values above top_k mean "retrieved but cut"
        ranks = match_golds(pool, golds)
        pranks = page_ranks(pool, golds)
        cited = {norm_cite(c) for c in CITE_RE.findall(answer)}
        retrieved_cites = {norm_cite(d.metadata.get("citation", "")) for d in docs}
        refused = is_refusal(answer)
        gold_cited = [c for c in gold_cites if c in cited]
        all_retrieved = all(x and x <= top_k for x in ranks)
        all_cited = len(gold_cited) == len(gold_cites)

        syn, syn_reason, faith, faith_err = None, "judge skipped", None, None
        if judge_llm is None:
            pass
        elif refused:
            syn = {"accurate": False, "covered": [False] * len(golds), "misattributed": False,
                   "integrated": False, "reason": "refusal"}
            syn_reason = "refusal"
        elif judge_off:
            syn_reason = "judge disabled after repeated failures"
        else:
            syn, syn_reason = judge_synthesis(judge_llm, judge_model, row["question"], golds, row["answer"], answer)
            fail_streak = fail_streak + 1 if syn is None else 0
            if fail_streak >= MAX_JUDGE_FAIL_STREAK:
                judge_off = True
                tqdm.write(f"Judge failed {fail_streak} times in a row ({syn_reason}). Judging disabled for the rest of the run.")
            if do_faith and syn is not None and not judge_off:
                faith, faith_err = judge_faithfulness(judge_llm, judge_model, docs, answer)

        r.update(
            answer=answer,
            reference=row["answer"],
            gold_citations=[c for c, _ in golds],
            retrieved=[d.metadata.get("citation", "") for d in docs],
            pool_size=len(pool),
            ranks=ranks,
            page_ranks=pranks,
            refused=refused,
            partial_refusal=has_partial_refusal(answer),
            n_cited=len(cited),
            n_gold_cited=len(gold_cited),
            cite_valid=len(cited & retrieved_cites),
            line_cite_cov=line_cite_coverage(answer),
            all_retrieved=all_retrieved,
            all_cited=all_cited,
            syn=syn,
            syn_reason=syn_reason,
            faith=faith,
            faith_err=faith_err,
            token_f1=token_f1(row["answer"], answer),
            rouge_l=rouge_l_f1(row["answer"], answer),
            num_recall=numeric_recall(row["answer"], answer),
            fmt_ok=format_ok(answer),
            latency=lat,
            retr_latency=rlat,
        )
        # multi-source success needs the judge: accurate, every source covered, no cross-source mix-up
        if syn is not None:
            r["success"] = bool(syn["accurate"] and all(syn["covered"]) and not syn["misattributed"])
            r["frac_covered"] = sum(syn["covered"]) / len(syn["covered"])
        else:
            r["success"], r["frac_covered"] = None, None
        # failure funnel: first stage that fails
        if not all(x for x in ranks):
            r["failure"] = "retrieval miss (a gold chunk not in candidate pool)"
        elif not all_retrieved:
            r["failure"] = "reranker cut (all gold in pool, not all in top_k)"
        elif refused:
            r["failure"] = "refusal despite all gold retrieved"
        elif not all_cited:
            r["failure"] = "synthesis gap (all gold retrieved, not all cited)"
        elif r["success"] is None:
            r["failure"] = "not judged"
        elif not r["success"]:
            r["failure"] = "answer flawed (all gold retrieved and cited, judge: inaccurate/incomplete/mixed up)"
        else:
            r["failure"] = "success"
        results.append(r)

    ok = [r for r in results if "error" not in r]
    n_err = len(results) - len(ok)
    n = len(ok)
    if n == 0:
        sys.exit("No question completed successfully.")

    L = []

    def prop(label, k, total, w=44):
        lo, hi = wilson(k, total)
        pct = 100.0 * k / total if total else 0.0
        L.append(f"{label:<{w}}{pct:6.1f}%  ({k}/{total})  95% CI [{100*lo:.1f}, {100*hi:.1f}]")

    def mean_line(label, values, w=44, digits=4):
        values = [v for v in values if v is not None]
        if not values:
            L.append(f"{label:<{w}}n/a")
            return
        lo, hi = bootstrap_ci(values)
        L.append(f"{label:<{w}}{statistics.mean(values):7.{digits}f}  (n={len(values)})  95% CI [{lo:.{digits}f}, {hi:.{digits}f}]")

    gold_counts = sorted({r["n_gold"] for r in ok})
    mix = ", ".join(f"{sum(1 for r in ok if r['n_gold'] == g)} with {g} sources" for g in gold_counts)

    L.append("MULTI-SOURCE SYNTHESIS RAG TEST RESULTS")
    L.append(f"Date                : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Chat model          : {rp.CHAT_MODEL}")
    L.append(f"Embedding model     : {rp.EMBED_MODEL}")
    judge_note = " (same as chat model, self-grading bias)" if judge_llm is not None and judge_model == rp.CHAT_MODEL else ""
    L.append(f"Judge model         : {'none' if judge_llm is None else judge_model}{judge_note}")
    L.append(f"Judge checks        : {'none' if judge_llm is None else 'synthesis vs reference' + (' + faithfulness/citation support' if do_faith else '')}")
    L.append(f"Chunk size/overlap  : {rp.CHUNK_SIZE}/{rp.CHUNK_OVERLAP}")
    L.append("Retriever           : hybrid BM25 + dense (Ensemble, weights 0.5/0.5) + cross-encoder rerank")
    L.append(f"Reranker model      : {rp.RERANK_MODEL}")
    L.append(f"k_retrieve / top_k  : {k_retrieve} per retriever / {top_k} after rerank")
    L.append(f"QA file             : {qa_path.name}")
    L.append(f"Questions           : {len(results)} (completed {n}, errors {n_err}); {mix}")
    L.append(f"Chunk match rule    : same citation and (normalized text equal or word Jaccard >= {JACCARD_MATCH}); one doc per gold chunk")
    if warn_topk:
        L.append(f"WARNING             : top_k={top_k} is smaller than the largest number of required sources ({max_gold}); full coverage is impossible for those questions.")
    L.append("")

    L.append("1. RETRIEVAL, all required chunks (exact chunk)")
    for k in KS:
        mean_line(f"Recall@{k} (macro mean)", [recall_at(r["ranks"], k) for r in ok])
    for k in KS:
        prop(f"FULL-COVERAGE@{k} (all gold in top {k})", sum(full_at(r["ranks"], k) for r in ok), n)
    for k in KS:
        prop(f"Any-hit@{k}", sum(any_at(r["ranks"], k) for r in ok), n)
    for k in KS:
        mean_line(f"Precision@{k}", [precision_at(r["ranks"], k) for r in ok])
    mean_line(f"MRR@{top_k} (first gold chunk)", [mrr_at(r["ranks"], top_k) for r in ok])
    mean_line(f"MAP@{top_k} (context precision)", [ap_at(r["ranks"], top_k) for r in ok])
    for k in KS:
        mean_line(f"nDCG@{k}", [ndcg_at(r["ranks"], k) for r in ok])
    avg_pool = statistics.mean(r["pool_size"] for r in ok)
    mean_line(f"Candidate pool recall (avg {avg_pool:.0f} cands)", [recall_at(r["ranks"], 10 ** 9) for r in ok])
    prop("Candidate pool FULL coverage", sum(full_at(r["ranks"], 10 ** 9) for r in ok), n)
    mean_line("Mean rank of worst gold chunk (found)", [max(x for x in r["ranks"] if x) for r in ok if any(r["ranks"])], digits=2)
    L.append("")

    L.append("2. RETRIEVAL, required pages (citation only)")
    for k in KS:
        mean_line(f"Page recall@{k} (macro mean)", [recall_at(r["page_ranks"], k) for r in ok])
    for k in KS:
        prop(f"Page FULL-COVERAGE@{k}", sum(full_at(r["page_ranks"], k) for r in ok), n)
    prop("Candidate pool page FULL coverage", sum(full_at(r["page_ranks"], 10 ** 9) for r in ok), n)
    L.append("")

    L.append("3. SOURCE USE IN THE ANSWER (deterministic, from [Source: ...] tags)")
    mean_line("Gold sources cited (macro mean)", [r["n_gold_cited"] / len(set(r["gold_citations"])) for r in ok])
    prop("ALL gold sources cited", sum(1 for r in ok if r["all_cited"]), n)
    prop("Cites 2+ distinct sources", sum(1 for r in ok if r["n_cited"] >= 2), n)
    both = [r for r in ok if r["all_retrieved"]]
    if both:
        prop("All cited | all gold in top_k (conversion)", sum(1 for r in both if r["all_cited"]), len(both))
    total_cited = sum(r["n_cited"] for r in ok)
    valid = sum(r["cite_valid"] for r in ok)
    if total_cited:
        prop("Citation validity (cited in retrieved set)", valid, total_cited)
        prop("Hallucinated citation rate", total_cited - valid, total_cited)
    mean_line("Lines carrying a citation (mean)", [r["line_cite_cov"] for r in ok])
    L.append("")

    judged = [r for r in ok if r["syn"] is not None]
    L.append("4. MULTI-SOURCE ANSWER QUALITY (judge vs reference)")
    if judged:
        prop("MULTI-SOURCE SUCCESS (acc.+complete+no mix)", sum(1 for r in judged if r["success"]), len(judged))
        prop("Accurate", sum(1 for r in judged if r["syn"]["accurate"]), len(judged))
        mean_line("Source completeness (mean share covered)", [r["frac_covered"] for r in judged])
        prop("Fully complete (every source covered)", sum(1 for r in judged if all(r["syn"]["covered"])), len(judged))
        prop("Integrated (one coherent answer)", sum(1 for r in judged if r["syn"]["integrated"]), len(judged))
        prop("Cross-source misattribution rate", sum(1 for r in judged if r["syn"]["misattributed"]), len(judged))
        # per-source-slot coverage shows whether the first, second or third source gets dropped
        for slot in range(max(r["n_gold"] for r in judged)):
            sub = [r for r in judged if r["n_gold"] > slot]
            prop(f"Source #{slot + 1} covered", sum(1 for r in sub if r["syn"]["covered"][slot]), len(sub))
        L.append(f"{'Judge failures (excluded)':<44}{len(ok) - len(judged):7d}")
        hit = [r for r in judged if r["all_retrieved"]]
        miss = [r for r in judged if not r["all_retrieved"]]
        if hit:
            prop("Success | all gold in top_k", sum(1 for r in hit if r["success"]), len(hit))
        if miss:
            prop("Success | some gold missing from top_k", sum(1 for r in miss if r["success"]), len(miss))
        prop("END-TO-END (all gold in top_k AND success)", sum(1 for r in judged if r["all_retrieved"] and r["success"]), len(judged))
    else:
        L.append("Judge checks skipped")
    L.append("")

    L.append("5. GROUNDING (judge vs retrieved context)")
    fj = [r for r in ok if r["faith"] is not None and r["faith"]["claims"] > 0]
    if fj:
        tot_claims = sum(r["faith"]["claims"] for r in fj)
        tot_sup = sum(r["faith"]["supported"] for r in fj)
        prop("Faithfulness (supported / claims, micro)", tot_sup, tot_claims)
        mean_line("Faithfulness (macro mean per answer)", [r["faith"]["supported"] / r["faith"]["claims"] for r in fj])
        prop("Fully faithful answers", sum(1 for r in fj if r["faith"]["supported"] == r["faith"]["claims"]), len(fj))
        cs = sum(r["faith"]["cited_statements"] for r in fj)
        wc = sum(r["faith"]["wrongly_cited"] for r in fj)
        if cs:
            prop("Citation support (tag matches its passage)", cs - wc, cs)
        L.append(f"{'Faithfulness judge failures (excluded)':<44}{sum(1 for r in ok if r['syn'] is not None and r['faith'] is None and do_faith):7d}")
    else:
        L.append("Faithfulness check skipped" if not do_faith else "No faithfulness results")
    L.append("")

    L.append("6. LEXICAL SIMILARITY TO REFERENCE")
    mean_line("Token F1", [r["token_f1"] for r in ok])
    mean_line("ROUGE-L F1", [r["rouge_l"] for r in ok])
    mean_line("Numeric recall", [r["num_recall"] for r in ok])
    L.append("")

    L.append("7. REFUSAL AND FORMAT")
    prop("Refusal rate (whole answer)", sum(1 for r in ok if r["refused"]), n)
    prop("Partial answer (names what is missing)", sum(1 for r in ok if r["partial_refusal"]), n)
    answered = [r for r in ok if not r["refused"]]  # refusals have no Summary line by design
    if answered:
        prop("Format compliance (non-refusals)", sum(1 for r in answered if r["fmt_ok"]), len(answered))
    L.append("")

    L.append("8. FAILURE FUNNEL (first failing stage per question)")
    order = ["retrieval miss (a gold chunk not in candidate pool)",
             "reranker cut (all gold in pool, not all in top_k)",
             "refusal despite all gold retrieved",
             "synthesis gap (all gold retrieved, not all cited)",
             "answer flawed (all gold retrieved and cited, judge: inaccurate/incomplete/mixed up)",
             "not judged",
             "success"]
    for name in order:
        k = sum(1 for r in ok if r["failure"] == name)
        if k or name == "success":
            prop(name[:43], k, n)
    L.append("")

    L.append("9. BY NUMBER OF REQUIRED SOURCES")
    for g in gold_counts:
        sub = [r for r in ok if r["n_gold"] == g]
        L.append(f"-- {g} required sources ({len(sub)} questions)")
        mean_line(f"Recall@{top_k}", [recall_at(r["ranks"], top_k) for r in sub])
        prop(f"FULL-COVERAGE@{top_k}", sum(full_at(r["ranks"], top_k) for r in sub), len(sub))
        prop("ALL gold sources cited", sum(1 for r in sub if r["all_cited"]), len(sub))
        js = [r for r in sub if r["syn"] is not None]
        if js:
            prop("Accurate", sum(1 for r in js if r["syn"]["accurate"]), len(js))
            prop("Fully complete", sum(1 for r in js if all(r["syn"]["covered"])), len(js))
            prop("MULTI-SOURCE SUCCESS", sum(1 for r in js if r["success"]), len(js))
    L.append("")

    lat = [r["latency"] for r in ok]
    rl = [r["retr_latency"] for r in ok if r["retr_latency"] is not None]
    L.append("10. LATENCY (seconds)")
    L.append(f"{'End-to-end mean / p50 / p95':<44}{statistics.mean(lat):.2f} / {percentile(lat, .5):.2f} / {percentile(lat, .95):.2f}")
    if rl:
        L.append(f"{'Retrieval mean / p50 / p95':<44}{statistics.mean(rl):.2f} / {percentile(rl, .5):.2f} / {percentile(rl, .95):.2f}")
    L.append("")

    L.append("PER QUESTION")
    L.append("idx  src  gold_ranks   full@top_k  cited  covered  acc  mix  faith  tokF1  result")
    for r in results:
        if "error" in r:
            L.append(f"{r['idx']:<5}ERROR: {r['error'][:80]}")
            continue
        gr = ",".join(str(x) if x else "-" for x in r["ranks"])
        s = r["syn"]
        cov = "-" if s is None else f"{sum(s['covered'])}/{len(s['covered'])}"
        acc = "-" if s is None else int(s["accurate"])
        mix = "-" if s is None else int(s["misattributed"])
        fa = "-" if r["faith"] is None or not r["faith"]["claims"] else f"{r['faith']['supported']}/{r['faith']['claims']}"
        L.append(f"{r['idx']:<5}{r['n_gold']:<5}{gr:<13}{int(r['all_retrieved']):<12}"
                 f"{r['n_gold_cited']}/{len(set(r['gold_citations'])):<5}{cov:<9}{str(acc):<5}{str(mix):<5}{fa:<7}"
                 f"{r['token_f1']:<7.3f}{r['failure'].split(' (')[0]}")

    text = "\n".join(L) + "\n"
    TESTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""  # a partial run must not overwrite a full run
    base = f"rag_test_multisource_{safe_name(rp.CHAT_MODEL)}{suffix}"
    out_path = TESTS_DIR / f"{base}_results.txt"
    out_path.write_text(text, encoding="utf-8")
    details_path = TESTS_DIR / f"{base}_details.json"
    details_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(text)
    print(f"Saved to {out_path}")
    print(f"Per-question details (answers, judge reasons, unsupported claims, ranks): {details_path}")


if __name__ == "__main__":
    main()
