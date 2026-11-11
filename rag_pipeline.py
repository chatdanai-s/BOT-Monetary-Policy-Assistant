# General imports
from pathlib import Path
import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)
from tqdm import tqdm
import hashlib
import os
import sys
from datetime import datetime
import torch 

# Directory loading and vector storage
from pdf_chunking import load_and_chunk_pdfs
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Hybrid retrieval
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import re
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

# Chat imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama


# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_DIR = str(SCRIPT_DIR / "documents")          # Put your PDFs and files here
CHROMA_DIR = str(SCRIPT_DIR / "chroma_db")        # Where vectors are saved
EMBED_MODEL = "qwen3-embedding:0.6b"        # Embedding model (adjustable)
CHAT_MODEL = "qwen3:4b-instruct"            # Chat model (adjustable)
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"    # Reranking model (adjustable)

K_RETRIEVE = 30         # candidates fetched by EACH retriever (BM25 and dense)
TOP_K = 5               # chunks kept after reranking
CHUNK_SIZE = 1500       # Characters per chunk
CHUNK_OVERLAP = 250     # Overlap between chunks
COLLECTION_NAME = "bot_mpr_documents"


def build_vector_store():
    """Load documents, chunk them, embed them, and store them in ChromaDB."""

    print(f"Loading PDF pages from {DOCS_DIR}/ ...")

    chunks, n_pages = load_and_chunk_pdfs(DOCS_DIR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Loaded {n_pages} PDF page(s), created {len(chunks)} chunk(s).")

    if not chunks:
        print("No PDF pages found. Please add PDF files to the documents/ directory.")
        return None


    # Add citation metadata to each chunk
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown Source")
        page = chunk.metadata.get("page", 0)

        filename = Path(source).stem

        chunk.metadata["citation"] = (
            f"{filename}, PDF p. {page}"
        )


    # Embed and store vectors in ChromaDB
    print("Embedding chunks...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=CHROMA_DIR,
    )
    vector_store.delete_collection()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=CHROMA_DIR,
    )
    BATCH_SIZE = 32

    for i in tqdm(
        range(0, len(chunks), BATCH_SIZE),
        desc="Embedding chunks",
        unit="batch"
    ):
        batch = chunks[i:i + BATCH_SIZE]
        ids = [hashlib.sha256(
            f"{c.metadata['file']}_{c.metadata['page']}_{c.metadata['chunk_in_page']}".encode()
        ).hexdigest() for c in batch]
        vector_store.add_documents(
            documents=batch,
            ids=ids
        )
        
    print("Vector store built and stored in ChromaDB.")
    return vector_store


def load_vector_store():
    """Load an existing vector store from disk."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=CHROMA_DIR,
    )


MONTH_Q = {"march": 1, "june": 2, "sept": 3, "september": 3, "december": 4}
_REPORT = r"(?:mpr|monetary\s+policy\s+report|report)\b"
RID_PATTERNS = [
    (re.compile(rf"\bMPR[_\s-]?(20\d\d)[_\s-]?Q([1-4])\b", re.I), lambda m: (m[1], m[2])),
    (re.compile(rf"\b(20\d\d)[\s_/-]*Q([1-4])\s+{_REPORT}", re.I), lambda m: (m[1], m[2])),
    (re.compile(rf"\bQ([1-4])[\s_/-]*(20\d\d)\s+{_REPORT}", re.I), lambda m: (m[2], m[1])),
    (re.compile(rf"\b(march|june|sept(?:ember)?|december)\s+(20\d\d)\s+{_REPORT}", re.I),
     lambda m: (m[2], str(MONTH_Q[m[1].lower()]))),
]

def parse_report_files(question):
    """Return report PDF names the question explicitly names, e.g. {'MPR_2023_Q2.pdf'}."""
    files = set()
    for pat, get in RID_PATTERNS:
        for m in pat.finditer(question):
            y, q = get(m)
            files.add(f"MPR_{y}_Q{q}.pdf")
    return files


def build_retriever(vector_store):
    """Build BM25 + dense + reranker once. Returns retrieve(question, k_retrieve, top_k)."""

    data = vector_store.get(include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=m)
            for t, m in zip(data["documents"], data["metadatas"])]
    all_files = {d.metadata["file"] for d in docs}

    def financial_tokenizer(text):
        text = text.lower()
        return re.findall(
            r"[a-z]+\d+[a-z0-9]*"
            r"|\d+[a-z]+\d*[a-z0-9]*"
            r"|[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?"
            r"|[a-z]+(?:[&'/.-][a-z0-9]+)*",
            text,
        )

    reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(
        RERANK_MODEL,
        max_length=768,
        device=reranker_device,
        model_kwargs=({"torch_dtype": torch.float16} if reranker_device == "cuda" else {}),
    )

    bm25_cache = {}

    def get_bm25(files, k):
        key = tuple(sorted(files)) if files else ()
        if key not in bm25_cache:
            sub = [d for d in docs if not files or d.metadata["file"] in files]
            bm25_cache[key] = BM25Retriever.from_documents(
                sub, k=k, preprocess_func=financial_tokenizer)
        bm25_cache[key].k = k
        return bm25_cache[key]

    def hybrid_candidates(question, k, files):
        kwargs = {"k": k}
        if files:
            kwargs["filter"] = {"file": {"$in": sorted(files)}}
        dense = vector_store.as_retriever(search_kwargs=kwargs)
        hybrid = EnsembleRetriever(retrievers=[get_bm25(files, k), dense],
                                   weights=[0.5, 0.5])
        return hybrid.invoke(question)

    def retrieve(question, k_retrieve=K_RETRIEVE, top_k=TOP_K):
        files = parse_report_files(question) & all_files
        candidates = hybrid_candidates(question, k_retrieve, files) if files else []
        if len(candidates) < 3:                       # no report named, or filter too thin
            candidates = hybrid_candidates(question, k_retrieve, set())
        if not candidates:
            return []
        scores = reranker.predict([(question, d.page_content) for d in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]]

    return retrieve


def format_docs(docs):
    """Format retrieved chunks with citations in the exact output format."""
    return "\n\n".join(
        f"[Source: {doc.metadata.get('citation', 'Unknown Source')}]\n"
        f"Content:\n{doc.page_content}"
        for doc in docs
    )


def query_documents(retrieve, question, k_retrieve=K_RETRIEVE, top_k=TOP_K):
    """Run a citation-grounded RAG query against BOT Monetary Policy Reports."""

    # Initialize LLM
    llm = ChatOllama(
        model=CHAT_MODEL,
        temperature=0,
        think=False,
        keep_alive="30m",
        num_ctx=8192,
    )

    # Prompt template for RAG system
    prompt = ChatPromptTemplate.from_template(
            """
You are an expert financial and research assistant specializing in Thailand's
monetary policy and macroeconomy.

You answer questions about Thailand's monetary policy using ONLY the
Bank of Thailand Monetary Policy Reports (MPR) in CONTEXT. Today is {today}.

RULES
1. Use only facts written in CONTEXT. No outside knowledge, no guessing, no causes or implications unless CONTEXT states them.
2. If CONTEXT does not answer the question, say "The provided reports do not contain this information." and briefly name what is missing.
   If it answers only part, give that part and name what is missing.
3. Use the report the question names if specified. "2023 Q2" means MPR_2023_Q2. Ignore other reports unless the question asks for them.
4. Copy numbers, units, periods and direction words exactly (YoY vs QoQ, revised up, revised down, unchanged).
   For a revision, give the previous and the new value. Do not calculate new figures.
5. Separate forecasts from actual outcomes.
6. End every sentence or bullet that states a fact with its source, copied exactly from the CONTEXT tag, like [Source: MPR_2025_Q1, PDF p. 14].

FORMAT
- First sentence: the direct answer.
- Then add detail in whatever layout is easiest for a person to read: short bullets for several figures or points, a short paragraph for one explanation.
- Keep it short. Do not repeat yourself. Do not mention "context".
- Last line: "Summary:" followed by one short sentence.

CONTEXT
{context}

QUESTION
{question}

ANSWER
"""
    )

    # Build the RAG chain
    chain = (
        {"context": RunnableLambda(lambda q: retrieve(q, k_retrieve, top_k)) | format_docs,
         "question": RunnablePassthrough(),
         "today": lambda _: datetime.now().strftime('%Y %B %d')
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Stream output tokens to the console
    for token in chain.stream(question):
        print(token, end="", flush=True)
    print()


# UX Part
if __name__ == "__main__":
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created {DOCS_DIR}/ directory.")
        print("Please run scrape_BOT_MPR.py, then build_vector_db.bat")
        sys.exit(0)

    if "--build" in sys.argv:
        build_vector_store()
    
    else:
        # Load the existing store and start a question loop
        store = load_vector_store()
        retriever = build_retriever(store)

        print(f"\nBOT Monetary Policy RAG system ready.")
        print(f"Model: {CHAT_MODEL}.\nEmbedding: {EMBED_MODEL}.\nReranker: {RERANK_MODEL}.")
        print("Type your questions (Ctrl+C to exit):\n")

        while True:
            try:
                question = input("Q: ").strip()
                if not question:
                    continue
                print("\nSearching documents and generating answer...\n")

                print("A: ", end="")
                query_documents(retriever, question)
                print()
            
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
