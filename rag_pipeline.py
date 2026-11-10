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


def build_retriever(vector_store):
    """Build BM25 + dense + reranker once. Returns retrieve(question, k_retrieve, top_k)."""
    # Setup BM25 retrieval
    data = vector_store.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(data["documents"], data["metadatas"])
    ]
    def financial_tokenizer(text):
        text = text.lower()

        return re.findall(
            r"[a-z]+\d+[a-z0-9]*"
            r"|\d+[a-z]+\d*[a-z0-9]*"
            r"|[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?"
            r"|[a-z]+(?:[&'/.-][a-z0-9]+)*",
            text
        )

    reranker_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Begin hybrid retrieval
    bm25 = BM25Retriever.from_documents(
        docs,
        k=K_RETRIEVE,
        preprocess_func=financial_tokenizer
    )
    dense = vector_store.as_retriever(search_kwargs={"k": K_RETRIEVE})
    hybrid = EnsembleRetriever(retrievers=[bm25, dense],
                               weights=[0.5, 0.5])
    reranker = CrossEncoder(
        RERANK_MODEL,
        max_length=512,
        device=reranker_device,
        model_kwargs=(
            {"torch_dtype": torch.float16}
            if reranker_device == "cuda" else {}
        )
    )

    def retrieve(question, k_retrieve=K_RETRIEVE, top_k=TOP_K):
        bm25.k = k_retrieve
        dense.search_kwargs["k"] = k_retrieve
        candidates = hybrid.invoke(question)
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
        temperature=0.1,
        think=False,
        keep_alive="30m",
        num_ctx=8192,
    )

    # Prompt template for RAG system
    prompt = ChatPromptTemplate.from_template(
            """
You are a financial and research assistant specializing in Thailand's
monetary policy and macroeconomy.

Your ONLY knowledge source is the Bank of Thailand (BOT) quarterly
Monetary Policy Reports (MPR) from 2013 Q1 through today's date: {today}.

Answer the user's question using ONLY the retrieved context.

RULES
1. Use ONLY information explicitly supported by CONTEXT.
   Never introduce facts, statistics, assumptions, figures, or dates
   from your own knowledge.

2. If CONTEXT does not contain sufficient information to answer,
   reply exactly:
   "The provided reports do not contain this information."
   Then briefly state what information is missing. Do not guess.

3. Every sentence containing factual claims or numerical information
   must end with a citation.
   Copy the [Source: ...] tag from CONTEXT exactly, character for character.
   If multiple reports support a claim, cite all relevant sources.
   Never fabricate report names or page numbers.

   Example:
   Headline inflation averaged 1.2% in 2024 [Source: MPR_2025_Q1, PDF p. 14].

4. Include relevant numerical evidence whenever available.
   Preserve numerical values, units, and reporting periods exactly
   as stated in the reports.
   Distinguish year-on-year (YoY) from quarter-on-quarter (QoQ)
   whenever specified.

5. Identify reports using their MPR_YYYY_QN naming convention.
   Distinguish historical forecasts from actual outcomes.
   Do not present forecasts from past reports as current projections.
   When reports contain conflicting information, prioritize the
   most recent relevant report and explicitly identify its period.

6. Do not explain causes, relationships, or economic implications
   unless explicitly supported by CONTEXT.

7. When comparing periods, ensure that the compared figures use
   compatible units, definitions, and reporting periods.
   Do not calculate or derive additional figures unless the question
   explicitly requires it and the necessary data is available.

FORMAT
- Begin with a direct answer in the first sentence.
- Do not begin with a bullet point.
- Follow with bullet points containing supporting evidence.
- Include numerical values and reporting periods where available.
- End with "Summary:" followed by one concise sentence.
- Do not mention context or retrieved text in the answer.
- Use this exact citation format:
  [Source: MPR_YYYY_QN, PDF p. X]

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
