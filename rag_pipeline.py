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

# Directory loading and vector storage
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

K_RETRIEVE = 20         # candidates fetched by EACH retriever (BM25 and dense)
TOP_K = 8               # chunks kept after reranking
CHUNK_SIZE = 1000       # Characters per chunk
CHUNK_OVERLAP = 200     # Overlap between chunks
COLLECTION_NAME = "bot_mpr_documents"


def build_vector_store():
    """Load documents, chunk them, embed them, and store them in ChromaDB."""

    print(f"Loading PDF pages from {DOCS_DIR}/ ...")

    # Load PDF files
    pdf_loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = pdf_loader.load()
    print(f"Loaded {len(documents)} PDF page(s).")

    if not documents:
        print("No PDF pages found. Please add PDF files to the documents/ directory.")
        return None


    # Chunk the documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} PDF page(s) into {len(chunks)} chunk(s).")


    # Add citation metadata to each chunk
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown Source")
        page = chunk.metadata.get("page", 0)

        filename = Path(source).stem

        chunk.metadata["citation"] = (
            f"{filename}, PDF p. {page + 1}"
        )


    # Embed and store vectors in ChromaDB
    print("Embedding chunks...")

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
        ids = [
            hashlib.sha256(
                f"{chunk.metadata['source']}_{chunk.metadata['page']}_{chunk.metadata.get('chunk_index', i+j)}".encode()
            ).hexdigest()
            for j, chunk in enumerate(batch)
        ]
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
    data = vector_store.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(data["documents"], data["metadatas"])
    ]

    bm25 = BM25Retriever.from_documents(
        docs,
        k=K_RETRIEVE,
        preprocess_func=lambda t: re.findall(r"\w+", t.lower()),
    )
    dense = vector_store.as_retriever(search_kwargs={"k": K_RETRIEVE})
    hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])
    reranker = CrossEncoder(RERANK_MODEL, max_length=512)

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
    """Format retrieved chunks with source citations, sorted chronologically."""

    return "\n\n".join(
        f"Source: [{doc.metadata.get('citation', 'Unknown Source')}]\n"
        f"Content:\n{doc.page_content}"
        for doc in docs
    )


def query_documents(retrieve, question, k_retrieve=K_RETRIEVE, top_k=TOP_K):
    """Run a citation-grounded RAG query against BOT Monetary Policy Reports."""

    # Initialize LLM
    llm = ChatOllama(
        model=CHAT_MODEL,
        temperature=0.1,
        num_ctx=8192,
        num_predict=2048
    )

    # Prompt template for RAG system
    prompt = ChatPromptTemplate.from_template(
        """
You are a financial and research assistant for Thailand's monetary policy and
macroeconomy. Your ONLY knowledge source is the Bank of Thailand (BOT) quarterly
monetary policy reports (MPR) from 2013 Q1 onward to today's date of {today}.

Answer the user's question using ONLY the retrieved context.

RULES
1. Use ONLY the text under CONTEXT. Never add facts, statistics, assumptions,
   figures, or dates from your own knowledge.
2. If the context does not answer the question, reply exactly:
   "The provided reports do not contain this information."
   Then state briefly what is missing. Do not guess.
3. Every sentence containing a fact or number must always end with a citation.
   Never invent a citation. If multiple sources support a claim, cite each relevant source.
   Example: Headline inflation averaged 1.2% in 2024 [Source: MPR_2025_Q1, PDF p. 14].
4. Include relevant numerical evidence where available.
   Keep units and periods exactly as reported. Always state whether a change
   is year-on-year (YoY) or quarter-on-quarter (QoQ) when the source says so.
5. Each report is dated by its name (MPR_YYYY_QN). Compare that date with
   today's date: forecasts from past periods are historical, not current.
   When reports conflict, prefer the most recent one and state its period.
6. Do not explain causes or implications unless the context states them.

FORMAT
- First sentence: the direct answer. Do not start with a bullet.
- Then bullets for supporting points, each with a number and period if available.
- Last line: "Summary:" followed by one sentence.
- Maximum 500 words. Do not mention "context" or "retrieved text" in the answer.
- Citation format: [Source: MPR_YYYY_QN, PDF p. X]

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
         "today": lambda _: datetime.now().strftime('%Y-%B-%d')
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
