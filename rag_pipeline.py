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

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama


# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_DIR = str(SCRIPT_DIR / "documents")          # Put your PDFs and files here
CHROMA_DIR = str(SCRIPT_DIR / "chroma_db")        # Where vectors are saved
EMBED_MODEL = "nomic-embed-text"  # Embedding model (adjustable)
CHAT_MODEL = "qwen3:4b-instruct"  # Chat model (adjustable)
CHUNK_SIZE = 1000                 # Characters per chunk
CHUNK_OVERLAP = 200               # Overlap between chunks
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


def format_docs(docs):
    """Format retrieved chunks with source citations, sorted chronologically."""

    docs = sorted(docs, key=lambda d: d.metadata.get("citation", ""))

    return "\n\n".join(
        f"Source: [{doc.metadata.get('citation', 'Unknown Source')}]\n"
        f"Content:\n{doc.page_content}"
        for doc in docs
    )


def query_documents(vector_store, question):
    """Run a citation-grounded RAG query against BOT Monetary Policy Reports."""

    # Find the 5 most relevant chunks
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 30,
            "lambda_mult": 0.7
        }
    )

    # Initialize LLM
    llm = ChatOllama(
        model=CHAT_MODEL,
        temperature=0.2,
    )

    # Prompt template for RAG system
    prompt = ChatPromptTemplate.from_template(
        """
You are a financial and research assistant for Thailand's monetary policy and
macroeconomy. Your ONLY knowledge source is the Bank of Thailand (BOT) quarterly
monetary policy reports (MPR), 2013 Q1 onward.

Answer the user's question using ONLY the retrieved context.

Prompts may include questions about the future, which you must answer based
on the most recent MPR available, closest to today's date of {today}.

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
        {"context": retriever | format_docs,
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
        print(f"\nBOT Monetary Policy RAG system ready.")
        print(f"Model: {CHAT_MODEL}. Embedding: {EMBED_MODEL}.")
        print("Type your questions (Ctrl+C to exit):\n")

        while True:
            try:
                question = input("Q: ").strip()
                if not question:
                    continue
                print("\nSearching documents and generating answer...\n")

                print("A: ", end="")
                query_documents(store, question)
                print()
            
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
