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
CHAT_MODEL = "qwen3:4b"           # Chat model (adjustable)
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
    """Format retrieved chunks with source citations."""

    return "\n\n".join(
        f"Source: {doc.metadata.get('citation', 'Unknown Source')}\n"
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
    llm = ChatOllama(model=CHAT_MODEL)

    # Prompt template for RAG system
    prompt = ChatPromptTemplate.from_template(
        """
You are a financial and economic research assistant specializing in
Thailand's monetary policy and macroeconomic conditions.

Your primary knowledge source consists of the Bank of Thailand (BOT)
quarterly monetary policy reports since 2013 Q1.

Answer the user's question using ONLY the retrieved context.

INSTRUCTIONS:

1. Factual accuracy
- Do not introduce facts, statistics, or assumptions that are not
  supported by the retrieved context.
- Do not fabricate economic indicators, figures, dates, or policy decisions.
- If the retrieved context is insufficient, explicitly state what
  information is missing.
- Distinguish between reported facts, economic interpretations, and
  forward-looking assessments.

2. Financial and economic analysis
- Pay attention to monetary policy, inflation, GDP growth, interest rates,
  exchange rates, household debt, employment, consumption, investment,
  exports, and financial stability.
- Preserve the units, time periods, and economic definitions of reported
  statistics.
- When comparing economic indicators, identify the relevant periods.
- Distinguish between year-on-year (YoY) and quarter-on-quarter (QoQ)
  changes whenever applicable.
- Do not interpret correlation as causation.

3. Citation requirements
- Cite every factual claim using the provided source metadata.
- Use this citation format:
  [Source: MPR_YYYY_QN, PDF p. X]
- If multiple sources support a claim, cite each relevant source.
- Never invent page numbers or source filenames.
- If citation metadata is unavailable, explicitly state that the source
  reference could not be identified.

4. Response structure
- Start with a direct answer to the question.
- Use bullet points when explaining multiple economic factors.
- Include relevant numerical evidence where available.
- Explain economic implications only when supported by the context.
- End with a short summary of the main findings.
- Do not use generic citation wordings like "as documented in the monetary policy reports,"
  "context," "the retrieved context," or "supported by context."

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""
    )

    # Build the RAG chain
    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)


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
        print(f"\nBOT Monetary Policy RAG system ready. Model: {CHAT_MODEL}")
        print("Type your questions (Ctrl+C to exit):\n")

        while True:
            try:
                question = input("Q: ").strip()
                if not question:
                    continue
                print("\nSearching documents and generating answer...\n")
                answer = query_documents(store, question)
                print(f"A: {answer}\n")
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
