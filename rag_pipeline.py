import warnings
warnings.filterwarnings("ignore", module="langchain_community")

# --- Configuration ---
DOCS_DIR = "./documents"          # Put your PDFs and files here
CHROMA_DIR = "./chroma_db"        # Where vectors are saved
EMBED_MODEL = "nomic-embed-text"  # Embedding model (adjustable)
CHAT_MODEL = "qwen3:4b"           # Chat model (adjustable)
CHUNK_SIZE = 1000                 # Characters per chunk
CHUNK_OVERLAP = 200               # Overlap between chunks


from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_store():
    """Load documents, chunk them, embed them, and store them in ChromaDB."""

    print(f"Loading documents from {DOCS_DIR}/ ...")

    # Load PDFs, Markdown and Text files
    pdf_loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    md_loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    txt_loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents = pdf_loader.load() + md_loader.load() + txt_loader.load()
    print(f"Loaded {len(documents)} document(s).")

    if not documents:
        print("No documents found. Please add files to the documents/ directory.")
        return None


    # Chunk the documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s).")


    # Embed and store
    print("Embedding chunks (this may take a few minutes on the first run)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBED_MODEL),
        collection_name="my_documents",
        persist_directory=CHROMA_DIR,
    )
    print("Vector store built and stored in ChromaDB.")
    return vector_store


def load_vector_store():
    """Load an existing vector store from disk."""
    return Chroma(
        collection_name="my_documents",
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=CHROMA_DIR,
    )


def format_docs(docs):
    """Join the retrieved chunks into one block of plain text."""
    return "\n\n".join(doc.page_content for doc in docs)


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

def query_documents(vector_store, question):
    """Run a RAG query against the vector store."""

    # Find the 5 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Initialize LLM
    llm = ChatOllama(model=CHAT_MODEL)

    # Prompt template for RAG
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the question based ONLY on "
        "the following context. If the context doesn't contain enough "
        "information to answer fully, say what you can and note what's "
        "missing.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    # Build the RAG chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)


# UX Part
import os
import sys

if __name__ == "__main__":
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created {DOCS_DIR}/ directory. Please add your documents there.")
        print("Then run build_vector_db.bat")
        sys.exit(0)

    if "--build" in sys.argv:
        build_vector_store()
    else:
        # Load the existing store and start a question loop
        store = load_vector_store()
        print(f"\nRAG system ready. Using {CHAT_MODEL} for answers.")
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
