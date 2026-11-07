"""
Sample PDF chunks for RAG evaluation.

Usage:
    python test/answer_generator.py

Optional environment variables:
    DOCS_DIR       PDF directory (default: ./documents)
    CHUNK_SIZE     Character chunk size (default: 1000)
    CHUNK_OVERLAP  Character overlap (default: 200)
    SAMPLE_SIZE    Number of chunks to sample (default: 100)
    RANDOM_SEED    Reproducible sampling seed (default: 42)

Output:
    test/answer_sample.txt
"""

import random
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize
DOCS_DIR = Path("./documents")
OUTPUT_FILE = Path("./tests/answer_sample.txt")

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
SAMPLE_SIZE = 100
RANDOM_SEED = 42


def load_chunks():
    """Load PDF pages and split them using the configured chunk settings."""
    print(f"Loading PDF pages from {DOCS_DIR}/ ...")

    pdf_loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = pdf_loader.load()

    print(f"Loaded {len(documents)} PDF page(s).")

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    print(f"Split {len(documents)} PDF page(s) into {len(chunks)} chunk(s).")

    # Add citation metadata, matching the vector-store construction script.
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown Source")
        page = chunk.metadata.get("page", 0)
        filename = Path(source).stem

        chunk.metadata["citation"] = f"{filename}, PDF p. {page + 1}"

    return chunks


def write_sample(chunks):
    """Randomly sample chunks and export them in LLM-friendly TXT format."""
    if not chunks:
        print("No chunks available. Check the PDF directory.")
        return

    sample_count = min(SAMPLE_SIZE, len(chunks))
    rng = random.Random(RANDOM_SEED)
    sampled_chunks = rng.sample(chunks, sample_count)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk in sampled_chunks:
            citation = chunk.metadata.get("citation", "Unknown citation")
            
            file.write(f"CITATION: {citation}\n")
            file.write("CONTENT:\n")
            file.write(chunk.page_content.strip())
            file.write("\n\n")
            file.write("=" * 50 + "\n\n")

    print(f"Saved {sample_count} sampled chunks to {OUTPUT_FILE}")

def main():
    chunks = load_chunks()
    write_sample(chunks)


if __name__ == "__main__":
    main()

