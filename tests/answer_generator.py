"""
Sample PDF chunks for RAG evaluation.

Uses the same chunker, chunk settings and citation format as rag_pipeline.py,
so sampled chunks and citations match what the vector store contains.

Usage (from the project root or from the tests/ folder):
    python tests/answer_generator.py

Optional environment variables:
    SAMPLE_SIZE    Number of chunks to sample (default: 250)
    RANDOM_SEED    Reproducible sampling seed (default: 42)

Output:
    tests/answer_sample.txt
"""

import os
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Locate the project root (the folder that contains pdf_chunking.py).
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR if (SCRIPT_DIR / "pdf_chunking.py").exists() else SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from pdf_chunking import load_and_chunk_pdfs  # noqa: E402

# Keep these identical to rag_pipeline.py
DOCS_DIR = ROOT_DIR / "documents"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250

OUTPUT_FILE = ROOT_DIR / "tests" / "answer_sample.txt"
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 250))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 42))


def load_chunks():
    """Load and chunk PDFs with the pipeline's chunker, then add citations."""
    print(f"Loading PDF pages from {DOCS_DIR}/ ...")

    chunks, n_pages = load_and_chunk_pdfs(
        DOCS_DIR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    print(f"Loaded {n_pages} PDF page(s), created {len(chunks)} chunk(s).")

    # Same citation format as build_vector_store(): page is already 1-based.
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown Source")
        page = chunk.metadata.get("page", 0)
        chunk.metadata["citation"] = f"{Path(source).stem}, PDF p. {page}"

    return chunks


def write_sample(chunks):
    """Randomly sample chunks and export them in LLM-friendly TXT format."""
    if not chunks:
        print("No chunks available. Check the PDF directory.")
        return

    sample_count = min(SAMPLE_SIZE, len(chunks))
    rng = random.Random(RANDOM_SEED)
    sampled_chunks = rng.sample(chunks, sample_count)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
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
    write_sample(load_chunks())


if __name__ == "__main__":
    main()
