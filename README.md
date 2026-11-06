# BOT Monetary Policy Report RAG Chatbot

A local retrieval-augmented generation (RAG) system for answering questions about the Bank of Thailand (BOT) quarterly Monetary Policy Reports (MPR), from 2013 Q1 onward.
(https://www.bot.or.th/en/our-roles/monetary-policy/mpc-publication/Monetary-Policy-Report/mpr-previous-reports.html)
Answers are generated only from retrieved report text and include citations to the source report and PDF page.

The system runs entirely locally through Ollama, with zero API costs.

## Overview

1. **Download:** `scrape_BOT_MPR.py` downloads the MPR PDFs from the BOT website into `documents/`
2. **PDF Loading:** Loads every PDF page in `documents/` with PyPDFLoader
3. **Chunking:** Splits pages into overlapping chunks with RecursiveCharacterTextSplitter
4. **Citation Metadata:** Attaches a citation (`<filename>, PDF p. <page>`) to every chunk
5. **Embedding:** Embeds chunks with `nomic-embed-text` through Ollama and stores them in ChromaDB
6. **Retrieval:** Finds relevant chunks via Maximum Marginal Relevance (MMR) search
7. **Generation:** Generates a citation-grounded answer with `qwen3:4b-instruct` through Ollama

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Interface** | Terminal (question loop) |
| **RAG Framework** | LangChain |
| **PDF Loading** | PyPDFLoader via DirectoryLoader |
| **Text Chunking** | RecursiveCharacterTextSplitter (1000 characters, 200 overlap) |
| **Embedding Model** | nomic-embed-text (Ollama) |
| **Vector Database** | ChromaDB (collection: `bot_mpr_documents`) |
| **Retriever** | MMR search (k = 8, fetch_k = 30, lambda_mult = 0.7) |
| **LLM** | Ollama + qwen3:4b-instruct |
| **Data Source** | BOT Monetary Policy Reports (PDF) |
| **Programming Language** | Python |

## File Structure

```text
.
├── documents/            Downloaded MPR PDFs (created by the scraper)
├── chroma_db/            Persisted ChromaDB vector store
├── tests/
├── scrape_BOT_MPR.py     Downloads MPR PDFs from the BOT website
├── build_vector_db.bat   Builds the vector store (runs rag_pipeline.py --build)
├── rag_pipeline.py       Vector store building and question answering
├── README.md
├── .env
├── .gitattributes
└── .gitignore
```

## Setup

### Prerequisites
- Python
- pip (Python package manager)
- Ollama (download from [ollama.com](https://ollama.com/download))

### Installation

1. **Pull the Ollama models**

```text
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct
```

2. **Install the Python packages**

Packages imported by the scripts:

```text
pip install langchain-community langchain-text-splitters langchain-core langchain-ollama langchain-chroma pypdf tqdm requests
```

3. **Download the reports**

```text
python scrape_BOT_MPR.py
```

This downloads `MPR_<year>_Q<quarter>.pdf` files into `documents/`, starting from 2013 Q1.
Files that already exist are skipped. The script stops when it reaches a file that is not found (HTTP 404), which marks the latest available report.

4. **Build the vector store**

Double-click `build_vector_db.bat`, or run:

```text
python rag_pipeline.py --build
```

5. **Run!**

```text
python rag_pipeline.py
```

## Usage

1. Run `python rag_pipeline.py`
2. Wait for the "BOT Monetary Policy RAG system ready" message
3. Type a question at the `Q:` prompt
4. Receive an answer with citations in the format `[Source: MPR_YYYY_QN, PDF p. X]`
5. Press Ctrl+C to exit

## Architecture

```text
   scrape_BOT_MPR.py (BOT website → documents/*.pdf)
                      │
                      ▼
          PyPDFLoader (PDF → pages)
                      │
                      ▼
       RecursiveCharacterTextSplitter
          (1000 chars, 200 overlap)
    Citation metadata added to each chunk
                      │
                      ▼
       nomic-embed-text Embeddings (Ollama)
                      │
                      ▼
           ChromaDB Vector Database
                      │
                      ▼
      MMR Retrieval (k = 8, fetch_k = 30)
                      │
                      ▼
   Prompt = Retrieved Context + User Question
                      │
                      ▼
          Ollama (qwen3:4b-instruct)
                      │
                      ▼
        Cited Response in the Terminal
```

## Prompt Behavior

The prompt instructs the model to:

- Answer using only the retrieved context
- Not fabricate statistics, dates, or policy decisions
- State what information is missing when the context is insufficient
- Preserve units and time periods, and distinguish year-on-year (YoY) from quarter-on-quarter (QoQ) changes
- Cite every factual claim as `[Source: MPR_YYYY_QN, PDF p. X]`
- Start with a direct answer and end with a short summary

## Configuration

Settings are defined at the top of `rag_pipeline.py`:

| Setting | Value |
|---------|-------|
| `DOCS_DIR` | `documents/` |
| `CHROMA_DIR` | `chroma_db/` |
| `EMBED_MODEL` | `nomic-embed-text` |
| `CHAT_MODEL` | `qwen3:4b` |
| `CHUNK_SIZE` | 1000 |
| `CHUNK_OVERLAP` | 200 |
| `COLLECTION_NAME` | `bot_mpr_documents` |

Chunk IDs are generated from the source file, page, and chunk position. If `CHUNK_SIZE` or `CHUNK_OVERLAP` is changed, delete `chroma_db/` and rebuild.

## Limitations

### Hardware Constraints

This project is run on:

- **CPU:** Intel Core i5-12400F
- **Memory:** 16 GB RAM
- **GPU:** NVIDIA GeForce RTX 3050 (6 GB VRAM)

The available VRAM constrains the selection of local models.

### Current Limitations

- Only PDF files in `documents/` are loaded.
- Text is extracted with PyPDFLoader; scanned PDFs without a text layer are not OCR'd and may produce poor results.
- Retrieval uses MMR search without reranking.
- Fixed chunk size (1000 characters) and overlap (200 characters); chunking is not adaptive.
- No query rewriting or multi-stage retrieval.
- Each question is answered independently; there is no conversation memory.
- Local inference on a small model is slower and less accurate than cloud-hosted LLMs.
- Like all RAG systems, incorrect or unsupported answers remain possible if relevant context is not retrieved.
