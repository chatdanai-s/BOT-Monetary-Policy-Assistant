# BOT Monetary Policy Report RAG Assistant

A locally hosted retrieval-augmented generation (RAG) assistant for answering questions about the Bank of Thailand (BOT) quarterly Monetary Policy Reports (MPR), from 2013 Q1 onward.
Answers are generated only from retrieved report text and include citations to the source report and PDF page.

The assistant runs locally through Ollama and HuggingFace with zero API costs. The RAG evaluation script uses the Gemini API (free tier) as the answer judge.

## Key Results

<sub>Evaluation run: 11 November 2026 (185 questions, 0 evaluation errors). LLM judge: `gemini-3.5-flash-lite` against reference answers.</sub>

| Category | Metric | Result |
|---|---|---:|
| Retrieval | Recall@1/Recall@5 | **62.2%**/**92.4%** |
| Retrieval | MRR@5 | **0.7387** |
| Retrieval | nDCG@5 | **0.7852** |
| Answer quality | Answer correctness (LLM judge: Gemini 3.5 Flash-Lite) | **90.3%** |
| Answer quality | End-to-end success (gold chunk retrieved & answer judged correct) | **86.5%** |
| Citations | Citation precision (within retrieved set) | **100.0%** |
| Citations | Gold citation present in answer | **75.1%** |
| Performance | End-to-end latency, Mean/P95 | **17.73 s**/**26.77 s** |

## Overview

1. **Download reports:** `scrape_BOT_MPR.py` downloads the MPR PDFs from the BOT website into `documents/`.
2. **PDF Loading and Chunking:** `PyPDFLoader` loads the PDF pages in `documents/`, and splits content into overlapping chunks with with `PyMuPDF` and RecursiveCharacterTextSplitter.
3. **Citation Metadata:** Each chunk retains its source filename and PDF page for citation (`<filename>, PDF p. <page>`).
4. **Embedding:** `qwen3-embedding:0.6b` embeds the chunks and stores them in ChromaDB.
5. **Retrieval:** Hybrid retrieval combines BM25 and dense-vector search, with equal weights (0.5/0.5). Each retriever returns up to 30 candidates.
6. **Reranking:** `BAAI/bge-reranker-v2-m3` scores the candidate passages against the question; the top 5 passages are passed to the generator.
7. **Generation:** `qwen3:4b-instruct` generates an answer using the retrieved context and citation instructions.
8. **Evaluation (optional):** `tests/rag_test.py` runs a question set through the pipeline and scores retrieval, answers, citations and format. Answer correctness is graded by `gemini-3.5-flash-lite` via the Gemini API free tier.

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Interface** | Terminal (question loop) |
| **RAG Framework** | LangChain |
| **PDF Loading** | `PyPDFLoader` and `PyMuPDF` |
| **Text Chunking** | RecursiveCharacterTextSplitter (1,500 characters, 250 overlap) |
| **Embedding Model** | `qwen3-embedding:0.6b` (Ollama) |
| **Vector Database** | ChromaDB (collection: `bot_mpr_documents`) |
| **Retriever** | Hybrid BM25 + dense retrieval (equal weights) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` |
| **LLM** | Ollama + `qwen3:4b-instruct` |
| **Evaluation Judge** | `gemini-3.5-flash-lite` (free tier API) |
| **Data Source** | BOT Monetary Policy Reports (PDF) |
| **Programming Language** | Python |

## File Structure

```text
.
├── documents/            Downloaded MPR PDFs (created by the scraper)
├── chroma_db/            Persisted ChromaDB vector store
├── tests/                Evaluation script (rag_test.py) and QA set (qa_pairs.json)
├── scrape_BOT_MPR.py     Downloads MPR PDFs from the BOT website
├── build_vector_db.bat   Builds the vector store (runs rag_pipeline.py --build)
├── rag_pipeline.py       Vector store building and question answering
├── README.md
├── .env                  GEMINI_API_KEY, used only by the evaluation script
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
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:4b-instruct
```

2. **Install the Python packages**

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
          (1500 chars, 250 overlap)
     Citation metadata added to each chunk
                      │
                      ▼
         qwen3-embedding:0.6b (Ollama)
                      │
                      ▼
           ChromaDB Vector Database
                      │
                      ▼
        Hybrid BM25 + Dense Retrieval
        (30 candidates per retriever)
                      │
                      ▼
        BGE Cross-Encoder Reranking
             (top 5 passages)
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

## Limitations

### Hardware Constraints

This project is run on:

- **CPU:** Intel Core i5-12400F
- **Memory:** 16 GB RAM
- **GPU:** NVIDIA GeForce RTX 3050 (6 GB VRAM)

The available RAM and VRAM constrains the selection of local models.

### Current Limitations

- Text is extracted with PyMuPdf; scanned PDFs without a text layer are not OCR'd and may produce poor results.
- Fixed chunk size (1,500 characters) and overlap (250 characters); chunking is not adaptive.
- No query rewriting.
- Each question is answered independently; there is no conversation memory.
- Local inference on a small model is slower and less accurate than cloud-hosted LLMs.
- Like all RAG systems, incorrect or unsupported answers remain possible if relevant context is not retrieved.
- Answer correctness comes from an LLM judge (Gemini 3.5 Flash-Lite), which can disagree with a human grader.

## Data Source

[Bank of Thailand — Monetary Policy Report: Previous Reports](https://www.bot.or.th/en/our-roles/monetary-policy/mpc-publication/Monetary-Policy-Report/mpr-previous-reports.html)

The reports are published by the Bank of Thailand. This project is an independent tool for searching and querying those public documents.