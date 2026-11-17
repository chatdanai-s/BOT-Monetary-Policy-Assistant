import time
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import rag_pipeline as rag

state = {}
lock = threading.Lock()   # BM25 cache is shared state, so retrieval runs one at a time


@asynccontextmanager
async def lifespan(app):
    store = rag.load_vector_store()
    state["retrieve"] = rag.build_retriever(store)
    yield
    state.clear()


app = FastAPI(title="BOT MPR RAG", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=rag.TOP_K, ge=1, le=20)


class Source(BaseModel):
    citation: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int


def _retrieve(req):
    with lock:
        return state["retrieve"](req.question, rag.K_RETRIEVE, req.top_k)


@app.get("/health")
def health():
    return {"status": "ok", "ready": "retrieve" in state}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.perf_counter()
    docs = _retrieve(req)
    try:
        answer = rag.generate_answer(req.question, docs)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM backend unavailable: {e}")
    return AskResponse(
        answer=answer,
        sources=[Source(citation=d.metadata["citation"], text=d.page_content) for d in docs],
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    docs = _retrieve(req)
    return StreamingResponse(rag.stream_answer(req.question, docs), media_type="text/plain")