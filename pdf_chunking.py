"""Page-aware chunking for slide-style PDFs (e.g. PowerPoint exports).

Each PDF page is treated as one semantic unit. A page is only split further
if it exceeds CHUNK_SIZE. Every chunk is prefixed with a context header
(file, section, page) so it stays meaningful when retrieved on its own.
"""
import re
from pathlib import Path

import pymupdf
pymupdf.no_recommend_layout()
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PAGE_NO_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")             # "15/56"
NOISE_RE = re.compile(r"^[\s\d.,%()\-–+*/:]*$")                # axis ticks, bare numbers
MONTH_AXIS_RE = re.compile(r"^(Q[1-4]-\d{2}|[A-Z][a-z]{2}-\d{2}|\d{4})$")  # Q1-26, Jan-26, 2026
MIN_PAGE_CHARS = 80                                            # skip cover/empty pages


STEM_RE = re.compile(r"MPR_(\d{4})_Q([1-4])")
QUARTER_MONTH = {1: "March", 2: "June", 3: "September", 4: "December"}

def _make_header(stem, section, page, total):
    """Header tuned for BM25 (exact tokens) and dense embedding (natural phrasing)."""
    sec = section or "n/a"
    m = STEM_RE.search(stem)
    if not m:
        return f"[{stem} | {sec} | page {page}/{total}]"
    year, q = m.group(1), int(m.group(2))
    return (
        f"[{stem}] Bank of Thailand Monetary Policy Report {year} Q{q} "
        f"({QUARTER_MONTH[q]} {year} report, quarter {q} of {year}). "
        f"Section: {sec}. PDF p. {page}."
    )


def _clean_block(text: str) -> str:
    """Drop page numbers and chart axis noise, keep real sentences/labels."""
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if not s or PAGE_NO_RE.match(s):
            continue
        kept.append(s)
    return " ".join(kept)


def _table_rows(page, bbox, y_tol=3.0, gap=14.0):
    """Rebuild table rows from words so labels stay on the same line as their values."""
    words = [w for w in page.get_text("words")
             if bbox[0] <= (w[0] + w[2]) / 2 <= bbox[2] and bbox[1] <= (w[1] + w[3]) / 2 <= bbox[3]]
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows, cur, cur_y = [], [], None
    for w in words:
        y = (w[1] + w[3]) / 2
        if cur_y is None or abs(y - cur_y) <= y_tol:
            cur.append(w)
            cur_y = y if cur_y is None else (cur_y + y) / 2
        else:
            rows.append(cur)
            cur, cur_y = [w], y
    if cur:
        rows.append(cur)
    lines = []
    for r in rows:
        r.sort(key=lambda w: w[0])
        out = r[0][4]
        for prev, w in zip(r, r[1:]):
            out += (" | " if w[0] - prev[2] > gap else " ") + w[4]
        lines.append(out)
    return "\n".join(lines)


def _inside(b, bbox):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def _is_axis_block(text: str) -> bool:
    """Chart axis ticks: 3+ lines that are all bare numbers/dates."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return len(lines) >= 3 and all(NOISE_RE.match(l) or MONTH_AXIS_RE.match(l) for l in lines)


def _page_text(page) -> tuple[str, str]:
    """Return (section, body). Text blocks in reading order, tables rebuilt row-wise."""
    try:
        table_boxes = [t.bbox for t in page.find_tables().tables]
    except Exception:
        table_boxes = []
    blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0]
    cleaned = []
    for b in blocks:
        if any(_inside(b, tb) for tb in table_boxes) or _is_axis_block(b[4]):
            continue
        c = _clean_block(b[4])
        if c:
            cleaned.append(c)
    if not cleaned:
        return "", ""
    # In these decks the first block is the top-bar section label
    # ("Global economy", "The Thai economy", ...).
    section = cleaned[0] if len(cleaned[0]) <= 60 else ""
    body_blocks = cleaned[1:] if section else cleaned
    for tb in table_boxes:
        rows = _table_rows(page, tb)
        if rows:
            body_blocks.append("TABLE:\n" + rows)
    return section, "\n\n".join(body_blocks)


def load_and_chunk_pdfs(docs_dir, chunk_size=1500, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks, n_pages = [], 0
    for pdf_path in sorted(Path(docs_dir).glob("**/*.pdf")):
        doc = pymupdf.open(pdf_path)
        total = len(doc)
        for i, page in enumerate(doc, start=1):
            n_pages += 1
            section, body = _page_text(page)
            if len(body) < MIN_PAGE_CHARS:
                continue
            header = _make_header(pdf_path.stem, section, i, total)
            base_meta = {"source": str(pdf_path), "file": pdf_path.name,
                         "page": i, "section": section}
            parts = [body] if len(body) <= chunk_size else splitter.split_text(body)
            for j, part in enumerate(parts):
                chunks.append(Document(
                    page_content=f"{header}\n{part}",
                    metadata={**base_meta, "chunk_in_page": j},
                ))
    return chunks, n_pages
