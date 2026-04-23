"""
Persistent FAISS index cache for PolicyLens.

Cached artefacts live in ~/.policylens_cache/{md5_of_pdf}/:
    index.faiss   — FAISS IndexFlatIP written with faiss.write_index
    chunks.json   — JSON-serialised chunk list from retriever.chunk_pages

The MD5 is computed over the raw PDF bytes, so re-uploading the same file
always hits the cache and uploading a different file always misses.
"""

import hashlib
import json
from pathlib import Path

import faiss

_CACHE_DIR = Path.home() / ".policylens_cache"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_md5(pdf_path: str) -> str:
    """Return the hex MD5 digest of a PDF file, reading in 64 KB blocks."""
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _compute_md5_bytes(data) -> str:
    """Return the hex MD5 digest of a bytes-like / buffer-protocol object."""
    return hashlib.md5(bytes(data)).hexdigest()


def _cache_paths(md5: str) -> tuple[Path, Path]:
    """Return (faiss_path, chunks_path) for a given MD5 hash."""
    d = _CACHE_DIR / md5
    return d / "index.faiss", d / "chunks.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_index_path(pdf_path: str) -> str:
    """
    Return the path to the cached FAISS index file for the given PDF.

    The path is derived from the MD5 hash of the PDF content:
        ~/.policylens_cache/{md5}/index.faiss

    The file may not exist yet; call is_cached() first to check.
    """
    md5 = _compute_md5(pdf_path)
    faiss_path, _ = _cache_paths(md5)
    return str(faiss_path)


def save_index(index: faiss.Index, chunks: list[dict], pdf_path: str) -> None:
    """
    Persist a FAISS index and its chunk list to the cache directory.

    Creates ~/.policylens_cache/{md5}/ if it does not exist.
    """
    md5 = _compute_md5(pdf_path)
    faiss_path, chunks_path = _cache_paths(md5)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)


def load_index(pdf_path: str) -> tuple[faiss.Index, list[dict]] | None:
    """
    Load a cached FAISS index and chunk list for the given PDF.

    Returns (index, chunks) if a valid cache exists, otherwise None.
    A corrupt or incomplete cache is treated as a miss (returns None).
    """
    md5 = _compute_md5(pdf_path)
    faiss_path, chunks_path = _cache_paths(md5)
    if not faiss_path.exists() or not chunks_path.exists():
        return None
    try:
        index = faiss.read_index(str(faiss_path))
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks
    except Exception:
        return None


def is_cached(pdf_path: str) -> bool:
    """Return True if a valid cached index exists for the given PDF path."""
    md5 = _compute_md5(pdf_path)
    faiss_path, chunks_path = _cache_paths(md5)
    return faiss_path.exists() and chunks_path.exists()


def is_cached_upload(pdf_bytes) -> bool:
    """
    Return True if a valid cached index exists for a PDF given as raw bytes
    (e.g. a Streamlit UploadedFile buffer).  No file I/O required.
    """
    md5 = _compute_md5_bytes(pdf_bytes)
    faiss_path, chunks_path = _cache_paths(md5)
    return faiss_path.exists() and chunks_path.exists()
