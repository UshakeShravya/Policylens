import faiss
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

_MODEL_NAME = EMBEDDING_MODEL
_model: SentenceTransformer | None = None
_nlp = None


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the sentence-transformers model (thread-safe via GIL)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _get_nlp():
    """Lazy-load and cache the spaCy pipeline with only the sentencizer enabled."""
    global _nlp
    if _nlp is None:
        # Only sentence segmentation needed here — disable heavier components
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "attribute_ruler"])
    return _nlp


def _sentencize(text: str) -> list[str]:
    """Split text into sentences using spaCy, filtering blank results."""
    doc = _get_nlp()(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def _embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised float32 embeddings for cosine similarity via IndexFlatIP."""
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings


def chunk_pages(pages: list[dict], chunk_size: int = 3) -> list[dict]:
    """
    Split each page's text into overlapping sentence-based chunks.

    Overlap is fixed at 1 sentence so consecutive chunks share context.
    A page with fewer sentences than chunk_size yields a single chunk.

    Parameters
    ----------
    pages : list[dict]
        Output of parser.extract_text_from_pdf.
    chunk_size : int
        Number of sentences per chunk (default 3).

    Returns
    -------
    list[dict]
        [{"chunk_id": int, "text": str, "page_number": int, "sentence_start": int}, ...]
    """
    OVERLAP = 1
    step = max(1, chunk_size - OVERLAP)

    chunks = []
    chunk_id = 1

    for page in pages:
        sentences = _sentencize(page["text"])
        if not sentences:
            continue

        for i in range(0, len(sentences), step):
            window = sentences[i : i + chunk_size]
            chunks.append({
                "chunk_id": chunk_id,
                "text": " ".join(window),
                "page_number": page["page_number"],
                "sentence_start": i,
            })
            chunk_id += 1

    return chunks




def build_index(
    chunks: list[dict],
    pdf_path: str | None = None,
) -> tuple[faiss.Index, list[dict]]:
    """
    Embed all chunks and build a FAISS IndexFlatIP for cosine similarity search.

    If pdf_path is provided the result is persisted to (and loaded from) the
    persistent cache at ~/.policylens_cache/{md5}/.  Subsequent calls with the
    same PDF skip embedding entirely.

    Cosine similarity is computed as inner product over L2-normalised vectors,
    which avoids the overhead of IndexFlatL2 + distance conversion.

    Parameters
    ----------
    chunks : list[dict]
        Output of chunk_pages.
    pdf_path : str | None
        Path to the source PDF, used as cache key.  Pass None to skip caching.

    Returns
    -------
    (faiss.Index, list[dict])
        The FAISS index and the original chunks list (index positions align).
    """
    if pdf_path is not None:
        try:
            from src.index_store import load_index, save_index
            cached = load_index(pdf_path)
            if cached is not None:
                print("[CACHE] Loaded index from cache", flush=True)
                return cached
        except Exception:
            pass  # cache miss or import error — fall through to rebuild

    texts = [c["text"] for c in chunks]
    embeddings = _embed(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    if pdf_path is not None:
        try:
            from src.index_store import save_index
            save_index(index, chunks, pdf_path)
            print("[CACHE] Built and saved new index", flush=True)
        except Exception:
            pass  # non-fatal — audit continues without caching

    return index, chunks


def retrieve_evidence(
    claim_text: str,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve the top_k most similar document chunks for a given claim.

    Parameters
    ----------
    claim_text : str
        The claim sentence to look up.
    index : faiss.Index
        FAISS index built by build_index.
    chunks : list[dict]
        Chunk list returned alongside the index by build_index.
    top_k : int
        Number of results to return (default 3).

    Returns
    -------
    list[dict]
        [{"chunk_id": int, "text": str, "page_number": int,
          "similarity_score": float}, ...]
        Ordered from most to least similar.  Scores are clamped to [0, 1].
    """
    query_embedding = _embed([claim_text])
    actual_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, actual_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "page_number": chunk["page_number"],
            "similarity_score": float(np.clip(score, 0.0, 1.0)),
        })

    return results


if __name__ == "__main__":
    sample_pages = [
        {
            "page_number": 1,
            "text": (
                "The review period covered fiscal years 2018 through 2022. "
                "Emissions declined during the review period, though macroeconomic "
                "factors may have contributed. "
                "The policy reduced emissions by 22% over five years. "
                "The Environmental Protection Agency confirmed the findings."
            ),
        },
        {
            "page_number": 2,
            "text": (
                "The program generated 3.5 million jobs across the Midwest. "
                "Unemployment never exceeded 4% during the implementation phase. "
                "Congress passed the Inflation Reduction Act in August 2022. "
                "All participating states reported improved outcomes. "
                "Independent auditors verified the results."
            ),
        },
    ]

    print("=== chunk_pages ===")
    chunks = chunk_pages(sample_pages, chunk_size=3)
    for c in chunks:
        print(f"[chunk {c['chunk_id']}] p.{c['page_number']} sent_start={c['sentence_start']}")
        print(f"  {c['text']}\n")

    print("=== build_index ===")
    index, chunks = build_index(chunks)
    print(f"Index built: {index.ntotal} vectors, dim={index.d}\n")

    print("=== retrieve_evidence ===")
    test_claims = [
        "The policy reduced emissions by 22%.",
        "Jobs were created across the Midwest.",
    ]
    for claim in test_claims:
        print(f"Claim: {claim!r}")
        hits = retrieve_evidence(claim, index, chunks, top_k=2)
        for h in hits:
            print(f"  [chunk {h['chunk_id']}] score={h['similarity_score']:.4f} p.{h['page_number']}")
            print(f"    {h['text']}")
        print()
