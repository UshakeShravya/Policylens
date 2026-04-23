"""
Batch processing layer for PolicyLens.

Runs the full audit pipeline for multiple (PDF, summary) pairs in parallel
and produces a cross-document comparison summary.
"""

import concurrent.futures
from pathlib import Path

from src.parser import extract_text_from_pdf
from src.claim_extractor import extract_claims_full
from src.retriever import chunk_pages, build_index
from src.agent import verify_claims_with_agent
from src.reporter import generate_report


def _empty_report(document_name: str, error: str = "") -> dict:
    return {
        "document_name": document_name,
        "error": error,
        "total_claims": 0,
        "verdict_counts": {
            "Supported": 0,
            "Partially Supported": 0,
            "Unsupported": 0,
            "High-Risk Silent Failure": 0,
        },
        "silent_failure_rate": 0.0,
        "unsupported_rate": 0.0,
        "high_risk_claims": [],
        "summary_table": [],
    }


def _audit_one(pdf_path: str, summary_text: str, document_name: str | None = None) -> dict:
    """Run the full PolicyLens pipeline for a single (PDF, summary) pair."""
    doc_name = document_name or Path(pdf_path).name

    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        return _empty_report(doc_name, "No text could be extracted from the PDF.")

    claim_pages = (
        [{"page_number": 1, "text": summary_text}]
        if summary_text.strip()
        else pages
    )

    def _build():
        c = chunk_pages(pages, chunk_size=3)
        return build_index(c, pdf_path=pdf_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        claims_future = pool.submit(extract_claims_full, pdf_path, claim_pages)
        index_future  = pool.submit(_build)
        claims        = claims_future.result()
        index, chunks = index_future.result()

    if not claims:
        report = generate_report([])
        report["document_name"] = doc_name
        report["error"] = ""
        return report

    results = verify_claims_with_agent(claims, index, chunks)
    report = generate_report(results)
    report["document_name"] = doc_name
    report["error"] = ""
    return report


def batch_audit(
    pdf_paths: list[str],
    summary_texts: list[str],
    document_names: list[str] | None = None,
) -> list[dict]:
    """
    Run the full PolicyLens pipeline for multiple (PDF, summary) pairs in parallel.

    Parameters
    ----------
    pdf_paths : list[str]
        Paths to source PDF files (may be temp paths).
    summary_texts : list[str]
        AI-generated summaries to verify, one per PDF.  Pass an empty string to
        self-audit the source document directly.
    document_names : list[str] | None
        Display names for each document (e.g. original upload filenames).
        Falls back to the basename of each pdf_path when not provided.

    Returns
    -------
    list[dict]
        One report dict per document (same order as input).  Each report has an
        extra ``document_name`` field and an ``error`` field (empty on success).
    """
    names = document_names or [Path(p).name for p in pdf_paths]
    triples = list(zip(pdf_paths, summary_texts, names))
    results: list[dict | None] = [None] * len(triples)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        future_to_idx = {
            pool.submit(_audit_one, path, summary, name): i
            for i, (path, summary, name) in enumerate(triples)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = _empty_report(names[idx], str(exc))

    return results  # type: ignore[return-value]


def compare_reports(reports: list[dict]) -> dict:
    """
    Summarise and compare multiple audit reports.

    Parameters
    ----------
    reports : list[dict]
        Output of batch_audit (or a list of generate_report results each with
        a ``document_name`` field).

    Returns
    -------
    dict
        {
          documents                 : list of per-doc summary dicts,
          overall_riskiest_document : str | None,
          overall_safest_document   : str | None,
          total_claims_across_all   : int,
          total_high_risk_across_all: int,
        }
    """
    docs = [
        {
            "document_name":      r.get("document_name", "Unknown"),
            "total_claims":       r["total_claims"],
            "silent_failure_rate": r["silent_failure_rate"],
            "unsupported_rate":   r["unsupported_rate"],
            "high_risk_count":    r["verdict_counts"]["High-Risk Silent Failure"],
            "error":              r.get("error") or "",
        }
        for r in reports
    ]

    valid = [d for d in docs if not d["error"] and d["total_claims"] > 0]

    riskiest = (
        max(valid, key=lambda d: d["silent_failure_rate"])["document_name"]
        if valid else None
    )
    safest = (
        min(valid, key=lambda d: d["silent_failure_rate"])["document_name"]
        if valid else None
    )

    return {
        "documents":                  docs,
        "overall_riskiest_document":  riskiest,
        "overall_safest_document":    safest,
        "total_claims_across_all":    sum(r["total_claims"] for r in reports),
        "total_high_risk_across_all": sum(
            r["verdict_counts"]["High-Risk Silent Failure"] for r in reports
        ),
    }
