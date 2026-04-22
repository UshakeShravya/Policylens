import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_VERDICTS = [
    "Supported",
    "Partially Supported",
    "Unsupported",
    "High-Risk Silent Failure",
]

_VERDICT_MARKER = {
    "Supported": "  [+]",
    "Partially Supported": "  [~]",
    "Unsupported": "  [-]",
    "High-Risk Silent Failure": "  [!]",
}

_LINE_WIDE = "=" * 72
_LINE_THIN = "-" * 72


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trunc(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(results: list[dict]) -> dict:
    """
    Aggregate a list of verdict dicts into a structured audit report.

    Parameters
    ----------
    results : list[dict]
        Output of verifier.verify_claims.

    Returns
    -------
    dict
        {
          total_claims        : int,
          verdict_counts      : dict[str, int],
          silent_failure_rate : float,
          unsupported_rate    : float,
          high_risk_claims    : list[dict],   # full verdict dicts
          summary_table       : list[dict],   # truncated, display-ready
        }
    """
    total = len(results)

    verdict_counts: dict[str, int] = {v: 0 for v in ALL_VERDICTS}
    for r in results:
        verdict_counts[r["verdict"]] = verdict_counts.get(r["verdict"], 0) + 1

    high_risk = [r for r in results if r["verdict"] == "High-Risk Silent Failure"]

    summary_table = []
    for r in results:
        top_evidence = ""
        if r.get("evidence"):
            top_evidence = _trunc(r["evidence"][0]["text"], 150)

        summary_table.append({
            "claim_id": r["claim_id"],
            "claim_text": _trunc(r["claim_text"], 100),
            "page_number": r["page_number"],
            "verdict": r["verdict"],
            "confidence_score": r["confidence_score"],
            "top_evidence": top_evidence,
            "risk_explanation": r["risk_explanation"],
        })

    return {
        "total_claims": total,
        "verdict_counts": verdict_counts,
        "silent_failure_rate": _rate(verdict_counts["High-Risk Silent Failure"], total),
        "unsupported_rate": _rate(verdict_counts["Unsupported"], total),
        "high_risk_claims": high_risk,
        "summary_table": summary_table,
    }


def report_to_dataframe(report: dict) -> pd.DataFrame:
    """
    Convert the summary_table portion of a report into a pandas DataFrame.

    Parameters
    ----------
    report : dict
        Output of generate_report.

    Returns
    -------
    pd.DataFrame
        One row per claim, columns matching summary_table keys.
    """
    return pd.DataFrame(report["summary_table"])


def report_to_json(report: dict, output_path: str) -> None:
    """
    Persist the full report as a formatted JSON file.

    Creates intermediate directories if they do not exist.

    Parameters
    ----------
    report : dict
        Output of generate_report.
    output_path : str
        Destination file path (e.g. "eval/report.json").
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def print_report(report: dict) -> None:
    """
    Print a formatted terminal audit summary.

    Shows overall statistics, verdict breakdown with percentages, and a
    detailed section for every High-Risk Silent Failure.
    """
    total = report["total_claims"]
    counts = report["verdict_counts"]

    print(_LINE_WIDE)
    print("  PolicyLens — Audit Report")
    print(_LINE_WIDE)
    print(f"  Total claims evaluated : {total}")
    print(_LINE_THIN)

    print("  VERDICT BREAKDOWN")
    for verdict in ALL_VERDICTS:
        n = counts[verdict]
        pct = f"{n / total:.1%}" if total else "—"
        marker = _VERDICT_MARKER[verdict]  # already includes brackets and leading spaces
        label = f"{marker} {verdict}"
        print(f"  {label:<36}  {n:>3}  ({pct})")

    print(_LINE_THIN)
    print(f"  Silent Failure Rate  : {report['silent_failure_rate']:.2%}")
    print(f"  Unsupported Rate     : {report['unsupported_rate']:.2%}")

    high_risk = report["high_risk_claims"]
    if not high_risk:
        print(_LINE_WIDE)
        print("  No High-Risk Silent Failures detected.")
        print(_LINE_WIDE)
        return

    print(_LINE_WIDE)
    print(f"  HIGH-RISK SILENT FAILURES  ({len(high_risk)} claim{'s' if len(high_risk) != 1 else ''})")
    print(_LINE_WIDE)

    for r in high_risk:
        print(f"  Claim {r['claim_id']}  |  page {r['page_number']}  "
              f"|  confidence {r['confidence_score']:.4f}")
        print(f"  \"{_trunc(r['claim_text'], 100)}\"")
        print(f"  Reasoning: {r['risk_explanation']}")

        if r.get("evidence"):
            top = r["evidence"][0]
            print(
                f"  Top evidence (similarity={top['similarity_score']:.3f}, "
                f"p.{top['page_number']}):"
            )
            print(f"    \"{_trunc(top['text'], 120)}\"")

        print(_LINE_THIN)

    print()


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import os

    MOCK_RESULTS = [
        {
            "claim_id": 1,
            "claim_text": "The policy reduced emissions by 22% over five years.",
            "page_number": 1,
            "flags": ["numeric", "causal_verb"],
            "evidence": [
                {
                    "chunk_id": 1,
                    "text": (
                        "Emissions declined during the review period, though macroeconomic "
                        "factors may have contributed to the trend."
                    ),
                    "page_number": 1,
                    "similarity_score": 0.62,
                }
            ],
            "verdict": "High-Risk Silent Failure",
            "risk_explanation": (
                "Claim contains specific value(s) [22.0] not found within ±10% "
                "in retrieved evidence."
            ),
            "rules_triggered": ["numeric_match"],
            "confidence_score": 0.0744,
        },
        {
            "claim_id": 2,
            "claim_text": "The Environmental Protection Agency confirmed the findings.",
            "page_number": 1,
            "flags": ["named_entity"],
            "evidence": [
                {
                    "chunk_id": 1,
                    "text": (
                        "The Environmental Protection Agency monitored air quality "
                        "across 12 states. No single cause was identified."
                    ),
                    "page_number": 1,
                    "similarity_score": 0.72,
                }
            ],
            "verdict": "Supported",
            "risk_explanation": "All verification rules passed — claim is grounded in evidence.",
            "rules_triggered": [],
            "confidence_score": 0.72,
        },
        {
            "claim_id": 3,
            "claim_text": "The program generated 3.5 million jobs across the Midwest.",
            "page_number": 2,
            "flags": ["numeric", "causal_verb", "named_entity"],
            "evidence": [
                {
                    "chunk_id": 3,
                    "text": (
                        "Approximately 1.2 million positions were added over three years "
                        "in the Midwest and Southeast."
                    ),
                    "page_number": 2,
                    "similarity_score": 0.619,
                }
            ],
            "verdict": "High-Risk Silent Failure",
            "risk_explanation": (
                "Claim contains specific value(s) [3500000.0] not found within ±10% "
                "in retrieved evidence. Claim asserts causation but evidence only "
                "describes general trends without any causal attribution."
            ),
            "rules_triggered": ["numeric_match", "causal_scrutiny"],
            "confidence_score": 0.0743,
        },
        {
            "claim_id": 4,
            "claim_text": "The legislation was authored by Senator Williams in 2019.",
            "page_number": 2,
            "flags": ["named_entity"],
            "evidence": [
                {
                    "chunk_id": 4,
                    "text": "Congress passed the Workforce Expansion Act in March 2021.",
                    "page_number": 2,
                    "similarity_score": 0.377,
                }
            ],
            "verdict": "Partially Supported",
            "risk_explanation": (
                "Key entities from claim absent in evidence: 'Senator Williams'."
            ),
            "rules_triggered": ["entity_consistency"],
            "confidence_score": 0.1885,
        },
        {
            "claim_id": 5,
            "claim_text": (
                "The report demonstrated unprecedented improvement in air quality metrics."
            ),
            "page_number": 1,
            "flags": ["superlative"],
            "evidence": [
                {
                    "chunk_id": 1,
                    "text": "Emissions declined during the review period.",
                    "page_number": 1,
                    "similarity_score": 0.41,
                }
            ],
            "verdict": "Partially Supported",
            "risk_explanation": (
                "Claim uses superlative language not substantiated by retrieved evidence."
            ),
            "rules_triggered": [],
            "confidence_score": 0.205,
        },
        {
            "claim_id": 6,
            "claim_text": "The WHO issued a global health warning in response to the findings.",
            "page_number": 3,
            "flags": ["named_entity"],
            "evidence": [
                {
                    "chunk_id": 1,
                    "text": "Emissions declined during the review period.",
                    "page_number": 1,
                    "similarity_score": 0.18,
                }
            ],
            "verdict": "Unsupported",
            "risk_explanation": (
                "Top evidence similarity 0.180 is below threshold 0.35 — "
                "no relevant source passage found."
            ),
            "rules_triggered": ["similarity_threshold"],
            "confidence_score": 0.036,
        },
    ]

    report = generate_report(MOCK_RESULTS)

    # --- Terminal output ---
    print_report(report)

    # --- DataFrame ---
    df = report_to_dataframe(report)
    print("DataFrame preview:")
    print(df[["claim_id", "verdict", "confidence_score", "page_number"]].to_string(index=False))
    print()

    # --- JSON round-trip ---
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "sub", "report.json")
        report_to_json(report, json_path)
        with open(json_path, encoding="utf-8") as f:
            loaded = json.load(f)

    print("JSON round-trip:")
    print(f"  total_claims        = {loaded['total_claims']}")
    print(f"  silent_failure_rate = {loaded['silent_failure_rate']:.2%}")
    print(f"  unsupported_rate    = {loaded['unsupported_rate']:.2%}")
    print(f"  verdict_counts      = {loaded['verdict_counts']}")
    print(f"  summary_table rows  = {len(loaded['summary_table'])}")
    print(f"  high_risk_claims    = {len(loaded['high_risk_claims'])}")
