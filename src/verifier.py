import re

from src.retriever import retrieve_evidence

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Direct causal attribution language in evidence
_CAUSAL_STRONG_RE = re.compile(
    r"\b(?:caused?|leads?|led\s+to|results?\s+in|resulted\s+in|produced?|"
    r"triggered?|because(?:\s+of)?|due\s+to|owing\s+to|"
    r"attributed?\s+to|contributed?\s+to|drove|driven\s+by)\b",
    re.IGNORECASE,
)

# Hedging / correlational language that weakens a causal claim
_CAUSAL_HEDGE_RE = re.compile(
    r"\b(?:may\s+have|might\s+have|could\s+have|possibly|perhaps|"
    r"potentially|appears?\s+to|likely|suggests?\s+that|factors?\s+may|"
    r"associated\s+with|linked\s+to|correlated\s+with|(?:macro)?economic\s+factors?)\b",
    re.IGNORECASE,
)

# Numbers with optional magnitude suffix — percentages captured via suffix "%"
_NUMBER_RE = re.compile(
    r"(\d[\d,]*\.?\d*)\s*(%|trillion|billion|million|thousand)?",
    re.IGNORECASE,
)

# Pre-processing patterns for _extract_numbers — mirrors claim_extractor filters
_LIST_MARKER_RE = re.compile(r"^\d{1,3}\.\s+")
_FY_YEAR_RE     = re.compile(
    r"\bFY\s*\d{4}\b|\bfiscal\s+year\s+\d{4}\b", re.IGNORECASE
)

_MULTIPLIERS: dict[str, float] = {
    "%": 1.0,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}

# Entity labels that carry semantic identity in policy documents.
# Numeric labels (PERCENT, MONEY, QUANTITY) are already handled by Rule 2.
_KEY_ENTITY_LABELS = {"ORG", "GPE", "PERSON", "LOC", "LAW"}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD: float = 0.35
NUMERIC_TOLERANCE: float = 0.10  # ±10 %

VERDICT_PRIORITY: dict[str, int] = {
    "High-Risk Silent Failure": 4,
    "Unsupported": 3,
    "Partially Supported": 2,
    "Supported": 1,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> list[float]:
    """Return all positive numeric values found in text, multipliers applied.

    Excludes three categories of false-positive numbers:
      1. Leading list markers  — "1. " / "2. " at sentence start
      2. FY / fiscal-year labels — "FY 2022", "fiscal year 2022"
      3. Digits embedded in compound terms — "19" in "COVID-19", "5"/"1" in "H5N1"
    """
    # Strip leading list marker and fiscal-year labels before scanning
    cleaned = _LIST_MARKER_RE.sub("", text, count=1)
    cleaned = _FY_YEAR_RE.sub(" ", cleaned)

    results: list[float] = []
    for m in _NUMBER_RE.finditer(cleaned):
        start = m.start()
        # Skip digits that are part of alphanumeric compound terms.
        # Two cases: alpha immediately before digit (H5N1) or
        # alpha-hyphen before digit (COVID-19, SARS-CoV-2).
        if start >= 1 and cleaned[start - 1].isalpha():
            continue
        if start >= 2 and cleaned[start - 1] == "-" and cleaned[start - 2].isalpha():
            continue
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            continue
        suffix = (m.group(2) or "").lower()
        val *= _MULTIPLIERS.get(suffix, 1.0)
        if val > 0:
            results.append(val)
    return results


def _numbers_match(claim_nums: list[float], evidence_nums: list[float]) -> bool:
    """
    Return True if at least one claim number appears in evidence within ±NUMERIC_TOLERANCE.

    Year-like integers in the range 1800–2100 are compared exactly to prevent the
    ±10% window from matching nearby years (e.g. 2020 ≈ 2022 would be a false pass).
    """
    def _is_year(n: float) -> bool:
        return n == int(n) and 1800 <= n <= 2100

    for cn in claim_nums:
        for en in evidence_nums:
            if _is_year(cn) or _is_year(en):
                if cn == en:
                    return True
            elif cn == 0:
                if en == 0:
                    return True
            else:
                if abs(cn - en) / abs(cn) <= NUMERIC_TOLERANCE:
                    return True
    return False


def _missing_key_entities(raw_entities: list[dict], evidence_text: str) -> list[str]:
    """
    Return the text of key entities from the claim that are absent from evidence.

    Only checks entity labels in _KEY_ENTITY_LABELS (ORG, GPE, PERSON, LOC, LAW).
    Numeric labels (PERCENT, MONEY, QUANTITY) are handled separately by Rule 2.
    Matching is case-insensitive substring search.
    """
    evidence_lower = evidence_text.lower()
    return [
        e["text"]
        for e in raw_entities
        if e["label"] in _KEY_ENTITY_LABELS
        and e["text"].lower() not in evidence_lower
    ]


def _compute_confidence(verdict: str, top_score: float) -> float:
    """
    Map verdict type and top evidence similarity to a 0-1 confidence score.

    "Supported"            → similarity score directly (high evidence alignment)
    "Partially Supported"  → attenuated similarity (structural concern present)
    "Unsupported"          → heavily attenuated (evidence relevance already low)
    "High-Risk Silent Failure" → near-zero despite potentially high similarity,
                                 because this is the most dangerous failure mode
    """
    if verdict == "Supported":
        score = top_score
    elif verdict == "Partially Supported":
        score = top_score * 0.50
    elif verdict == "Unsupported":
        score = top_score * 0.20
    else:  # High-Risk Silent Failure — high similarity + numeric mismatch = red flag
        score = min(top_score * 0.12, 0.08)

    return round(max(0.01, min(0.99, score)), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_claim(claim: dict, evidence: list[dict]) -> dict:
    """
    Apply the four-rule deterministic engine to a single claim and return a
    structured verdict.

    Rules are applied in order; all that fire are collected, and the highest-
    priority verdict wins:
        High-Risk Silent Failure > Unsupported > Partially Supported > Supported

    Parameters
    ----------
    claim : dict
        A single output dict from claim_extractor.extract_claims.
    evidence : list[dict]
        Retrieved passages from retriever.retrieve_evidence.

    Returns
    -------
    dict
        {claim_id, claim_text, page_number, flags, evidence,
         verdict, risk_explanation, rules_triggered, confidence_score}
    """
    top_score = evidence[0]["similarity_score"] if evidence else 0.0
    relevant = [e for e in evidence if e["similarity_score"] >= SIMILARITY_THRESHOLD]

    verdicts: list[str] = []
    rules_triggered: list[str] = []
    explanations: list[str] = []

    # ------------------------------------------------------------------
    # Rule 1 — Similarity Threshold
    # ------------------------------------------------------------------
    if top_score < SIMILARITY_THRESHOLD:
        verdicts.append("Unsupported")
        rules_triggered.append("similarity_threshold")
        explanations.append(
            f"Top evidence similarity {top_score:.3f} is below threshold "
            f"{SIMILARITY_THRESHOLD} — no relevant source passage found."
        )

    # Rules 2-4 operate on evidence that cleared the similarity bar.
    # Checking against irrelevant passages would produce meaningless signals.
    if relevant:
        evidence_text = " ".join(e["text"] for e in relevant)

        # --------------------------------------------------------------
        # Rule 2 — Numeric Match
        # --------------------------------------------------------------
        if "numeric" in claim["flags"]:
            claim_nums = _extract_numbers(claim["text"])
            if claim_nums:
                evidence_nums = _extract_numbers(evidence_text)
                if not _numbers_match(claim_nums, evidence_nums):
                    verdicts.append("High-Risk Silent Failure")
                    rules_triggered.append("numeric_match")
                    explanations.append(
                        f"Claim contains specific value(s) {[round(n, 4) for n in claim_nums]} "
                        f"not found within ±{int(NUMERIC_TOLERANCE * 100)}% in retrieved evidence."
                    )

        # --------------------------------------------------------------
        # Rule 3 — Causal Scrutiny
        # --------------------------------------------------------------
        if "causal_verb" in claim["flags"]:
            if not _CAUSAL_STRONG_RE.search(evidence_text):
                verdicts.append("Partially Supported")
                rules_triggered.append("causal_scrutiny")
                if _CAUSAL_HEDGE_RE.search(evidence_text):
                    explanations.append(
                        "Claim asserts direct causation but evidence uses hedging language "
                        "(e.g. 'may have', 'potentially') — causal link is not established."
                    )
                else:
                    explanations.append(
                        "Claim asserts causation but evidence only describes general trends "
                        "without any causal attribution."
                    )

        # --------------------------------------------------------------
        # Rule 4 — Entity Consistency
        # --------------------------------------------------------------
        if "named_entity" in claim["flags"]:
            missing = _missing_key_entities(
                claim.get("raw_entities", []), evidence_text
            )
            if missing:
                verdicts.append("Partially Supported")
                rules_triggered.append("entity_consistency")
                explanations.append(
                    f"Key entities from claim absent in evidence: "
                    f"{', '.join(repr(e) for e in missing)}."
                )

    # ------------------------------------------------------------------
    # Resolve final verdict
    # ------------------------------------------------------------------
    if not verdicts:
        final_verdict = "Supported"
        explanations.append("All verification rules passed — claim is grounded in evidence.")
    else:
        final_verdict = max(verdicts, key=lambda v: VERDICT_PRIORITY[v])

    return {
        "claim_id":         claim["claim_id"],
        "claim_text":       claim["text"],
        "page_number":      claim["page_number"],
        "flags":            claim["flags"],
        "evidence":         evidence,
        "verdict":          final_verdict,
        "risk_explanation": " ".join(explanations),
        "rules_triggered":  rules_triggered,
        "confidence_score": _compute_confidence(final_verdict, top_score),
        "detection_method": claim.get("detection_method", "regex"),
    }


def verify_claims(claims: list[dict], index, chunks: list[dict]) -> list[dict]:
    """
    Run the full retrieval → verification pipeline for a list of claims.

    Parameters
    ----------
    claims : list[dict]
        Output of claim_extractor.extract_claims.
    index : faiss.Index
        FAISS index from retriever.build_index.
    chunks : list[dict]
        Chunk list returned alongside the index by retriever.build_index.

    Returns
    -------
    list[dict]
        One verdict dict per claim (same order as input).
    """
    return [
        verify_claim(claim, retrieve_evidence(claim["text"], index, chunks, top_k=3))
        for claim in claims
    ]


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.parser import extract_text_from_pdf
    from src.claim_extractor import extract_claims
    from src.retriever import chunk_pages, build_index

    # Inline sample mimicking a policy document source + an AI-generated summary
    SOURCE_PAGES = [
        {
            "page_number": 1,
            "text": (
                "Emissions declined during the review period covering 2018 to 2022, "
                "though macroeconomic factors may have contributed to the trend. "
                "The Environmental Protection Agency monitored air quality across 12 states. "
                "No single cause was identified for the observed reduction."
            ),
        },
        {
            "page_number": 2,
            "text": (
                "The federal jobs program expanded operations across the Midwest and Southeast. "
                "Approximately 1.2 million positions were added over three years. "
                "Congress passed the Workforce Expansion Act in March 2021. "
                "Independent auditors reviewed the program outcomes."
            ),
        },
    ]

    # These simulate claims extracted from an AI-generated summary of the above
    SUMMARY_CLAIMS = [
        {
            "claim_id": 1,
            "text": "The policy reduced emissions by 22% over five years.",
            "page_number": 1,
            "flags": ["numeric", "causal_verb"],
            "raw_entities": [],
        },
        {
            "claim_id": 2,
            "text": "The Environmental Protection Agency confirmed the findings.",
            "page_number": 1,
            "flags": ["named_entity"],
            "raw_entities": [{"text": "The Environmental Protection Agency", "label": "ORG"}],
        },
        {
            "claim_id": 3,
            "text": "The program generated 3.5 million jobs across the Midwest.",
            "page_number": 2,
            "flags": ["numeric", "causal_verb", "named_entity"],
            "raw_entities": [{"text": "Midwest", "label": "GPE"}],
        },
        {
            "claim_id": 4,
            "text": "The legislation was authored by Senator Williams in 2019.",
            "page_number": 2,
            "flags": ["named_entity"],
            "raw_entities": [
                {"text": "Senator Williams", "label": "PERSON"},
                {"text": "2019", "label": "DATE"},
            ],
        },
    ]

    VERDICT_EMOJI = {
        "Supported": "✓",
        "Partially Supported": "~",
        "Unsupported": "✗",
        "High-Risk Silent Failure": "!",
    }

    print("Building index from source document...\n")
    chunks = chunk_pages(SOURCE_PAGES, chunk_size=3)
    index, chunks = build_index(chunks)

    print("Running verification pipeline...\n")
    print("=" * 70)
    for claim in SUMMARY_CLAIMS:
        evidence = retrieve_evidence(claim["text"], index, chunks, top_k=3)
        result = verify_claim(claim, evidence)

        marker = VERDICT_EMOJI[result["verdict"]]
        print(f"[{marker}] Claim {result['claim_id']}: {result['claim_text']}")
        print(f"    Verdict:    {result['verdict']}")
        print(f"    Confidence: {result['confidence_score']:.4f}")
        print(f"    Rules:      {result['rules_triggered'] or ['none']}")
        print(f"    Reasoning:  {result['risk_explanation']}")
        if result["evidence"]:
            top = result["evidence"][0]
            print(f"    Top evidence (score={top['similarity_score']:.3f}, p.{top['page_number']}):")
            print(f"      \"{top['text'][:120]}\"")
        print()
