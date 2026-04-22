"""
tests/test_pipeline.py — Unit tests for all five PolicyLens core modules.

Run directly:   python tests/test_pipeline.py
Run via module: python -m unittest tests.test_pipeline -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# Ensure the project root is on sys.path so `src.*` imports resolve correctly
# regardless of the working directory the test runner uses.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. TestParser
# ============================================================

class TestParser(unittest.TestCase):
    """Tests for src.parser.extract_text_from_pdf"""

    def _make_mock_pdf_ctx(self, pages_text: list):
        """Build a mock context manager that mimics pdfplumber.open."""
        mock_pages = []
        for text in pages_text:
            page = MagicMock()
            page.extract_text.return_value = text
            mock_pages.append(page)

        mock_pdf = MagicMock()
        mock_pdf.pages = mock_pages

        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_pdf
        mock_ctx.__exit__.return_value = False
        return mock_ctx

    @patch("pdfplumber.open")
    @patch("src.parser.Path")
    def test_extract_returns_list(self, mock_path_cls, mock_open):
        from src.parser import extract_text_from_pdf
        mock_path_cls.return_value.exists.return_value = True
        mock_open.return_value = self._make_mock_pdf_ctx(
            ["The agency reduced carbon emissions in fiscal year 2020."]
        )
        result = extract_text_from_pdf("dummy.pdf")
        self.assertIsInstance(result, list)

    @patch("pdfplumber.open")
    @patch("src.parser.Path")
    def test_page_dict_has_required_keys(self, mock_path_cls, mock_open):
        from src.parser import extract_text_from_pdf
        mock_path_cls.return_value.exists.return_value = True
        mock_open.return_value = self._make_mock_pdf_ctx(
            ["Some policy document text on page one.", "More text on page two."]
        )
        result = extract_text_from_pdf("dummy.pdf")
        self.assertGreater(len(result), 0, "Expected at least one non-blank page")
        for page in result:
            self.assertIn("page_number", page)
            self.assertIn("text", page)

    def test_invalid_path_raises_error(self):
        from src.parser import extract_text_from_pdf
        with self.assertRaises(FileNotFoundError):
            extract_text_from_pdf("/no/such/file/policy_document.pdf")


# ============================================================
# 2. TestClaimExtractor
# ============================================================

class TestClaimExtractor(unittest.TestCase):
    """Tests for src.claim_extractor.extract_claims"""

    def _pages(self, text: str, page_number: int = 1) -> list:
        return [{"page_number": page_number, "text": text}]

    def test_numeric_claim_flagged(self):
        from src.claim_extractor import extract_claims
        pages = self._pages(
            "The agency reduced carbon emissions by 47% in fiscal year 2020."
        )
        claims = extract_claims(pages)
        self.assertGreater(len(claims), 0, "Expected at least one claim")
        all_flags = [f for c in claims for f in c["flags"]]
        self.assertIn("numeric", all_flags)

    def test_causal_verb_flagged(self):
        from src.claim_extractor import extract_claims
        pages = self._pages(
            "The new clean energy policy resulted in a significant reduction of greenhouse gases."
        )
        claims = extract_claims(pages)
        self.assertGreater(len(claims), 0, "Expected at least one claim")
        all_flags = [f for c in claims for f in c["flags"]]
        self.assertIn("causal_verb", all_flags)

    def test_short_sentence_skipped(self):
        from src.claim_extractor import extract_claims
        # "Up 47 percent." is only 3 words — below the 5-word minimum
        pages = self._pages("Up 47 percent.")
        claims = extract_claims(pages)
        self.assertEqual(claims, [], "Sentences under 5 words must not be returned")

    def test_output_has_required_keys(self):
        from src.claim_extractor import extract_claims
        pages = self._pages(
            "Congress allocated $2.3 billion to clean water initiatives across all 50 states."
        )
        claims = extract_claims(pages)
        self.assertGreater(len(claims), 0, "Expected at least one claim")
        required = {"claim_id", "text", "page_number", "flags", "raw_entities"}
        for claim in claims:
            for key in required:
                self.assertIn(key, claim, f"Missing required key '{key}' in claim dict")


# ============================================================
# 3. TestRetriever
# ============================================================

class TestRetriever(unittest.TestCase):
    """Integration tests — these load the sentence-transformer model once via setUpClass."""

    _SAMPLE_PAGES = [
        {
            "page_number": 1,
            "text": (
                "Emissions declined during the review period covering 2018 to 2022. "
                "The Environmental Protection Agency monitored air quality across twelve states. "
                "No single cause was identified for the observed reduction in pollution."
            ),
        },
        {
            "page_number": 2,
            "text": (
                "The federal jobs program expanded operations across the Midwest. "
                "Approximately 1.2 million positions were added over three years. "
                "Congress passed the Workforce Expansion Act in March 2021."
            ),
        },
    ]

    @classmethod
    def setUpClass(cls):
        """Build the FAISS index once; all retriever tests reuse it."""
        from src.retriever import chunk_pages, build_index
        raw_chunks = chunk_pages(cls._SAMPLE_PAGES, chunk_size=3)
        cls.index, cls.chunks = build_index(raw_chunks)

    def test_chunk_pages_returns_chunks(self):
        from src.retriever import chunk_pages
        chunks = chunk_pages(self._SAMPLE_PAGES, chunk_size=3)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, dict)
            for key in ("chunk_id", "text", "page_number", "sentence_start"):
                self.assertIn(key, chunk)

    def test_build_index_returns_tuple(self):
        import faiss
        self.assertIsInstance(self.index, faiss.Index)
        self.assertIsInstance(self.chunks, list)
        self.assertGreater(self.index.ntotal, 0, "Index must contain at least one vector")

    def test_retrieve_returns_top_k(self):
        from src.retriever import retrieve_evidence
        top_k = 2
        results = retrieve_evidence(
            "emissions reduction climate policy", self.index, self.chunks, top_k=top_k
        )
        self.assertEqual(len(results), top_k)

    def test_similarity_scores_between_0_and_1(self):
        from src.retriever import retrieve_evidence
        results = retrieve_evidence(
            "jobs created in the Midwest", self.index, self.chunks, top_k=3
        )
        for r in results:
            self.assertGreaterEqual(r["similarity_score"], 0.0)
            self.assertLessEqual(r["similarity_score"], 1.0)


# ============================================================
# 4. TestVerifier
# ============================================================

class TestVerifier(unittest.TestCase):
    """Tests for src.verifier.verify_claim — uses pre-constructed dicts, no FAISS."""

    def _claim(self, text: str, flags: list, entities: list = None) -> dict:
        return {
            "claim_id": 1,
            "text": text,
            "page_number": 1,
            "flags": flags,
            "raw_entities": entities or [],
        }

    def _evidence(self, text: str, score: float) -> list:
        return [
            {"chunk_id": 1, "text": text, "page_number": 1, "similarity_score": score}
        ]

    def test_low_similarity_returns_unsupported(self):
        from src.verifier import verify_claim
        claim = self._claim(
            "The policy reduced emissions by 40% over five years.",
            ["numeric", "causal_verb"],
        )
        # Score below SIMILARITY_THRESHOLD (0.35) → no relevant evidence
        evidence = self._evidence("Weather was pleasant yesterday.", score=0.10)
        result = verify_claim(claim, evidence)
        self.assertEqual(result["verdict"], "Unsupported")

    def test_numeric_mismatch_returns_high_risk(self):
        from src.verifier import verify_claim
        claim = self._claim(
            "Carbon emissions fell by 47% over the review period.",
            ["numeric"],
        )
        # Evidence clears the similarity bar but contains a very different number
        evidence = self._evidence(
            "Emissions decreased by 3% over the review period due to market factors.",
            score=0.80,
        )
        result = verify_claim(claim, evidence)
        self.assertEqual(result["verdict"], "High-Risk Silent Failure")

    def test_supported_claim_passes(self):
        from src.verifier import verify_claim
        claim = self._claim(
            "The Environmental Protection Agency confirmed the findings.",
            ["named_entity"],
            entities=[{"text": "The Environmental Protection Agency", "label": "ORG"}],
        )
        # Evidence above threshold and contains the named entity verbatim
        evidence = self._evidence(
            "The Environmental Protection Agency confirmed and published the annual findings.",
            score=0.90,
        )
        result = verify_claim(claim, evidence)
        self.assertEqual(result["verdict"], "Supported")

    def test_verdict_has_required_keys(self):
        from src.verifier import verify_claim
        claim = self._claim(
            "Congress passed the Clean Air Act in 2020.", ["named_entity"]
        )
        evidence = self._evidence("Congress enacted new environmental legislation.", score=0.70)
        result = verify_claim(claim, evidence)
        required = {
            "claim_id", "claim_text", "page_number", "flags",
            "evidence", "verdict", "risk_explanation",
            "rules_triggered", "confidence_score",
        }
        for key in required:
            self.assertIn(key, result, f"Missing required key '{key}' in verdict dict")


# ============================================================
# 5. TestReporter
# ============================================================

class TestReporter(unittest.TestCase):
    """Tests for src.reporter.generate_report and report_to_dataframe."""

    _MOCK_RESULTS = [
        {
            "claim_id": 1,
            "claim_text": "The policy reduced emissions by 22% over five years.",
            "page_number": 1,
            "flags": ["numeric", "causal_verb"],
            "evidence": [
                {
                    "chunk_id": 1,
                    "text": "Emissions declined during the review period.",
                    "page_number": 1,
                    "similarity_score": 0.62,
                }
            ],
            "verdict": "High-Risk Silent Failure",
            "risk_explanation": "Specific value [22.0] not found within ±10% in evidence.",
            "rules_triggered": ["numeric_match"],
            "confidence_score": 0.074,
        },
        {
            "claim_id": 2,
            "claim_text": "The EPA confirmed the findings.",
            "page_number": 1,
            "flags": ["named_entity"],
            "evidence": [
                {
                    "chunk_id": 1,
                    "text": "The EPA monitored and published the outcomes.",
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
            "claim_text": "The WHO issued a global health warning.",
            "page_number": 2,
            "flags": ["named_entity"],
            "evidence": [
                {
                    "chunk_id": 2,
                    "text": "Emissions declined during the review period.",
                    "page_number": 1,
                    "similarity_score": 0.15,
                }
            ],
            "verdict": "Unsupported",
            "risk_explanation": "Top evidence similarity 0.150 is below threshold 0.35.",
            "rules_triggered": ["similarity_threshold"],
            "confidence_score": 0.03,
        },
    ]

    def test_generate_report_structure(self):
        from src.reporter import generate_report
        report = generate_report(self._MOCK_RESULTS)
        required = {
            "total_claims",
            "verdict_counts",
            "silent_failure_rate",
            "unsupported_rate",
            "high_risk_claims",
            "summary_table",
        }
        for key in required:
            self.assertIn(key, report, f"Missing required key '{key}' in report dict")

    def test_silent_failure_rate_calculation(self):
        from src.reporter import generate_report
        report = generate_report(self._MOCK_RESULTS)
        # 1 High-Risk Silent Failure out of 3 total claims
        expected = round(1 / 3, 4)
        self.assertAlmostEqual(report["silent_failure_rate"], expected, places=4)

    def test_report_to_dataframe_returns_df(self):
        from src.reporter import generate_report, report_to_dataframe
        report = generate_report(self._MOCK_RESULTS)
        df = report_to_dataframe(report)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self._MOCK_RESULTS))


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
