# PolicyLens

A skeptical verification system for AI-generated policy analysis. PolicyLens detects silent hallucination and logical overreach in policy documents by auditing claims against their source evidence — claim by claim, not fluency by fluency.

**Live site:** https://ushakeshravya.github.io/Policylens/  
**Course:** INFO 7375 — Prompt Engineering & AI, Northeastern University

## What It Does

PolicyLens does not generate summaries. It **audits** them.

Given a policy document and an AI-generated summary, it:

1. Extracts factual claims using three passes: regex/spaCy, LLM-assisted, and multimodal (Claude Vision reads charts and tables the text layer misses)
2. Embeds the source document into a persistent FAISS index for semantic retrieval
3. Retrieves the top-3 most relevant source passages for each claim
4. Applies four deterministic verification rules (numeric match, entity consistency, causal scrutiny, similarity threshold)
5. Escalates only High-Risk and Unsupported claims to a Claude agent loop for deeper investigation
6. Produces a structured audit report with full traceability — every verdict links back to a source passage and the rule that fired

## Silent Failure

A silent failure is a claim that *looks* evidenced but isn't.

> **Source:** "Emissions declined during the review period."
> **AI summary:** "The policy reduced emissions by **22%** over five years."

The summary introduces quantified causation that was never in the source. It passes quick review because it sounds right. PolicyLens catches it.

**Silent Failure Rate** = High-Risk Silent Failures ÷ Total Claims  
MVP target: < 5%

## Project Structure

```
policylens/
├── app.py                    # Streamlit web interface (single-document + batch modes)
├── requirements.txt          # Pinned dependencies
├── .env.example              # API key template
├── README.md
├── run_tests.sh              # Test runner script
├── src/
│   ├── config.py             # Central config: model IDs, retry policy, app limits
│   ├── parser.py             # PDF text extraction (pdfplumber)
│   ├── claim_extractor.py    # 3-pass claim detection: regex + LLM + Vision
│   ├── retriever.py          # Sentence-transformer embeddings + FAISS index
│   ├── index_store.py        # Persistent FAISS cache (~/.policylens_cache/)
│   ├── verifier.py           # 4-rule deterministic verification engine
│   ├── agent.py              # Claude tool-calling agent with smart escalation
│   ├── batch.py              # Parallel multi-document batch auditing
│   └── reporter.py           # Structured report generation (JSON + DataFrame)
├── tests/
│   └── test_pipeline.py      # Unit + integration tests (5 test classes, ~20 tests)
├── eval/
│   ├── evaluate.py           # Evaluation script: accuracy, Kappa, FNR, SFDR
│   ├── sample_annotations.csv # 15 manually annotated claims (ground truth)
│   └── policylens_report.json # Example JSON report output
├── data/
│   └── sample/               # Sample policy PDFs for testing
└── docs/
    └── index.html            # GitHub Pages landing page
```

## Setup

### Prerequisites

- Python 3.9+
- pip
- [poppler](https://poppler.freedesktop.org/) — required for multimodal PDF rendering

```bash
# macOS
brew install poppler

# Ubuntu / Debian
sudo apt-get install poppler-utils

# Windows — download from https://github.com/oschwartz10612/poppler-windows
```

### Installation

```bash
git clone https://github.com/UshakeShravya/Policylens.git
cd policylens
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Environment Variables

Copy the example and add your key:

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

The API key is **optional** for basic verification — regex/spaCy extraction and the 4-rule engine run entirely locally. The key is required for LLM-assisted extraction, multimodal Vision, and the agent orchestration layer.

## Usage

### Streamlit App (recommended)

```bash
streamlit run app.py
```

Upload a PDF policy document. Optionally paste an AI-generated summary. Click **Run Audit**.

Features available in the UI:
- **Single document audit** with live progress, verdict dashboard, and JSON download
- **Batch audit** — compare up to 3 documents side-by-side with riskiest document highlighted
- Sidebar shows "Index cached — instant load" when the FAISS index for a PDF already exists

### Programmatic API

```python
from src.parser import extract_text_from_pdf
from src.claim_extractor import extract_claims_full
from src.retriever import chunk_pages, build_index
from src.agent import verify_claims_with_agent
from src.reporter import generate_report, print_report

# Parse source
pages = extract_text_from_pdf("data/sample/federal-budget-2022.pdf")

# Extract claims from a summary (or pass pages directly to self-audit)
summary_pages = [{"page_number": 1, "text": "Your AI summary here..."}]
claims = extract_claims_full("data/sample/federal-budget-2022.pdf", summary_pages)

# Build retrieval index (cached on disk after first run)
chunks = chunk_pages(pages, chunk_size=3)
index, chunks = build_index(chunks, pdf_path="data/sample/federal-budget-2022.pdf")

# Verify — agent escalates only High-Risk and Unsupported claims
results = verify_claims_with_agent(claims, index, chunks)
report = generate_report(results)
print_report(report)
```

### Batch Processing

```python
from src.batch import batch_audit, compare_reports

reports = batch_audit(
    pdf_paths=["doc1.pdf", "doc2.pdf"],
    summary_texts=["Summary for doc1...", "Summary for doc2..."],
    document_names=["EPA Report", "Federal Budget"],
)
comparison = compare_reports(reports)
print(comparison["overall_riskiest_document"])
```

### Run Evaluation

```bash
# Against mock predictions (demonstrates metric computation)
python eval/evaluate.py

# Against a real audit report
python eval/evaluate.py eval/policylens_report.json
```

## Running Tests

```bash
# All tests with verbose output
python -m unittest tests.test_pipeline -v

# Or via the helper script
bash run_tests.sh
```

Tests cover: PDF parsing, claim extraction (numeric/causal flags), FAISS retrieval, all 4 verification rules, report generation, agent loop dispatch, and cache hit/miss behavior.

## Performance

| Document | Pages | Claims extracted | Audit time (no cache) | Audit time (cached) |
|---|---|---|---|---|
| EPA FY2020 (1.6 MB) | ~80 | 8–12 | ~90 sec | ~45 sec |
| Federal Budget 2022 (3.6 MB) | ~180 | 12–20 | ~3 min | ~90 sec |

Times measured on Apple M2, 16 GB RAM. LLM extraction and agent verification dominate runtime. Audit time scales with claim count, not page count directly.

## Troubleshooting

**No claims extracted**
- The summary may lack numbers, named entities, or causal language — these trigger extraction flags. Try pasting richer text.
- If LLM extraction fails silently, check that `ANTHROPIC_API_KEY` is set and has credits (`python -c "import anthropic; print(anthropic.__version__)"`).

**`pdf2image` / poppler error**
- Multimodal extraction requires poppler. Install it (see Prerequisites above). The system falls back to text-only extraction if poppler is missing — a warning appears in the terminal.

**Agent always falls back to deterministic**
- Run `python src/agent.py` to see the full diagnostic including raw API connectivity test.
- The most common cause is a zero-credit API key — check at console.anthropic.com.

**Index not caching**
- Check write permissions on `~/.policylens_cache/`.
- The cache is keyed by MD5 of the PDF bytes. Re-uploading a re-saved copy of the same PDF creates a new cache entry.

**FAISS import error on Apple Silicon**
- Use `faiss-cpu` (already in requirements.txt). The GPU variant requires CUDA and is not needed here.

## Evaluation Metrics

| Metric | Description |
|---|---|
| Claim Classification Accuracy | % of claims whose verdict matches human annotation |
| False Negative Rate | % of truly-problematic claims wrongly labeled Supported |
| Silent Failure Detection Rate | % of High-Risk SF claims correctly identified |
| Cohen's Kappa | Inter-rater agreement vs. human annotators |

Ground truth: `eval/sample_annotations.csv` — 15 manually annotated claims drawn from EPA and Federal Budget documents.

## Data

All evaluation documents are drawn from public government reports, open-access regulatory filings, and public audit documents. No private or proprietary data is used.

## Ethical Considerations

- Policy documents may contain institutional framing bias; the system evaluates internal consistency only and avoids ideological interpretation
- Embedding retrieval may favor linguistic similarity over conceptual nuance — flagged uncertainty is always shown explicitly
- Each output includes the original claim, retrieved passage, similarity score, classification label, and source reference for full traceability
- Claims sent to the Anthropic API include document text fragments; do not use with confidential documents

## Tech Stack

| Component | Library |
|---|---|
| PDF text extraction | pdfplumber |
| NLP / NER | spaCy `en_core_web_sm` |
| Multimodal rendering | pdf2image + poppler |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector search | FAISS IndexFlatIP |
| LLM / Agent / Vision | Anthropic Claude API |
| Parallel processing | concurrent.futures |
| Interface | Streamlit |
| Evaluation | pandas, scikit-learn |
