# PolicyLens

A skeptical verification system for AI-generated policy analysis. PolicyLens detects silent hallucination and logical overreach in policy documents by auditing claims against their source evidence — claim by claim, not fluency by fluency.

## What It Does

PolicyLens does not generate summaries. It audits them.

Given a policy document, it:
1. Extracts factual claims (especially those with numbers, named entities, or causal language)
2. Retrieves the most relevant source passages using a local RAG pipeline (sentence-transformers + FAISS)
3. Applies deterministic verification rules (numeric match, entity consistency, causal scrutiny, similarity threshold)
4. Classifies each claim as **Supported**, **Partially Supported**, **Unsupported**, or **High-Risk Silent Failure**
5. Produces a structured audit report with full traceability

## Silent Failure

A silent failure is an output that looks correct but is logically or factually broken.

> Original: "Emissions declined during the review period."  
> Summary: "The policy reduced emissions by 22%."

The second statement introduces quantified causation never present in the source. It passes quick review because it sounds right. PolicyLens catches it.

**Silent Failure Rate = False Negatives / Total Evaluated Claims**

The MVP target is < 5%.

## Project Structure

```
policylens/
├── app.py                  # Streamlit web interface
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── parser.py           # PDF text extraction (pdfplumber)
│   ├── claim_extractor.py  # Claim detection and segmentation
│   ├── retriever.py        # Embedding-based RAG retrieval (FAISS)
│   ├── verifier.py         # Deterministic verification rule engine
│   └── reporter.py         # Structured audit report generation
├── data/
│   └── sample/             # Sample policy documents for testing
└── eval/                   # Ground truth annotations and evaluation scripts
```

## Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/yourusername/policylens.git
cd policylens
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Environment Variables

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here   # optional — only needed for LLM-assisted features
```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

Upload a PDF policy document. The system will extract claims, retrieve evidence, and produce a verification report.

### Run the Pipeline Programmatically

```python
from src.parser import extract_text_from_pdf
from src.claim_extractor import extract_claims
from src.retriever import build_index, retrieve_evidence
from src.verifier import verify_claims
from src.reporter import generate_report

pages = extract_text_from_pdf("data/sample/policy_report.pdf")
claims = extract_claims(pages)
index, chunks = build_index(pages)
results = verify_claims(claims, index, chunks)
report = generate_report(results)
```

## Evaluation Metrics

| Metric | Description |
|---|---|
| Claim Classification Accuracy | % of claims correctly labeled vs. human annotations |
| Retrieval Recall | % of relevant source passages retrieved |
| Evidence Grounding Precision | % of retrieved passages that are genuinely relevant |
| False Negative Rate | % of unsupported claims wrongly classified as supported |
| Human Audit Agreement Score | Cohen's Kappa vs. manual annotations |

## Data

All evaluation documents are drawn from public government reports, open-access regulatory filings, and public audit documents. No private or proprietary data is used.

## Ethical Considerations

- Policy documents may contain institutional framing bias; the system evaluates internal consistency only and avoids ideological interpretation
- Embedding retrieval may favor linguistic similarity over conceptual nuance — flagged uncertainty is always shown explicitly
- Each output includes the original claim, retrieved passage, similarity score, classification label, and source reference for full traceability

## Tech Stack

- **PDF Parsing**: pdfplumber
- **NLP / NER**: spaCy
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector Search**: FAISS
- **LLM (optional)**: Anthropic Claude API
- **Interface**: Streamlit
- **Evaluation**: pandas, numpy
