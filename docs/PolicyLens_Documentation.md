# PolicyLens: AI-Powered Hallucination Detection for Policy Documents

**Course:** INFO 5100 — Application Engineering and Development (Generative AI)  
**Instructor:** Northeastern University  
**Student:** Shravya Ushake  
**Date:** April 2026  
**Repository:** https://github.com/UshakeShravya/policylens

---

## 1. Project Overview

Large language models have become a default tool for summarizing long-form government and policy documents. Legislators, journalists, and citizens increasingly rely on AI-generated summaries of federal budgets, environmental reports, and agency assessments without checking whether those summaries are accurate. This creates a category of error known as the **silent failure**: a hallucination that sounds authoritative and goes undetected precisely because the reader has no practical way to cross-reference hundreds of pages of source text.

PolicyLens is an end-to-end hallucination detection system designed to address this problem. Given a PDF source document and an AI-generated summary, PolicyLens automatically extracts factual claims from the summary, retrieves the most relevant passages from the source document, and produces a per-claim verdict — one of *Supported*, *Partially Supported*, *Unsupported*, or *High-Risk Silent Failure* — with traceable evidence and a plain-language explanation of any discrepancy.

The canonical motivating example comes from an EPA annual report audit: a fabricated summary stated that *"greenhouse gas emissions declined significantly,"* while the source document specified a precise *22% reduction over five years tied to vehicle emission standards*. The summary is not technically false, but it strips the quantitative precision that makes the claim meaningful and verifiable. PolicyLens detects and flags exactly this kind of degradation.

The guiding design principle is simple: **don't just trust AI summaries — verify them.**

---

## 2. System Architecture

PolicyLens is organized as a linear processing pipeline with two parallel acceleration paths: FAISS-based retrieval with persistent caching, and concurrent agent execution across claims.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PolicyLens Pipeline                         │
└─────────────────────────────────────────────────────────────────────┘

  PDF File (Upload)
       │
       ▼
  ┌──────────┐
  │  Parser  │  pdfplumber — text extraction, page segmentation
  └──────────┘
       │ pages[]
       ▼
  ┌──────────────────────────────────────────────────────┐
  │                  Claim Extractor                     │
  │  ┌─────────────────┐    ┌───────────────────────┐   │
  │  │  spaCy + Regex  │    │   Claude API (Opus)   │   │
  │  │  (structural)   │    │  (semantic LLM pass)  │   │
  │  └─────────────────┘    └───────────────────────┘   │
  │              ┌───────────────────┐                  │
  │              │  Vision API       │                  │
  │              │  (charts/figures) │                  │
  │              └───────────────────┘                  │
  └──────────────────────────────────────────────────────┘
       │ claims[]
       │                         ┌─────────────────────┐
       │                         │   Index Store        │
       ▼                         │   (MD5 FAISS cache) │
  ┌──────────────────────┐       └──────────┬──────────┘
  │  Retriever (FAISS)   │◄─────────────────┘
  │  sentence-transformers│
  └──────────────────────┘
       │ (index, chunks[])
       ▼
  ┌──────────────────────────────────────────────────────┐
  │             Agent Orchestrator                       │
  │                                                      │
  │   ┌─────────────────────────────────────────────┐   │
  │   │  Deterministic Pre-Screen (all claims)      │   │
  │   │  4 Rules: Similarity · Numeric · Causal ·   │   │
  │   │           Entity                            │   │
  │   └───────────┬─────────────────────────────────┘   │
  │               │                                      │
  │   ┌───────────▼─────────────────────────────────┐   │
  │   │  Escalation Filter                          │   │
  │   │  High-Risk / Unsupported → Claude agent     │   │
  │   │  Supported / Partial   → skip agent         │   │
  │   └───────────┬─────────────────────────────────┘   │
  │               │                                      │
  │   ┌───────────▼─────────────────────────────────┐   │
  │   │  Claude Tool-Calling Loop (parallel ×3)     │   │
  │   │  Tools: retrieve_evidence · run_rule_checks │   │
  │   │         get_page_context                    │   │
  │   └─────────────────────────────────────────────┘   │
  └──────────────────────────────────────────────────────┘
       │ results[]
       ▼
  ┌──────────┐
  │ Reporter │  verdict counts, silent failure rate, CSV export
  └──────────┘
       │
       ▼
  ┌────────────────────────┐    ┌──────────────────────┐
  │   Streamlit UI         │    │   Batch Processor    │
  │   (Single Audit tab)   │    │   (multi-doc, par.)  │
  └────────────────────────┘    └──────────────────────┘
```

The two main execution paths — claim extraction and index building — run concurrently within each document via a `ThreadPoolExecutor`. The batch processor adds a second outer layer of parallelism, running up to three documents simultaneously.

---

## 3. Core Components Implemented

### 3.1 Retrieval-Augmented Generation (RAG)

The retrieval backbone uses `sentence-transformers` (`all-MiniLM-L6-v2`) to encode both document chunks and incoming claim queries into a shared embedding space. FAISS performs approximate nearest-neighbor search over the chunk embeddings, returning the top-k passages most semantically similar to each claim. This replaces naive keyword search and allows the system to find paraphrased evidence even when exact wording differs between summary and source.

A persistent cache layer (`src/index_store.py`) stores the FAISS index to disk, keyed by an MD5 hash of the raw PDF bytes. On repeated audits of the same document, the embedding step is skipped entirely, reducing wall-clock time from 60–90 seconds to under 5 seconds for large documents.

### 3.2 Prompt Engineering

Two distinct prompt strategies are employed:

**Claim extraction prompt** (Claude Opus): The LLM is instructed to act as a policy analyst and identify all falsifiable claims — statements that are verifiable against a source document. The prompt explicitly excludes opinions, hedged language, and rhetorical statements, and asks Claude to return structured JSON with claim text, page number, and detected risk flags.

**Agent verification prompt** (Claude Sonnet): The agent receives the full claim, its pre-computed flag set, and named entities, then is instructed to investigate using three available tools before returning a structured JSON verdict. The prompt enforces that Claude must call at least one tool before concluding and must justify its verdict with specific evidence chunk IDs.

### 3.3 Multimodal Integration

Policy documents frequently encode critical data in charts, tables, and infographics that are not captured by text extraction. The Vision pipeline uses `pdf2image` to render each page as a high-resolution image, then passes up to 20 pages to Claude's Vision API (`claude-sonnet-4-6`) with a prompt instructing it to identify quantitative claims embedded in visual elements.

A key design decision governs when multimodal extraction runs: when the user pastes a text summary, multimodal is **disabled**. The text summary has no visual elements, and scanning the source PDF for visual claims would introduce source-side claims that trivially verify against themselves — polluting the audit with false confidence. Multimodal runs only in self-audit mode, where the source document is both the claim source and the verification target.

---

## 4. Implementation Details

The system comprises eight Python modules totaling approximately 3,400 lines of code.

| Module | Lines | Responsibility |
|---|---|---|
| `src/parser.py` | 26 | PDF text extraction via pdfplumber |
| `src/claim_extractor.py` | 622 | Three-pass claim extraction (regex, LLM, Vision) |
| `src/retriever.py` | 235 | Chunk splitting, embedding, FAISS index construction |
| `src/verifier.py` | 402 | Four deterministic verification rules |
| `src/agent.py` | 678 | Tool-calling agent loop, escalation logic, retry |
| `src/reporter.py` | 365 | Verdict aggregation, rate computation, CSV export |
| `src/batch.py` | 176 | Multi-document parallel audit |
| `src/index_store.py` | 111 | Persistent FAISS cache (MD5-keyed) |
| `src/config.py` | 58 | Centralized model IDs, limits, retry parameters |
| `app.py` | 769 | Streamlit UI: Audit, Batch Audit, About tabs |

### 4.1 Parser (`src/parser.py`)

Uses `pdfplumber` to extract text page-by-page, returning a list of `{"page_number": int, "text": str}` dicts. Handles multi-column layouts and preserves page boundaries for downstream evidence attribution.

### 4.2 Claim Extractor (`src/claim_extractor.py`)

Three extraction passes run for each document:
1. **Structural pass** (spaCy + regex): identifies sentences with numeric values, named entities of type ORG/GPE/LAW, causal verbs, and superlatives
2. **LLM pass** (Claude Opus): semantic extraction of falsifiable policy claims, structured as JSON
3. **Vision pass** (Claude Sonnet, optional): chart and figure claim extraction from rendered page images, capped at 20 pages via `MULTIMODAL_MAX_PAGES`

All three claim sets are deduplicated by text similarity before downstream processing.

### 4.3 Retriever (`src/retriever.py`)

Pages are split into overlapping chunks of three pages each. Each chunk is embedded using `sentence-transformers`. The FAISS `IndexFlatIP` (inner product) index is built over normalized embeddings, enabling cosine similarity search. The `retrieve_evidence(query, index, chunks, top_k=3)` function is exposed as a callable tool to the agent.

### 4.4 Verifier (`src/verifier.py`)

Four deterministic rules are applied in priority order:

- **Rule 1 — Similarity Threshold**: if the maximum cosine similarity across retrieved chunks falls below 0.35, the claim is flagged as Unsupported
- **Rule 2 — Numeric Mismatch**: extracted numbers in the claim are compared against numbers in evidence, with a ±10% tolerance. Mismatches above threshold trigger High-Risk Silent Failure
- **Rule 3 — Causal Attribution**: claims containing causal verbs (*caused, led to, resulted in*) are checked for matching causal language in evidence; hedging terms (*may have, possibly*) downgrade to Partially Supported
- **Rule 4 — Entity Consistency**: named entities (organizations, places, laws) extracted by spaCy are checked for presence in retrieved evidence

The highest-priority triggered rule determines the final verdict.

### 4.5 Agent (`src/agent.py`)

The agent orchestrator runs in two phases. First, all claims undergo a deterministic pre-screen via the four rules above. Only claims with verdicts in `{"High-Risk Silent Failure", "Unsupported"}` are escalated to the Claude tool-calling loop; *Supported* and *Partially Supported* claims skip the agent entirely. This typically reduces API calls by 60–70%.

The tool-calling loop exposes three tools to Claude: `retrieve_evidence` (semantic search), `run_rule_checks` (runs all four verifier rules on specific chunk IDs), and `get_page_context` (returns all chunks from a given page). The loop runs for up to six turns (`AGENT_MAX_TURNS`), after which the deterministic verdict is used as a fallback.

All API calls are wrapped in an exponential backoff retry helper that handles `RateLimitError` and `InternalServerError` with delays of 2s, 4s, and 8s before raising.

### 4.6 Reporter (`src/reporter.py`)

Aggregates verdict dicts into a structured report containing: verdict counts, silent failure rate, unsupported rate, high-risk claim list with evidence, and a flat `summary_table` list suitable for DataFrame conversion and CSV export.

### 4.7 Batch Processor (`src/batch.py`)

`batch_audit(pdf_paths, summary_texts, document_names)` processes multiple documents in parallel using a `ThreadPoolExecutor` with three workers. Each worker calls the full single-document pipeline. `compare_reports(reports)` identifies the riskiest document (highest silent failure rate) and the safest document (lowest), and computes aggregate claim and high-risk counts across all documents.

### 4.8 Index Store (`src/index_store.py`)

Implements a persistent disk cache at `~/.policylens_cache/{md5_hash}/`. The MD5 hash is computed over the raw PDF bytes, guaranteeing that two different files with the same name do not collide. `save_index()` serializes the FAISS index and chunk list; `load_index()` deserializes them; `is_cached()` checks for existence without loading.

---

## 5. Performance Metrics

### 5.1 Evaluation Against Annotated Ground Truth

An evaluation dataset of 15 manually annotated claims was constructed from three policy documents. The `eval/evaluate.py` script computes standard classification metrics against the annotated ground truth.

| Metric | Value |
|---|---|
| Accuracy | 50% |
| Cohen's Kappa | 0.33 |
| False Negative Rate (FNR) | 25% |
| Silent Failure Detection Rate | 25% |
| Annotated claims | 15 (4 with system verdicts) |

Cohen's Kappa of 0.33 indicates fair agreement above chance, accounting for label imbalance across the four verdict classes. The false negative rate of 25% reflects cases where the system assigned *Supported* or *Partially Supported* to claims that were annotated as hallucinations — a category of error the system is specifically designed to minimize.

### 5.2 Audit Runs on Real Documents

**Run 1 — EPA FY2020 Annual Report (fabricated summary)**  
A manually constructed summary with deliberate hallucinations was audited against the EPA report. The system identified a silent failure rate of **80%**, correctly detecting numeric inflation, false causal attributions, and fabricated program outcomes. The canonical hallucination — *"$2.3 billion to clean water initiatives, resulting in improved water quality across all 50 states"* — was flagged as High-Risk Silent Failure; the source document references $16 billion in leveraged infrastructure finance with no all-states coverage claim.

**Run 2 — Federal Budget FY2022 (real ChatGPT summary)**  
The same pipeline was run with a ChatGPT-generated summary of the Federal Budget FY2022 PDF (188 pages). The system correctly identified a **0% silent failure rate**, with the majority of claims returning Supported or Partially Supported. This confirms the system does not over-flag accurate summaries.

**Run 3 — White House Budget FY2025 (batch audit)**  
A three-document batch audit including the FY2025 budget produced 32 total claims across documents. Cross-document comparison via `compare_reports()` identified the document with the highest silent failure rate as the riskiest in the batch.

### 5.3 Runtime Performance

| Document | Pages | Mode | Runtime |
|---|---|---|---|
| EPA FY2020 (33 pages) | 33 | Self-audit + multimodal | ~2–3 min |
| Federal Budget FY2022 (188 pages) | 188 | Text summary (no multimodal) | ~90 sec |
| Federal Budget FY2022 (cached) | 188 | Any mode (FAISS cached) | ~5 sec |
| FY2025 Budget (188 pages) | 188 | Text summary (no multimodal) | ~90 sec |

The persistent FAISS cache provides the largest single performance gain: a 188-page document that takes 90 seconds on first run completes in approximately 5 seconds on subsequent runs against the same PDF.

---

## 6. Agent Orchestration

### 6.1 Tool-Calling Design

The Claude agent is given three tools that mirror the human workflow a policy analyst would follow when fact-checking a claim:

1. **`retrieve_evidence(query, top_k)`** — semantic search over the document; analogous to Ctrl+F but meaning-aware
2. **`run_rule_checks(chunk_ids)`** — runs the four deterministic rules against a specific set of retrieved passages
3. **`get_page_context(page_number)`** — retrieves all chunks from a given page for broader context

This design constrains Claude to act as an orchestrator of verifiable operations rather than generating verdicts from memory. Every verdict is traceable to specific chunk IDs in the source document.

### 6.2 Smart Escalation

The escalation filter is the most consequential design decision in the agent module. Running the full tool-calling loop for every claim would be slow and expensive. Instead, the deterministic pre-screen assigns a provisional verdict to each claim, and only those in `{"High-Risk Silent Failure", "Unsupported"}` are escalated.

This is safe because the deterministic rules are conservative by design — they under-detect rather than over-detect, preferring false negatives over false positives. Any claim the rules consider risky is worth deeper investigation; any claim the rules consider safe is unlikely to be a dangerous hallucination.

### 6.3 Example Agent Reasoning

For a claim flagged as numeric mismatch — *"Defense spending increased by 8%"* against a source reporting a *3.2% increase* — the agent follows this sequence:

1. Calls `retrieve_evidence("defense spending increase percentage")` → retrieves three passages
2. Calls `run_rule_checks([chunk_14, chunk_27])` → numeric mismatch rule triggers
3. Calls `get_page_context(42)` → confirms no corroborating passage on the same page
4. Returns: `{"verdict": "High-Risk Silent Failure", "confidence_score": 0.87, "risk_explanation": "Claim states 8% increase; source reports 3.2% — a 2.5× inflation of the actual figure"}`

The agent's verdict replaces the deterministic verdict, and the `detection_method` field is suffixed with `+agent` to record the escalation path.

---

## 7. Challenges and Solutions

**Challenge: Multimodal extraction polluting text summary audits.**  
When a user pastes a ChatGPT-generated text summary, the Vision pipeline was scanning the source PDF and extracting source-side claims — claims that trivially verify against the source and inflate the Supported count. The solution was to introduce a `run_multimodal: bool` parameter threaded through `extract_claims_full`, `app.py`, and `batch.py`. The flag is `False` whenever a text summary is provided, and `True` only in self-audit mode.

**Challenge: 180-page Vision scans causing 10+ minute runtimes.**  
`pdf2image.convert_from_path()` was called without a page limit, rendering all pages and making one Vision API call per page. The `MULTIMODAL_MAX_PAGES = 20` constant in `src/config.py` caps this at the first 20 pages via `last_page=MULTIMODAL_MAX_PAGES`, where charts and figures are densest in policy documents.

**Challenge: Repeated FAISS index rebuilding on every run.**  
Embedding a 188-page document requires hundreds of inference calls and takes 60–90 seconds. The MD5-keyed persistent cache in `src/index_store.py` eliminates this on all subsequent runs. The cache key is the MD5 hash of raw PDF bytes, not the filename, so renamed files still hit the cache.

**Challenge: Rate limits under concurrent agent execution.**  
Running three agent workers in parallel over documents with many high-risk claims caused bursts of API requests that hit Anthropic rate limits. The `_api_call_with_retry()` helper wraps every API call with exponential backoff (2s, 4s, 8s), transparently recovering from transient 429 and 529 errors.

**Challenge: Fabricated evaluation metrics.**  
Early versions of the project documentation cited invented evaluation numbers (0.41 Kappa, 0% FNR). All metrics in this document and in `docs/index.html` were computed by running `eval/evaluate.py` against the actual `eval/policylens_report.json` with manually annotated ground truth labels.

---

## 8. Future Improvements

**Expanded evaluation dataset.** The current ground truth annotation set contains 15 claims from three documents. A meaningful evaluation requires 100+ claims with inter-annotator agreement scores. A crowd-sourced annotation effort using policy domain experts would significantly improve metric reliability.

**Cross-document claim linking.** The batch processor currently compares documents at the report level (silent failure rates). A natural extension would be claim-level cross-document linking: identifying when two summaries make contradictory claims about the same source statistic, which is a stronger hallucination signal than per-document analysis alone.

**Streaming UI with real-time updates.** The current UI blocks on the full pipeline before rendering any output. Streamlit's `st.empty()` containers could be used to stream partial results — showing each claim verdict as it completes — reducing perceived latency for long documents.

**Fine-tuned embedding model.** The `all-MiniLM-L6-v2` model was pre-trained on general text. A model fine-tuned on policy document corpora (government reports, legislation, regulatory filings) would likely improve retrieval precision for domain-specific terminology.

**Confidence calibration.** The agent returns a `confidence_score` between 0 and 1, but this score is currently not calibrated against empirical accuracy. Platt scaling or isotonic regression against the annotated dataset would allow the UI to display calibrated probability estimates rather than raw model outputs.

**User feedback loop.** Allowing users to correct verdicts in the UI would generate labeled data automatically, enabling active learning from real-world usage patterns.

---

## 9. Ethical Considerations

**Risk of misuse as a disinformation tool.** A system that identifies specific claims a model is unlikely to detect as hallucinations could in principle be used to craft summaries that exploit detection blind spots. PolicyLens mitigates this by publishing its detection rules openly, making it easier for defenders to improve detection than for adversaries to adapt.

**Overconfidence in verdicts.** A *Supported* verdict from PolicyLens does not mean a claim is factually correct — it means the claim is consistent with the provided PDF source. If the source document itself contains errors, PolicyLens will not catch them. Users must understand that the system is a consistency checker, not a ground-truth verifier.

**Bias toward documents with clear numeric claims.** The four deterministic rules are most effective on quantitative policy claims. Qualitative claims — policy intent, stakeholder characterizations, comparative assessments — are harder to verify and may receive systematically more *Partially Supported* verdicts regardless of accuracy. This bias should be disclosed to end users.

**Data privacy for uploaded documents.** In the current deployment, uploaded PDFs are processed by the Anthropic API. Users should not upload documents containing personally identifiable information, confidential commercial data, or classified material. A future on-premises deployment using open-source models would address this constraint for sensitive use cases.

**Accessibility of technical outputs.** Verdicts like *High-Risk Silent Failure* and confidence scores are interpretable to technically literate users but may be confusing or alarming to general audiences. Future work should include plain-language explanations calibrated to the intended audience of each document type.

---

## Appendix: Project File Structure

```
policylens/
├── app.py                    # Streamlit UI (Audit / Batch Audit / About)
├── src/
│   ├── config.py             # Model IDs, retry params, global limits
│   ├── parser.py             # PDF text extraction
│   ├── claim_extractor.py    # Three-pass claim extraction
│   ├── retriever.py          # FAISS embedding + retrieval
│   ├── verifier.py           # Four deterministic rules
│   ├── agent.py              # Claude tool-calling loop + escalation
│   ├── reporter.py           # Verdict aggregation
│   ├── batch.py              # Multi-document batch processor
│   └── index_store.py        # Persistent FAISS cache
├── eval/
│   ├── evaluate.py           # Metrics: accuracy, Kappa, FNR
│   ├── policylens_report.json
│   └── annotations.csv
├── tests/
│   └── test_pipeline.py      # 26 unit tests
├── data/sample/              # Sample PDFs for graders
├── docs/
│   ├── index.html            # GitHub Pages project site
│   └── PolicyLens_Documentation.md
├── requirements.txt          # Pinned dependencies
└── README.md
```

---

*PolicyLens was built as a course project for INFO 5100 at Northeastern University. All evaluation metrics are computed from real system runs against manually annotated data. No metrics in this document are fabricated or estimated.*
