import concurrent.futures
import json
import os
import tempfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.parser import extract_text_from_pdf
from src.claim_extractor import extract_claims_full
from src.retriever import chunk_pages, build_index
from src.agent import verify_claims_with_agent
from src.reporter import generate_report, report_to_dataframe
from src.index_store import is_cached_upload, is_cached
from src.batch import batch_audit, compare_reports

# Must be the very first Streamlit call
st.set_page_config(
    page_title="PolicyLens",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERDICT_COLORS: dict[str, str] = {
    "High-Risk Silent Failure": "background-color: #ffcccc; color: #b71c1c; font-weight: bold",
    "Unsupported":              "background-color: #ffd6b3; color: #bf360c; font-weight: bold",
    "Partially Supported":      "background-color: #fff9c4; color: #f57f17; font-weight: bold",
    "Supported":                "background-color: #c8e6c9; color: #1b5e20; font-weight: bold",
}

_VERDICT_HEX: dict[str, str] = {
    "Supported":                "#2ecc71",
    "Partially Supported":      "#f1c40f",
    "Unsupported":              "#e67e22",
    "High-Risk Silent Failure": "#ff4b4b",
}

_VERDICT_PRIORITY: dict[str, int] = {
    "High-Risk Silent Failure": 4,
    "Unsupported":              3,
    "Partially Supported":      2,
    "Supported":                1,
}

_DISPLAY_COLS = {
    "claim_id":          "ID",
    "page_number":       "Page",
    "verdict":           "Verdict",
    "confidence_score":  "Confidence",
    "detection_method":  "Method",
    "claim_text":        "Claim",
    "top_evidence":      "Top Evidence",
    "risk_explanation":  "Explanation",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_uploaded_pdf(uploaded_file) -> str:
    """Write a Streamlit UploadedFile to a named temp file; return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.getbuffer())
        return f.name


def _color_verdict(val: str) -> str:
    return _VERDICT_COLORS.get(val, "")


def _style_dataframe(df: pd.DataFrame):
    display = df[list(_DISPLAY_COLS)].rename(columns=_DISPLAY_COLS)
    # pandas >= 2.1 renamed Styler.applymap → Styler.map
    try:
        return display.style.map(_color_verdict, subset=["Verdict"])
    except AttributeError:
        return display.style.applymap(_color_verdict, subset=["Verdict"])


def _run_pipeline(source_path: str, summary_text: str):
    """
    Execute the full PolicyLens pipeline with a live progress bar.
    Returns (results, report) on success, (None, None) on any failure.
    """
    bar = st.progress(0, text="Parsing source document...")
    try:
        # 1 — Parse source
        pages = extract_text_from_pdf(source_path)
        if not pages:
            st.error(
                "No text could be extracted from the PDF. "
                "The file may be scanned or image-based."
            )
            return None, None
        bar.progress(15, text=f"Parsed {len(pages)} pages — extracting claims & building index...")

        # 2 — Determine claim source
        if summary_text.strip():
            claim_pages = [{"page_number": 1, "text": summary_text}]
            source_label = "the pasted summary"
        else:
            claim_pages = pages
            source_label = "the source document (no summary provided)"

        # 3 — Parallel: multimodal claim extraction + FAISS index build
        index_msg = (
            "Loading cached index..." if is_cached(source_path)
            else "Building retrieval index..."
        )
        bar.progress(20, text=f"Extracting claims (text + visual) · {index_msg}")

        def _build_index():
            c = chunk_pages(pages, chunk_size=3)
            return build_index(c, pdf_path=source_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            claims_future = pool.submit(extract_claims_full, source_path, claim_pages)
            index_future  = pool.submit(_build_index)
            claims        = claims_future.result()
            index, chunks = index_future.result()

        if not claims:
            st.warning(
                f"No verifiable claims found in {source_label}. "
                "Try providing text that contains numbers, named entities, or causal language."
            )
            return None, None
        bar.progress(55, text=f"Found {len(claims)} claims, {len(chunks)} index chunks — verifying claims with agent...")

        # 4 — Agent verification (parallel across claims internally)
        results = verify_claims_with_agent(claims, index, chunks)
        bar.progress(90, text="Verified — generating report...")

        # 5 — Report
        report = generate_report(results)
        bar.progress(100, text="Audit complete.")
        bar.empty()
        return results, report

    except Exception as exc:
        bar.empty()
        st.error(f"Pipeline error: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _render_charts(report: dict) -> None:
    """Render the three plotly visualisations below the summary metrics."""
    total  = report["total_claims"]
    counts = report["verdict_counts"]

    # ── 1. Confidence Gauge ────────────────────────────────────────────────
    trust_score = (
        (counts["Supported"] * 1.0 + counts["Partially Supported"] * 0.5)
        / total * 100
    ) if total else 0.0

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(trust_score, 1),
        title={"text": "Document Trust Score", "font": {"size": 15, "color": "#e8eaf0"}},
        number={"suffix": "%", "font": {"size": 34, "color": "#e8eaf0"}},
        delta={
            "reference": 80,
            "increasing": {"color": "#2ecc71"},
            "decreasing": {"color": "#ff4b4b"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#8b90a8",
                "tickfont": {"color": "#8b90a8"},
            },
            "bar": {"color": "#e8eaf0", "thickness": 0.22},
            "bgcolor": "#0f1117",
            "steps": [
                {"range": [0,  30],  "color": "rgba(255,75,75,0.22)"},
                {"range": [30, 60],  "color": "rgba(241,196,15,0.18)"},
                {"range": [60, 80],  "color": "rgba(230,126,34,0.18)"},
                {"range": [80, 100], "color": "rgba(46,204,113,0.20)"},
            ],
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#1a1d27",
        height=290,
        margin=dict(l=30, r=30, t=30, b=10),
    )

    # ── 2. Verdict Distribution (horizontal bar) ───────────────────────────
    _ordered = ["Supported", "Partially Supported", "Unsupported", "High-Risk Silent Failure"]
    fig_bar = go.Figure(go.Bar(
        x=[counts[v] for v in _ordered],
        y=_ordered,
        orientation="h",
        marker_color=[_VERDICT_HEX[v] for v in _ordered],
        text=[counts[v] for v in _ordered],
        textposition="outside",
        textfont={"color": "#e8eaf0"},
        cliponaxis=False,
    ))
    fig_bar.update_layout(
        title={"text": "Verdict Distribution", "font": {"color": "#e8eaf0", "size": 15}},
        paper_bgcolor="#1a1d27",
        plot_bgcolor="#1a1d27",
        font={"color": "#e8eaf0"},
        height=290,
        margin=dict(l=20, r=50, t=50, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, tickfont={"size": 11}),
    )

    col_gauge, col_bar = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col_bar:
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── 3. Claims Flagged per Page (vertical bar, coloured by worst verdict) ──
    df_summary = pd.DataFrame(report["summary_table"])
    if df_summary.empty or "page_number" not in df_summary.columns:
        return

    page_stats = (
        df_summary.groupby("page_number")["verdict"]
        .agg(
            count="count",
            worst=lambda s: max(s, key=lambda v: _VERDICT_PRIORITY.get(v, 0)),
        )
        .reset_index()
        .sort_values("page_number")
    )

    fig_heat = go.Figure(go.Bar(
        x=[f"Page {p}" for p in page_stats["page_number"]],
        y=page_stats["count"],
        marker_color=[_VERDICT_HEX[v] for v in page_stats["worst"]],
        text=page_stats["count"],
        textposition="auto",
        textfont={"color": "#ffffff"},
        showlegend=False,
    ))
    fig_heat.update_layout(
        title={"text": "Claims Flagged per Page", "font": {"color": "#e8eaf0", "size": 15}},
        paper_bgcolor="#1a1d27",
        plot_bgcolor="#1a1d27",
        font={"color": "#e8eaf0"},
        height=310,
        margin=dict(l=20, r=20, t=50, b=40),
        xaxis=dict(
            title="Page",
            showgrid=False,
            tickfont={"color": "#e8eaf0"},
        ),
        yaxis=dict(
            title="Claims",
            showgrid=True,
            gridcolor="#2a2d3a",
            tickfont={"color": "#e8eaf0"},
            dtick=1,
        ),
        bargap=0.35,
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ---------------------------------------------------------------------------
# Results renderer (shared between fresh run and cached state)
# ---------------------------------------------------------------------------

def _display_results(report: dict) -> None:
    total = report["total_claims"]
    counts = report["verdict_counts"]

    # — Metrics row —
    c1, c2, c3, c4 = st.columns(4)
    n_sfr = counts["High-Risk Silent Failure"]
    n_un  = counts["Unsupported"]
    with c1:
        st.metric("Total Claims Found", total)
    with c2:
        st.metric(
            "Silent Failure Rate",
            f"{report['silent_failure_rate']:.1%}",
            delta=f"{n_sfr} claim{'s' if n_sfr != 1 else ''}",
            delta_color="inverse",
        )
    with c3:
        st.metric(
            "Unsupported Rate",
            f"{report['unsupported_rate']:.1%}",
            delta=f"{n_un} claim{'s' if n_un != 1 else ''}",
            delta_color="inverse",
        )
    with c4:
        st.metric("Supported", counts["Supported"])

    # — Charts —
    _render_charts(report)

    # — Results table —
    st.markdown("### Verification Results")
    df = report_to_dataframe(report)
    st.dataframe(_style_dataframe(df), use_container_width=True, hide_index=True)

    # — High-Risk Silent Failures detail —
    high_risk = report["high_risk_claims"]
    if high_risk:
        st.markdown(f"### High-Risk Silent Failures &nbsp;({len(high_risk)})")
        st.caption(
            "These claims appear evidenced by the source but contain specific values "
            "that cannot be found or verified in any retrieved passage."
        )
        for r in high_risk:
            preview = (
                r["claim_text"][:75] + "..."
                if len(r["claim_text"]) > 75
                else r["claim_text"]
            )
            with st.expander(f"Claim {r['claim_id']} — {preview}", expanded=True):
                st.markdown("**Claim text**")
                st.info(r["claim_text"])

                if r.get("evidence"):
                    top = r["evidence"][0]
                    st.markdown(
                        f"**Top retrieved evidence** "
                        f"*(similarity: {top['similarity_score']:.3f} · p.{top['page_number']})*"
                    )
                    st.success(top["text"])

                st.markdown("**Risk explanation**")
                st.warning(r["risk_explanation"])

                left, right = st.columns(2)
                rules = r["rules_triggered"]
                left.markdown(
                    f"**Rules triggered:** "
                    + ("`" + "`, `".join(rules) + "`" if rules else "_none_")
                )
                right.markdown(f"**Confidence score:** `{r['confidence_score']:.4f}`")

                agent_notes = r.get("agent_notes", "")
                if agent_notes:
                    st.markdown("**Agent notes**")
                    st.info(agent_notes)

    # — Download —
    st.divider()
    st.download_button(
        label="Download Full Report (JSON)",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="policylens_report.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def show_audit() -> None:
    st.title("PolicyLens")
    st.caption("Claim Verification for Policy Documents")

    # Sidebar — inputs
    with st.sidebar:
        st.header("Source Document")
        source_pdf = st.file_uploader(
            "Upload the policy document (PDF)",
            type=["pdf"],
            key="source_pdf",
        )
        if source_pdf is not None:
            _cached = is_cached_upload(source_pdf.getbuffer())
            if _cached:
                st.success("Index cached — instant load")
            else:
                st.caption("Index will be built on first run")
        st.header("AI Summary to Verify")
        summary_text = st.text_area(
            "Paste an AI-generated summary",
            height=200,
            placeholder=(
                "Paste the AI-generated summary you want to verify against "
                "the source document.\n\n"
                "Leave blank to extract and self-verify claims directly from "
                "the source document."
            ),
            key="summary_text",
        )
        st.caption("— or upload a summary PDF —")
        summary_pdf = st.file_uploader(
            "Upload summary as PDF (optional)",
            type=["pdf"],
            key="summary_pdf",
        )

        st.divider()
        run_clicked = st.button("Run Audit", type="primary", use_container_width=True)

    # Handle button click
    if run_clicked:
        if source_pdf is None:
            st.error("Please upload a source policy document before running the audit.")
        else:
            source_path = _save_uploaded_pdf(source_pdf)
            final_summary = ""

            # Resolve summary: uploaded PDF takes precedence over pasted text
            if summary_pdf is not None:
                sum_path = _save_uploaded_pdf(summary_pdf)
                try:
                    sum_pages = extract_text_from_pdf(sum_path)
                    final_summary = " ".join(p["text"] for p in sum_pages)
                except Exception as exc:
                    st.error(f"Could not parse summary PDF: {exc}")
                    os.unlink(source_path)
                    st.stop()
                finally:
                    os.unlink(sum_path)
            else:
                final_summary = summary_text

            try:
                with st.spinner("Running full audit (multimodal + agent)..."):
                    results, report = _run_pipeline(source_path, final_summary)
                if results is not None:
                    st.session_state.results = results
                    st.session_state.report = report
            finally:
                os.unlink(source_path)

    # Render results (from this run or cached from a previous run in this session)
    if st.session_state.report is not None:
        _display_results(st.session_state.report)
    elif not run_clicked:
        st.info(
            "Upload a source policy document in the sidebar, optionally paste an "
            "AI-generated summary, then click **Run Audit**."
        )


def _display_batch_results(reports: list[dict], comparison: dict) -> None:
    riskiest = comparison["overall_riskiest_document"]
    safest   = comparison["overall_safest_document"]

    # Top metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents Audited", len(reports))
    c2.metric("Total Claims", comparison["total_claims_across_all"])
    c3.metric("Total High-Risk Failures", comparison["total_high_risk_across_all"])

    # Comparison table
    st.markdown("### Comparison Table")
    if riskiest:
        st.error(f"Riskiest: **{riskiest}** — highest silent failure rate")
    if safest and safest != riskiest:
        st.success(f"Safest: **{safest}** — lowest silent failure rate")

    valid_docs = [d for d in comparison["documents"] if not d["error"]]
    if valid_docs:
        comp_df = pd.DataFrame([
            {
                "Document":            d["document_name"],
                "Claims":              d["total_claims"],
                "High-Risk":           d["high_risk_count"],
                "Silent Failure Rate": f"{d['silent_failure_rate']:.1%}",
                "Unsupported Rate":    f"{d['unsupported_rate']:.1%}",
            }
            for d in valid_docs
        ])

        def _highlight_riskiest(row):
            if row["Document"] == riskiest:
                return [
                    "background-color: #ffcccc; color: #b71c1c; font-weight: bold"
                ] * len(row)
            return [""] * len(row)

        st.dataframe(
            comp_df.style.apply(_highlight_riskiest, axis=1),
            use_container_width=True,
            hide_index=True,
        )

    for d in comparison["documents"]:
        if d["error"]:
            st.error(f"{d['document_name']}: {d['error']}")

    # Per-document detail
    st.markdown("### Per-Document Detail")
    for i, report in enumerate(reports):
        if report.get("error"):
            continue
        doc_name = report.get("document_name", f"Document {i + 1}")
        is_riskiest = doc_name == riskiest
        counts = report["verdict_counts"]

        with st.expander(
            f"{'[HIGH RISK]  ' if is_riskiest else ''}{doc_name}",
            expanded=is_riskiest,
        ):
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Supported",          counts["Supported"])
            mc2.metric("Partially Supported", counts["Partially Supported"])
            mc3.metric("Unsupported",         counts["Unsupported"])
            mc4.metric("High-Risk",           counts["High-Risk Silent Failure"])

            if report["summary_table"]:
                df_detail = report_to_dataframe(report)
                st.dataframe(
                    _style_dataframe(df_detail),
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                label=f"Download {doc_name} Report (JSON)",
                data=json.dumps(report, indent=2, ensure_ascii=False),
                file_name=f"policylens_{doc_name}.json",
                mime="application/json",
                key=f"batch_dl_{i}",
            )


def show_batch() -> None:
    st.title("Batch Audit")
    st.caption("Audit and compare multiple policy documents in one run")

    # Input grid — 3 columns, one document per column
    cols = st.columns(3)
    uploaded_pairs: list[tuple] = []  # (pdf_file, summary_text)

    for i, col in enumerate(cols, 1):
        with col:
            st.subheader(f"Document {i}")
            pdf = st.file_uploader(
                f"PDF {i}",
                type=["pdf"],
                key=f"batch_pdf_{i}",
                label_visibility="collapsed",
            )
            summary = st.text_area(
                f"Summary {i}",
                height=160,
                placeholder="Paste AI summary (optional — leave blank to self-audit)",
                key=f"batch_summary_{i}",
                label_visibility="collapsed",
            )
            if pdf is not None:
                uploaded_pairs.append((pdf, summary))

    st.divider()
    run_clicked = st.button(
        "Run Batch Audit",
        type="primary",
        disabled=len(uploaded_pairs) == 0,
        use_container_width=True,
    )

    if run_clicked and uploaded_pairs:
        temp_paths: list[str] = []
        try:
            original_names = [pdf.name for pdf, _ in uploaded_pairs]
            for pdf, _ in uploaded_pairs:
                temp_paths.append(_save_uploaded_pdf(pdf))
            final_summaries = [s for _, s in uploaded_pairs]

            with st.spinner(f"Auditing {len(uploaded_pairs)} document(s) in parallel..."):
                reports = batch_audit(temp_paths, final_summaries, document_names=original_names)
                comparison = compare_reports(reports)

            st.session_state.batch_reports    = reports
            st.session_state.batch_comparison = comparison
        except Exception as exc:
            st.error(f"Batch audit error: {exc}")
        finally:
            for p in temp_paths:
                try:
                    os.unlink(p)
                except Exception:
                    pass

    if st.session_state.batch_comparison is not None:
        _display_batch_results(
            st.session_state.batch_reports,
            st.session_state.batch_comparison,
        )
    elif not run_clicked:
        st.info(
            "Upload up to 3 policy PDFs above (and optionally paste an AI-generated "
            "summary for each), then click **Run Batch Audit**."
        )


def show_about() -> None:
    st.title("About PolicyLens")
    st.markdown("""
PolicyLens is a skeptical verification system for AI-generated policy analysis.
It does not generate summaries — it **audits** them, claim by claim.

---

### The Problem

Modern AI systems summarize complex policy documents with confidence and fluency.
But **fluency is not verification**.

> **Source document:** *"Emissions declined during the review period, though macroeconomic factors may have contributed."*
>
> **AI summary:** *"The policy reduced emissions by 22% over five years."*

The summary introduces quantified causation that was never in the source.
It passes quick review because it *sounds* right. PolicyLens catches it.

This failure mode — a claim that *looks* verified but isn't — is called a **Silent Failure**.

---

### How It Works

**Step 1 — Claim Extraction**
Sentences in the AI summary are scanned for factual assertions: anything containing
numbers or percentages, named entities (organisations, places, people, laws),
causal verbs (*reduced*, *generated*, *led to*), or superlative language (*highest*, *never*, *all*).

**Step 2 — Retrieval Index**
The source document is chunked into overlapping 3-sentence windows and embedded
using `sentence-transformers` (`all-MiniLM-L6-v2`). Embeddings are stored in a
FAISS `IndexFlatIP` with L2 normalisation for cosine similarity search.

**Step 3 — Evidence Retrieval**
For each extracted claim, the top-3 most semantically similar source passages
are retrieved as candidate evidence.

**Step 4 — Deterministic Verification**
Four rules are applied in order; the highest-priority verdict wins:

| Rule | Checks | Failure verdict |
|---|---|---|
| Similarity threshold | Top evidence score ≥ 0.35 | Unsupported |
| Numeric match | Claim numbers present in evidence ±10% | High-Risk Silent Failure |
| Causal scrutiny | Causal claim backed by causal evidence | Partially Supported |
| Entity consistency | Key entities (ORG, GPE, LAW…) in evidence | Partially Supported |

---

### Verdict Labels

| Label | Meaning |
|---|---|
| **Supported** | Claim is grounded in evidence across all applicable rules |
| **Partially Supported** | Evidence exists but causal language or entities are misaligned |
| **Unsupported** | No relevant source passage found (similarity below threshold) |
| **High-Risk Silent Failure** | Claim *appears* evidenced but specific values cannot be verified |

**Silent Failure Rate** = High-Risk Silent Failures ÷ Total Claims
Target for the MVP: < 5 %

---

### Tech Stack

`pdfplumber` · `spaCy en_core_web_sm` · `sentence-transformers` · `FAISS` · `Streamlit`

---

### Source Code

[GitHub Repository](https://github.com/UshakeShravya/Policylens)
""")


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

# Session state — initialise once per browser session
for _key in ("results", "report", "batch_reports", "batch_comparison"):
    if _key not in st.session_state:
        st.session_state[_key] = None

# Sidebar navigation (always visible, rendered before page functions add their own content)
with st.sidebar:
    st.markdown("## PolicyLens")
    _mode = st.radio(
        "Navigation",
        ["Audit Document", "Batch Audit", "About"],
        label_visibility="collapsed",
    )
    st.divider()

if _mode == "About":
    show_about()
elif _mode == "Batch Audit":
    show_batch()
else:
    show_audit()
