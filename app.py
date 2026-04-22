import json
import os
import tempfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.parser import extract_text_from_pdf
from src.claim_extractor import extract_claims, extract_claims_full
from src.retriever import chunk_pages, build_index
from src.verifier import verify_claims
from src.reporter import generate_report, report_to_dataframe

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


def _run_pipeline(source_path: str, summary_text: str, multimodal: bool = False):
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
        bar.progress(20, text=f"Parsed {len(pages)} pages — extracting claims...")

        # 2 — Decide what to extract claims FROM
        if summary_text.strip():
            claim_pages = [{"page_number": 1, "text": summary_text}]
            source_label = "the pasted summary"
        else:
            claim_pages = pages
            source_label = "the source document (no summary provided)"

        if multimodal:
            bar.progress(25, text="Extracting claims (text + visual)...")
            claims = extract_claims_full(source_path, claim_pages)
        else:
            claims = extract_claims(claim_pages)
        if not claims:
            st.warning(
                f"No verifiable claims found in {source_label}. "
                "Try providing text that contains numbers, named entities, or causal language."
            )
            return None, None
        bar.progress(40, text=f"Found {len(claims)} claims — building retrieval index...")

        # 3 — Build FAISS index over source document
        chunks = chunk_pages(pages, chunk_size=3)
        index, chunks = build_index(chunks)
        bar.progress(65, text=f"Index ready ({len(chunks)} chunks) — verifying claims...")

        # 4 — Verify
        results = verify_claims(claims, index, chunks)
        bar.progress(85, text="Verified — generating report...")

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
        multimodal = st.checkbox(
            "Enable multimodal extraction (charts & graphs)",
            value=False,
            key="multimodal",
            help=(
                "Uses Claude vision to extract claims from charts, tables "
                "and graphs in addition to text. Slower but more thorough."
            ),
        )

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
                results, report = _run_pipeline(source_path, final_summary, multimodal)
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
for _key in ("results", "report"):
    if _key not in st.session_state:
        st.session_state[_key] = None

# Sidebar navigation (always visible, rendered before page functions add their own content)
with st.sidebar:
    st.markdown("## PolicyLens")
    _mode = st.radio(
        "Navigation",
        ["Audit Document", "About"],
        label_visibility="collapsed",
    )
    st.divider()

if _mode == "About":
    show_about()
else:
    show_audit()
