"""
PolicyLens Evaluation Script

Measures system performance against manually annotated ground truth.
Produces accuracy, false negative rate, silent failure detection rate,
Cohen's Kappa, verdict distribution, and a full confusion matrix.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ALL_VERDICTS = [
    "Supported",
    "Partially Supported",
    "Unsupported",
    "High-Risk Silent Failure",
]

_ABBREV = {
    "Supported":                "Supported       ",
    "Partially Supported":      "Part. Supported ",
    "Unsupported":              "Unsupported     ",
    "High-Risk Silent Failure": "High-Risk SF    ",
}

_LINE_WIDE = "=" * 72
_LINE_THIN = "-" * 72


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_annotations(csv_path: str) -> pd.DataFrame:
    """
    Load manually annotated ground truth from a CSV file.

    Expected columns: claim_id (int), claim_text (str), true_verdict (str).

    Returns
    -------
    pd.DataFrame  with columns [claim_id, claim_text, true_verdict]
    """
    df = pd.read_csv(csv_path)
    required = {"claim_id", "claim_text", "true_verdict"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotation CSV is missing columns: {missing}")
    df["claim_id"] = df["claim_id"].astype(int)
    return df[["claim_id", "claim_text", "true_verdict"]]


def load_results(json_path: str) -> pd.DataFrame:
    """
    Load system verdict results from a PolicyLens JSON report.

    Reads the ``summary_table`` key produced by reporter.report_to_json.

    Returns
    -------
    pd.DataFrame  with at least columns [claim_id, verdict, confidence_score]
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if "summary_table" not in data:
        raise ValueError("JSON report must contain a 'summary_table' key.")

    df = pd.DataFrame(data["summary_table"])
    df["claim_id"] = df["claim_id"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _cohen_kappa(y_true: list, y_pred: list) -> float:
    """
    Compute Cohen's Kappa for multi-class classification.

    κ = (po − pe) / (1 − pe)
    where po = observed agreement, pe = expected agreement by chance.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    po = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n

    pe = sum(
        (true_counts.get(label, 0) / n) * (pred_counts.get(label, 0) / n)
        for label in ALL_VERDICTS
    )

    if abs(1.0 - pe) < 1e-10:
        return 1.0
    return round((po - pe) / (1.0 - pe), 4)


def compute_metrics(annotations_df: pd.DataFrame, results_df: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics by joining annotations and system results on claim_id.

    Parameters
    ----------
    annotations_df : pd.DataFrame
        Output of load_annotations — has [claim_id, true_verdict].
    results_df : pd.DataFrame
        Output of load_results — has [claim_id, verdict].

    Returns
    -------
    dict
        claim_classification_accuracy : float
        false_negative_rate           : float  — truly-problematic claims
                                                  wrongly predicted as Supported
        silent_failure_detection_rate : float  — High-Risk SF correctly identified
        cohen_kappa                   : float
        verdict_distribution          : dict[str, int]  (predicted counts)
        confusion_matrix              : dict[str, dict[str, int]]
                                        {true_label: {pred_label: count}}
        n_evaluated                   : int  — claims in the joined set
    """
    merged = annotations_df.merge(
        results_df[["claim_id", "verdict"]],
        on="claim_id",
        how="inner",
    ).rename(columns={"verdict": "predicted_verdict"})

    if merged.empty:
        raise ValueError("No overlapping claim_ids between annotations and results.")

    y_true = merged["true_verdict"].tolist()
    y_pred = merged["predicted_verdict"].tolist()
    n = len(merged)

    # --- Accuracy ---
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = round(correct / n, 4)

    # --- False Negative Rate ---
    # Truly-problematic = any verdict that is NOT "Supported"
    # FN = truly-problematic claim predicted as "Supported"
    truly_problematic = merged[merged["true_verdict"] != "Supported"]
    if len(truly_problematic) > 0:
        false_negatives = (truly_problematic["predicted_verdict"] == "Supported").sum()
        fnr = round(false_negatives / len(truly_problematic), 4)
    else:
        fnr = 0.0

    # --- Silent Failure Detection Rate ---
    truly_high_risk = merged[merged["true_verdict"] == "High-Risk Silent Failure"]
    if len(truly_high_risk) > 0:
        correctly_flagged = (
            truly_high_risk["predicted_verdict"] == "High-Risk Silent Failure"
        ).sum()
        sfdr = round(correctly_flagged / len(truly_high_risk), 4)
    else:
        sfdr = 0.0

    # --- Cohen's Kappa ---
    kappa = _cohen_kappa(y_true, y_pred)

    # --- Verdict Distribution (predicted) ---
    pred_counter = Counter(y_pred)
    verdict_distribution = {v: pred_counter.get(v, 0) for v in ALL_VERDICTS}

    # --- Confusion Matrix ---
    cm: dict[str, dict[str, int]] = {
        true_v: {pred_v: 0 for pred_v in ALL_VERDICTS}
        for true_v in ALL_VERDICTS
    }
    for true_v, pred_v in zip(y_true, y_pred):
        if true_v in cm and pred_v in cm[true_v]:
            cm[true_v][pred_v] += 1

    return {
        "claim_classification_accuracy": accuracy,
        "false_negative_rate":           fnr,
        "silent_failure_detection_rate": sfdr,
        "cohen_kappa":                   kappa,
        "verdict_distribution":          verdict_distribution,
        "confusion_matrix":              cm,
        "n_evaluated":                   n,
    }


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict) -> None:
    """
    Print a formatted evaluation report to stdout.
    """
    n = metrics["n_evaluated"]
    dist = metrics["verdict_distribution"]
    cm = metrics["confusion_matrix"]

    print(_LINE_WIDE)
    print("  PolicyLens — Evaluation Report")
    print(_LINE_WIDE)
    print(f"  Claims evaluated               : {n}")
    print(_LINE_THIN)

    print("  PERFORMANCE METRICS")
    rows = [
        ("Claim Classification Accuracy",   f"{metrics['claim_classification_accuracy']:.2%}"),
        ("False Negative Rate",             f"{metrics['false_negative_rate']:.2%}"),
        ("Silent Failure Detection Rate",   f"{metrics['silent_failure_detection_rate']:.2%}"),
        ("Cohen's Kappa",                   f"{metrics['cohen_kappa']:.4f}"),
    ]
    for label, value in rows:
        print(f"    {label:<38}  {value}")

    print(_LINE_THIN)
    print("  VERDICT DISTRIBUTION  (predicted)")
    for verdict in ALL_VERDICTS:
        count = dist[verdict]
        pct = f"{count / n:.1%}" if n else "—"
        print(f"    {verdict:<34}  {count:>3}  ({pct})")

    print(_LINE_THIN)
    print("  CONFUSION MATRIX  (rows = true label, cols = predicted)")
    print()

    col_abbrevs = [_ABBREV[v] for v in ALL_VERDICTS]
    header = "                    " + "  ".join(col_abbrevs)
    print(f"  {header}")
    print(f"  {'True \\ Pred':<18}  " + "  ".join("-" * 16 for _ in ALL_VERDICTS))

    for true_v in ALL_VERDICTS:
        row_label = _ABBREV[true_v]
        cells = "  ".join(f"{cm[true_v][pred_v]:^16}" for pred_v in ALL_VERDICTS)
        print(f"  {row_label}  {cells}")

    print()
    print(_LINE_WIDE)

    # Interpretation notes
    kappa = metrics["cohen_kappa"]
    if kappa >= 0.80:
        kappa_note = "almost perfect agreement"
    elif kappa >= 0.60:
        kappa_note = "substantial agreement"
    elif kappa >= 0.40:
        kappa_note = "moderate agreement"
    elif kappa >= 0.20:
        kappa_note = "fair agreement"
    else:
        kappa_note = "slight agreement"

    fnr = metrics["false_negative_rate"]
    fnr_note = "within MVP target (<5%)" if fnr < 0.05 else "above MVP target of <5%"

    print(f"  Cohen's Kappa {kappa:.4f} — {kappa_note}")
    print(f"  False Negative Rate {fnr:.2%} — {fnr_note}")
    print(_LINE_WIDE)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _EVAL_DIR = Path(__file__).parent
    _ANNOTATIONS_PATH = _EVAL_DIR / "sample_annotations.csv"

    # --- Load annotations ---
    print(f"Loading annotations from: {_ANNOTATIONS_PATH}")
    annotations = load_annotations(str(_ANNOTATIONS_PATH))
    print(f"  {len(annotations)} annotated claims loaded.\n")

    # --- Load or generate results ---
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
        print(f"Loading system results from: {results_path}")
        results = load_results(results_path)
        print(f"  {len(results)} result rows loaded.\n")
    else:
        print("No results JSON provided — using mock predictions for demonstration.\n")

        # Designed to produce interesting metrics across all four verdict types.
        # Intentional errors: claims 4, 6, and 10 are false negatives.
        # Claims 7 and 9 are correctly detected High-Risk Silent Failures.
        MOCK_PREDICTIONS = {
            1:  "Supported",               # true: Supported          ✓
            2:  "Supported",               # true: Supported          ✓
            3:  "Partially Supported",     # true: Partially Supported ✓
            4:  "Supported",               # true: Partially Supported ✗ (false negative)
            5:  "Unsupported",             # true: Unsupported         ✓
            6:  "Supported",               # true: Unsupported         ✗ (false negative)
            7:  "High-Risk Silent Failure", # true: High-Risk SF       ✓
            8:  "Partially Supported",     # true: High-Risk SF        ✗ (under-detected)
            9:  "High-Risk Silent Failure", # true: High-Risk SF       ✓
            10: "Supported",               # true: High-Risk SF        ✗ (false negative)
        }

        results = pd.DataFrame([
            {"claim_id": cid, "verdict": verdict}
            for cid, verdict in MOCK_PREDICTIONS.items()
        ])

    # --- Compute and print metrics ---
    metrics = compute_metrics(annotations, results)
    print_metrics(metrics)
