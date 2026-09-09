from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

OUTPUT_CSV = REPORTS_DIR / "final_model_comparison.csv"
OUTPUT_MD = REPORTS_DIR / "final_model_comparison.md"

MODEL_FILES = [
    ("Diabetes", "diabetes_model.joblib"),
    ("Heart Disease", "heart_disease_model.joblib"),
    ("Chronic Kidney Disease", "kidney_disease_model.joblib"),
    ("Liver Disease", "liver_disease_model.joblib"),
    ("Respiratory Disease", "respiratory_disease_model.joblib"),
    ("Breast Cancer", "cancer_disease_model.joblib"),
]


def first_value(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def metric_value(metrics: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_row(disease: str, model_path: Path) -> dict[str, Any]:
    row = {
        "Disease": disease,
        "Best Model": "N/A",
        "Features": "N/A",
        "Threshold": None,
        "Accuracy": None,
        "Balanced Accuracy": None,
        "Precision": None,
        "Recall": None,
        "F1 Score": None,
        "ROC-AUC": None,
        "PR-AUC": None,
        "Artifact File": model_path.name,
    }

    artifact = joblib.load(model_path)

    if not isinstance(artifact, dict):
        estimator_name = artifact.__class__.__name__
        row["Best Model"] = estimator_name
        return row

    model = artifact.get("model")

    model_name = first_value(
        artifact,
        "model_name",
        "selected_model",
        "best_model_name",
        default=None,
    )

    if model_name is None and model is not None:
        try:
            # If saved object is a Pipeline, use its final estimator.
            if hasattr(model, "named_steps") and "classifier" in model.named_steps:
                model_name = model.named_steps["classifier"].__class__.__name__
            else:
                model_name = model.__class__.__name__
        except Exception:
            model_name = model.__class__.__name__

    row["Best Model"] = model_name or "N/A"

    features = artifact.get("features")
    if isinstance(features, (list, tuple)):
        row["Features"] = len(features)

    row["Threshold"] = to_float_or_none(
        first_value(
            artifact,
            "threshold",
            "selected_threshold",
            "decision_threshold",
            default=None,
        )
    )

    metrics = artifact.get("test_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    row["Accuracy"] = to_float_or_none(
        metric_value(metrics, "Accuracy", "accuracy")
    )
    row["Balanced Accuracy"] = to_float_or_none(
        metric_value(
            metrics,
            "Balanced Accuracy",
            "balanced_accuracy",
            "Balanced_Accuracy",
        )
    )
    row["Precision"] = to_float_or_none(
        metric_value(metrics, "Precision", "precision")
    )
    row["Recall"] = to_float_or_none(
        metric_value(metrics, "Recall", "recall", "Sensitivity")
    )
    row["F1 Score"] = to_float_or_none(
        metric_value(
            metrics,
            "F1 Score",
            "F1",
            "f1",
            "F1-Score",
            "F1_score",
        )
    )
    row["ROC-AUC"] = to_float_or_none(
        metric_value(
            metrics,
            "ROC-AUC",
            "ROC AUC",
            "roc_auc",
            "AUC",
        )
    )
    row["PR-AUC"] = to_float_or_none(
        metric_value(
            metrics,
            "PR-AUC",
            "PR AUC",
            "pr_auc",
            "Average Precision",
            "average_precision",
        )
    )

    return row


def format_markdown_value(column: str, value: Any) -> str:
    if pd.isna(value):
        return "N/A"

    if column == "Threshold":
        return f"{float(value):.3f}"

    if column in {
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
    }:
        return f"{float(value):.4f}"

    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in df.iterrows():
        values = [
            format_markdown_value(column, row[column])
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def main() -> int:
    print("=" * 78)
    print("FINAL MODEL COMPARISON — MULTI-DISEASE ML SYSTEM")
    print("=" * 78)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_files = []

    for disease, filename in MODEL_FILES:
        model_path = MODELS_DIR / filename

        if not model_path.exists():
            print(f"[FAIL] Missing model: {model_path}")
            missing_files.append(filename)
            continue

        try:
            row = extract_row(disease, model_path)
            rows.append(row)
            print(f"[PASS] {disease}: {filename}")
        except Exception as exc:
            print(f"[FAIL] {disease}: could not read {filename} -> {exc}")
            return 1

    if missing_files:
        print("\nCannot create the final comparison until all six model files exist.")
        return 1

    df = pd.DataFrame(rows)

    display_columns = [
        "Disease",
        "Best Model",
        "Features",
        "Threshold",
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
    ]

    df = df[display_columns]

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.6f",
    )

    markdown_table = dataframe_to_markdown(df)

    markdown_document = (
        "# Final Model Comparison\n\n"
        "The values below are extracted directly from the saved model artifacts "
        "in the `models/` directory. Metrics marked `N/A` were not stored in that "
        "artifact and should not be invented.\n\n"
        + markdown_table
        + "\n\n"
        "## Interpretation rule\n\n"
        "- Use held-out **test-set metrics** in the final report.\n"
        "- Do not compare diseases as though they share the same dataset difficulty.\n"
        "- A higher score for one disease model does not imply that disease is "
        "easier to diagnose clinically.\n"
        "- Report the respiratory model cautiously if its discrimination is weak "
        "after leakage-prone features are excluded.\n"
    )

    OUTPUT_MD.write_text(
        markdown_document,
        encoding="utf-8",
    )

    print("\n" + "-" * 78)
    print(df.to_string(index=False))
    print("-" * 78)

    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved Markdown: {OUTPUT_MD}")

    missing_metrics = []

    for _, row in df.iterrows():
        if pd.isna(row["Accuracy"]) and pd.isna(row["ROC-AUC"]):
            missing_metrics.append(row["Disease"])

    if missing_metrics:
        print("\n[WARN] Test metrics were not stored for:")
        for disease in missing_metrics:
            print(f"       - {disease}")
        print(
            "Open that disease's training notebook and use its held-out test-set "
            "results rather than guessing values."
        )

    print("\nFINAL RESULTS EXTRACTION: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
