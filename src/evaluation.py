# =============================================================================
# evaluation.py
# Component 6: Evaluation
#
# Stage 1 (core pipeline):
#   - metrics.json         → accuracy, precision, recall, F1
#   - predictions.csv      → per-applicant decisions
#   - confusion_matrix.png → visual confusion matrix
#
# Stage 2 (after core works):
#   - feature_importance.png
#   - component_comparison.png  ← KB vs Bayesian vs ML vs Integrated
#   - bias_analysis.json
# =============================================================================

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from src.config import RESULTS_DIR, TARGET_COL


def _ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray
) -> dict:
    """
    Compute and print accuracy, precision, recall, and F1-score.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).

    Returns:
        Dict of metric_name → float value (rounded to 4 dp).
    """
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    print("\n── Evaluation Metrics ────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("──────────────────────────────────────────────────────────\n")

    return metrics


def save_metrics(metrics: dict, path: str | None = None) -> None:
    """
    Save metrics dict to results/metrics.json.

    Args:
        metrics: Output of compute_metrics().
        path   : Override default path (optional).
    """
    _ensure_results_dir()
    out = path or os.path.join(RESULTS_DIR, "metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluation] Metrics saved → {out}")


def save_predictions(
    X_test: pd.DataFrame,
    y_true: pd.Series,
    results: list[dict]
) -> None:
    """
    Save per-applicant predictions to results/predictions.csv.

    Args:
        X_test  : Test feature DataFrame.
        y_true  : Ground-truth labels.
        results : List of dicts from ExpertSystem.predict_batch().
    """
    _ensure_results_dir()
    df = X_test.copy().reset_index(drop=True)
    df["actual"]     = y_true.values
    df["decision"]   = [r["decision"]   for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    df["kb_score"]   = [r["kb_score"]   for r in results]
    df["bayes_prob"] = [r["bayes_prob"] for r in results]
    df["ml_prob"]    = [r["ml_prob"]    for r in results]

    out = os.path.join(RESULTS_DIR, "predictions.csv")
    df.to_csv(out, index=False)
    print(f"[evaluation] Predictions saved → {out}")


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    labels: list = ["Rejected", "Approved"]
) -> None:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Display labels for classes.
    """
    _ensure_results_dir()
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted", ylabel="Actual",
        title="Confusion Matrix"
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluation] Confusion matrix saved → {out}")


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    """
    Bar chart of RandomForest feature importances.
    Call after core pipeline works (Stage 2).

    Args:
        importance_df: Output of ml_model.get_feature_importance().
    """
    _ensure_results_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (RandomForest)")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluation] Feature importance saved → {out}")


def plot_component_comparison(
    y_true: pd.Series,
    results: list[dict]
) -> None:
    """
    Compare accuracy of each component individually vs integrated system.
    Uses a >= 0.5 threshold to convert probabilities to binary predictions.
    Addresses rubric P3: Depth of Analysis.

    Args:
        y_true  : Ground-truth labels.
        results : List of dicts from ExpertSystem.predict_batch().
    """
    _ensure_results_dir()
    y_arr = np.array(y_true)

    kb_pred    = (np.array([r["kb_score"]   for r in results]) >= 0.5).astype(int)
    bayes_pred = (np.array([r["bayes_prob"] for r in results]) >= 0.5).astype(int)
    ml_pred    = (np.array([r["ml_prob"]    for r in results]) >= 0.5).astype(int)
    int_pred   = np.array([
        1 if r["decision"] == "Approve" else 0 for r in results
    ])

    components  = ["Knowledge Base", "Bayesian", "ML Model", "Integrated"]
    accuracies  = [
        accuracy_score(y_arr, kb_pred),
        accuracy_score(y_arr, bayes_pred),
        accuracy_score(y_arr, ml_pred),
        accuracy_score(y_arr, int_pred),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(components, accuracies, color=["#e07b54", "#5b8db8", "#6dbf67", "#9b59b6"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Component-wise Accuracy Comparison")
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.2%}", ha="center", fontsize=10)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "component_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluation] Component comparison saved → {out}")


def run_bias_analysis(
    X_test: pd.DataFrame,
    y_true: pd.Series,
    results: list[dict],
    raw_test: pd.DataFrame | None = None
) -> None:
    """
    Compute approval rate by Gender and Married status (encoded).
    Saves results/bias_analysis.json.
    Addresses assignment ethical requirements.

    Args:
        X_test   : Encoded test features.
        y_true   : Ground-truth labels.
        results  : Output of ExpertSystem.predict_batch().
        raw_test : Optional raw (pre-encoding) test slice for readable labels.
    """
    _ensure_results_dir()
    df = X_test.copy().reset_index(drop=True)
    df["actual"]   = y_true.values
    df["decision"] = [r["decision"] for r in results]
    df["approved"] = (df["decision"] == "Approve").astype(int)

    bias = {}

    for col in ["Gender", "Married"]:
        if col in df.columns:
            group_rates = df.groupby(col)["approved"].mean().round(4).to_dict()
            bias[col] = {str(k): v for k, v in group_rates.items()}

    out = os.path.join(RESULTS_DIR, "bias_analysis.json")
    with open(out, "w") as f:
        json.dump(bias, f, indent=2)
    print(f"[evaluation] Bias analysis saved → {out}")
    print(f"  Bias breakdown: {bias}")
