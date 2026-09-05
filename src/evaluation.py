"""
Model Evaluation and Comparison Module for Student Performance Prediction System.
Calculates Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and ROC-AUC.
Identifies and highlights the best-performing model.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


def evaluate_single_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for a single model.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    report_dict = classification_report(
        y_true, y_pred, target_names=["Fail", "Pass"], output_dict=True, zero_division=0
    )

    return {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": report_dict,
    }


def compare_models(
    trained_results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
) -> Tuple[pd.DataFrame, str, Dict[str, Dict[str, Any]]]:
    """
    Evaluates all trained models and constructs a sorted comparison table.
    
    Returns:
        comparison_df: pd.DataFrame with formatted metrics
        best_model_name: Name of model with highest F1-Score & Accuracy
        detailed_evaluations: Dict mapping model name to evaluation metrics dict
    """
    eval_list: List[Dict[str, Any]] = []
    detailed_evals: Dict[str, Dict[str, Any]] = {}

    for name, res in trained_results.items():
        y_pred = res["y_pred"]
        y_prob = res["y_prob"]
        metrics = evaluate_single_model(y_test, y_pred, y_prob, name)
        detailed_evals[name] = metrics

        eval_list.append(
            {
                "Model": name,
                "Accuracy (%)": round(metrics["accuracy"] * 100, 2),
                "Precision (%)": round(metrics["precision"] * 100, 2),
                "Recall (%)": round(metrics["recall"] * 100, 2),
                "F1 Score (%)": round(metrics["f1_score"] * 100, 2),
                "ROC-AUC": (
                    round(metrics["roc_auc"], 4)
                    if metrics["roc_auc"] is not None
                    else "N/A"
                ),
                "_acc_raw": metrics["accuracy"],
                "_f1_raw": metrics["f1_score"],
            }
        )

    comparison_df = pd.DataFrame(eval_list)
    # Sort by F1 Score and Accuracy descending
    comparison_df = comparison_df.sort_values(
        by=["_f1_raw", "_acc_raw"], ascending=[False, False]
    ).reset_index(drop=True)

    best_model_name = comparison_df.iloc[0]["Model"]

    # Drop raw sorting helper columns
    clean_comparison_df = comparison_df.drop(columns=["_acc_raw", "_f1_raw"])

    return clean_comparison_df, best_model_name, detailed_evals
