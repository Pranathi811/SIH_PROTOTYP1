"""Model evaluation suite for telemetry anomaly detection.

Implements FR-16, FR-17, and FR-18.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics against ground-truth labels."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Calculate ROC-AUC & PR-AUC safely
    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        roc_auc = 0.0
        
    try:
        pr_auc = float(average_precision_score(y_true, y_scores))
    except Exception:
        pr_auc = 0.0

    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "raw_matrix": cm.tolist()
        }
    }


def evaluate_models(
    y_test: np.ndarray,
    predictions: Dict[str, np.ndarray],
    scores: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, Any]]:
    """Evaluate multiple models side-by-side."""
    results = {}
    for model_name in predictions.keys():
        results[model_name] = compute_metrics(
            y_true=y_test,
            y_pred=predictions[model_name],
            y_scores=scores[model_name]
        )
    return results


def evaluate_per_anomaly_type(
    test_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    y_pred: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Evaluate anomaly detection performance separately for each synthetic anomaly category:
    point, collective, level_shift, trend (FR-18 & FR-21).
    """
    # Merge anomaly_type annotation by timestamp
    type_map = labels_df.set_index("timestamp")["anomaly_type"].to_dict()
    test_types = test_df["timestamp"].map(type_map).fillna("normal")
    y_true = test_df["label"].to_numpy()
    
    anomaly_categories = ["point", "collective", "level_shift", "trend"]
    results = []
    
    for cat in anomaly_categories:
        # Mask where truth is this specific category OR normal
        # This isolates performance on this specific anomaly type against normal baseline
        cat_mask = test_types.eq(cat)
        normal_mask = test_types.eq("normal") | y_true == 0
        eval_mask = (cat_mask | normal_mask).to_numpy()
        
        subset_true = (test_types[eval_mask].eq(cat)).astype(int).to_numpy()
        subset_pred = y_pred[eval_mask]
        
        cat_count = int(cat_mask.sum())
        
        prec = float(precision_score(subset_true, subset_pred, zero_division=0))
        rec = float(recall_score(subset_true, subset_pred, zero_division=0))
        f1 = float(f1_score(subset_true, subset_pred, zero_division=0))
        
        results.append({
            "type": cat,
            "count": cat_count,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4)
        })
        
    return results
