"""
Evaluation metrics for PLH detection.

Implements: PHR, Precision/Recall/F1, ROC-AUC, PR-AUC,
            Confusion matrix, per-class metrics.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np

try:
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
        confusion_matrix, classification_report,
    )
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def protocol_hallucination_rate(entries: List[Dict[str, Any]]) -> float:
    """PHR = hallucinations / total tool attempts."""
    if not entries:
        return 0.0
    h = sum(1 for e in entries if e["label"]["is_hallucination"])
    return h / len(entries)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if not _HAS_SKLEARN:
        raise ImportError("pip install scikit-learn")
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(np.mean(y_true == y_pred)),
    }
    if y_proba is not None:
        prob = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, prob))
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not _HAS_SKLEARN:
        raise ImportError("pip install scikit-learn")
    report = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0)
    metrics: Dict[str, Any] = {
        "accuracy": float(report.get("accuracy", 0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0)),
        "per_class": {k: v for k, v in report.items()
                      if k not in ("accuracy", "macro avg", "weighted avg")},
    }
    if y_proba is not None:
        try:
            metrics["roc_auc_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            metrics["roc_auc_macro"] = None
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def phr_by_complexity(
    entries: List[Dict[str, Any]],
    complexity_key: str = "world",
) -> Dict[str, float]:
    """PHR broken down by world (or any other key)."""
    buckets: Dict[str, List] = {}
    for e in entries:
        k = e.get(complexity_key, "unknown")
        buckets.setdefault(k, []).append(e)
    return {k: protocol_hallucination_rate(v) for k, v in buckets.items()}
