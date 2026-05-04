"""
Confidence calibration analysis.

Implements ECE (Expected Calibration Error), MCE, Brier score,
and confidence-correctness correlation.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE = Σ (|B_m|/n) * |acc(B_m) - conf(B_m)|
    y_proba: probability of positive class, shape (N,)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_proba[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """MCE = max_m |acc(B_m) - conf(B_m)|"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        err = abs(float(y_true[mask].mean()) - float(y_proba[mask].mean()))
        mce = max(mce, err)
    return float(mce)


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(np.mean((y_proba - y_true) ** 2))


def confidence_correctness_correlation(
    y_true: np.ndarray,
    confidence_scores: np.ndarray,
) -> float:
    """Pearson correlation between LLM confidence and correctness."""
    if np.std(confidence_scores) < 1e-9 or np.std(y_true) < 1e-9:
        return 0.0
    corr = np.corrcoef(confidence_scores, y_true.astype(float))[0, 1]
    return float(corr)


def overconfidence_ratio(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.8,
) -> float:
    """P(incorrect | high_conf) / P(incorrect | overall)."""
    overall_err = float(np.mean(y_true != y_pred))
    if overall_err < 1e-9:
        return 1.0
    high_conf_mask = y_proba >= threshold
    if high_conf_mask.sum() == 0:
        return 0.0
    high_conf_err = float(np.mean(y_true[high_conf_mask] != y_pred[high_conf_mask]))
    return high_conf_err / overall_err


def calibration_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    confidence_scores: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    report = {
        "ece": expected_calibration_error(y_true, y_proba, n_bins),
        "mce": maximum_calibration_error(y_true, y_proba, n_bins),
        "brier_score": brier_score(y_true, y_proba),
        "overconfidence_ratio": overconfidence_ratio(y_true, y_proba, y_pred),
    }
    if confidence_scores is not None:
        report["confidence_correctness_correlation"] = confidence_correctness_correlation(
            (y_true == y_pred).astype(float), confidence_scores)
    return report
