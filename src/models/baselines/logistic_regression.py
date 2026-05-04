"""
Logistic Regression baseline PLH detector.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class PLHLogisticRegression:
    """
    Binary or multi-class LR detector:
      StandardScaler → LogisticRegression (L2, balanced weights).
    """
    def __init__(self, binary: bool = True, max_iter: int = 1000, C: float = 1.0):
        if not _HAS_SKLEARN:
            raise ImportError("pip install scikit-learn")
        self.binary = binary
        self.max_iter = max_iter
        self.C = C
        self._pipeline: Optional[Pipeline] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLHLogisticRegression":
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=self.max_iter, C=self.C,
                                       class_weight="balanced", solver="lbfgs")),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        metrics: Dict[str, Any] = {"classification_report": report,
                                    "accuracy": float(report.get("accuracy", 0))}
        try:
            if self.binary:
                metrics["roc_auc"] = float(roc_auc_score(y, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(
                    y, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            metrics["roc_auc"] = None
        return metrics

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "PLHLogisticRegression":
        with open(path, "rb") as f:
            return pickle.load(f)

    def feature_importance(self, feature_names: List[str]) -> List[Tuple[str, float]]:
        coef = self._pipeline.named_steps["clf"].coef_
        if coef.ndim > 1:
            coef = np.abs(coef).mean(axis=0)
        pairs = sorted(zip(feature_names, coef.tolist()), key=lambda x: -x[1])
        return pairs
