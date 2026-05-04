"""
Feature Extractor — converts log entry dicts into numpy feature vectors.

Feature categories (§7.1 of protocol):
  Static     — validation results
  Confidence — scores and word counts from reasoning
  Behavioral — exploration, retries, turn number

All field access uses .get() with safe defaults so the extractor
never crashes on partial or malformed log entries.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

STATIC_FEATURES = [
    "tool_exists", "schema_valid", "permission_granted", "state_valid",
    "num_violations", "violation_severity_max",
]
CONFIDENCE_FEATURES = [
    "confidence_score", "high_conf_word_count", "low_conf_word_count",
    "assertion_strength",
]
BEHAVIORAL_FEATURES = [
    "used_discovery_tools", "num_exploratory_queries",
    "conversation_turn", "had_parse_error",
]
ALL_FEATURE_NAMES: List[str] = STATIC_FEATURES + CONFIDENCE_FEATURES + BEHAVIORAL_FEATURES
_SEVERITY_MAP = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _tri(val: Optional[bool]) -> float:
    """Map True→1.0, False→0.0, None/missing→0.5 (unknown)."""
    if val is True:
        return 1.0
    if val is False:
        return 0.0
    return 0.5


def _safe_cs(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely extract confidence_signals from an entry regardless of
    whether it is nested under agent_reasoning or missing entirely.
    Always returns a dict with all expected keys.
    """
    agent_reasoning = entry.get("agent_reasoning") or {}

    # confidence_signals may be a nested dict, or absent
    cs = agent_reasoning.get("confidence_signals") or {}

    # Return with safe defaults for every key the extractor needs
    return {
        "score":                      cs.get("score", 0.5),
        "high_conf_words":            cs.get("high_conf_words") or [],
        "low_conf_words":             cs.get("low_conf_words") or [],
        "discovery_before_invocation": cs.get("discovery_before_invocation", False),
        "tools_queried":              cs.get("tools_queried") or [],
    }


class FeatureExtractor:
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or ALL_FEATURE_NAMES

    def extract(self, entry: Dict[str, Any]) -> np.ndarray:
        return np.array(
            [self._compute(entry)[k] for k in self.feature_names],
            dtype=np.float32,
        )

    def extract_batch(self, entries: List[Dict[str, Any]]) -> np.ndarray:
        rows = [self.extract(e) for e in entries]
        return np.stack(rows) if rows else np.empty((0, len(self.feature_names)))

    def _compute(self, entry: Dict[str, Any]) -> Dict[str, float]:
        # --- validation (safe) ---
        v = entry.get("validation") or {}
        violations = v.get("violations") or []
        severities = [
            _SEVERITY_MAP.get(vio.get("severity", "low"), 1)
            for vio in violations
            if isinstance(vio, dict)
        ]

        # --- confidence signals (safe) ---
        cs = _safe_cs(entry)
        hc = len(cs["high_conf_words"])
        lc = len(cs["low_conf_words"])

        # --- tool call (safe) ---
        tool_name = (entry.get("tool_call") or {}).get("name", "")

        return {
            # static
            "tool_exists":            float(v.get("tool_exists", False)),
            "schema_valid":           _tri(v.get("schema_valid")),
            "permission_granted":     _tri(v.get("permission_granted")),
            "state_valid":            _tri(v.get("state_valid")),
            "num_violations":         float(len(violations)),
            "violation_severity_max": float(max(severities, default=0)),
            # confidence
            "confidence_score":       float(cs["score"]),
            "high_conf_word_count":   float(hc),
            "low_conf_word_count":    float(lc),
            "assertion_strength":     hc / (hc + lc + 1),
            # behavioral
            "used_discovery_tools":   float(bool(cs["discovery_before_invocation"])),
            "num_exploratory_queries": float(len(cs["tools_queried"])),
            "conversation_turn":      float(entry.get("conversation_turn", 1)),
            "had_parse_error":        float(tool_name in ("__unknown__", "__api_error__")),
        }


class LabelEncoder:
    def __init__(self, classes: Optional[List[str]] = None):
        from src.labeling.taxonomy import HALLUCINATION_CLASSES
        self.classes = classes or HALLUCINATION_CLASSES
        self._to_idx = {c: i for i, c in enumerate(self.classes)}

    def encode(self, label: Optional[str]) -> int:
        if label is None:
            return self._to_idx.get("valid", len(self.classes) - 1)
        return self._to_idx.get(label, self._to_idx.get("valid", 0))

    def decode(self, idx: int) -> str:
        return self.classes[idx] if 0 <= idx < len(self.classes) else "unknown"

    def encode_batch(self, labels: List[Optional[str]]) -> np.ndarray:
        return np.array([self.encode(l) for l in labels], dtype=np.int64)

    def binary_encode(self, label: Optional[str]) -> int:
        return 0 if label is None else 1


def extract_dataset(
    entries: List[Dict[str, Any]],
    binary: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    extractor = FeatureExtractor()
    encoder = LabelEncoder()
    X = extractor.extract_batch(entries)
    if binary:
        y = np.array(
            [encoder.binary_encode(e.get("label", {}).get("primary"))
             for e in entries],
            dtype=np.int64,
        )
    else:
        y = encoder.encode_batch(
            [e.get("label", {}).get("primary") for e in entries]
        )
    return X, y