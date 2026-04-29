"""
Automated labeler — post-processes log entries to finalize labels.

The Validator in logging/validator.py does real-time labeling during
the agent loop.  This module provides:
  - Batch re-labeling of existing log files
  - Label refinement with additional context (e.g., expected_hallucination)
  - Quality metrics (label confidence distribution)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.labeling.taxonomy import TAXONOMY, ALL_KEYS, get_level


# ---------------------------------------------------------------------------
# Rule-based re-labeler (deterministic)
# ---------------------------------------------------------------------------

class RuleBasedLabeler:
    """
    Applies the PLH taxonomy rules to a log dict and returns
    (primary_label, secondary_labels, confidence, notes).
    """

    def label(self, entry: Dict[str, Any]) -> Tuple[Optional[str], List[str], float, str]:
        validation = entry["validation"]
        tool_call = entry["tool_call"]
        tool_name = tool_call["name"]
        params = tool_call["parameters"]

        primary: Optional[str] = None
        secondary: List[str] = []
        confidence = 0.90
        notes = ""

        # Exploratory calls → not hallucination
        from src.logging.validator import EXPLORATORY_TOOLS
        if tool_name in EXPLORATORY_TOOLS and validation.get("tool_exists", False):
            return None, [], 0.99, "Exploratory discovery call."

        # Agent-level failures
        if tool_name in ("__unknown__", "__api_error__"):
            return "tool_existence_hallucination", [], 0.85, "Agent produced no valid tool call."

        # L1 – Existence
        if not validation.get("tool_exists", True):
            violations = validation.get("violations", [])
            is_deprecated = any(v.get("type") == "temporal_violation" for v in violations)
            if is_deprecated:
                primary = "temporal_violation_hallucination"
                notes = "Called a deprecated tool."
            else:
                primary = "tool_existence_hallucination"
                notes = f"Tool '{tool_name}' not in registry."
            confidence = 0.97
            return primary, secondary, confidence, notes

        # L3 – Permission
        if validation.get("permission_granted") is False:
            primary = "permission_violation_hallucination"
            notes = "Caller lacks required permissions."
            confidence = 0.95

        # L2 – Schema
        schema_violations = [
            v for v in validation.get("violations", [])
            if v.get("type") == "parameter_schema"
        ]
        if schema_violations:
            label = "parameter_schema_hallucination"
            if primary:
                secondary.append(label)
            else:
                primary = label
                notes = "; ".join(v.get("message", "") for v in schema_violations)
                confidence = 0.93

        # L3 – State
        if validation.get("state_valid") is False:
            label = "state_assumption_hallucination"
            if primary:
                secondary.append(label)
            else:
                primary = label
                notes = "Precondition not satisfied."
                confidence = 0.90

        if primary:
            return primary, secondary, confidence, notes

        # No structural violations → valid
        return None, [], 0.95, "Structurally valid call."


# ---------------------------------------------------------------------------
# Batch re-labeler
# ---------------------------------------------------------------------------

def relabel_file(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """
    Read a JSONL log file, re-apply rule-based labels, and optionally save.
    """
    labeler = RuleBasedLabeler()
    updated: List[Dict[str, Any]] = []

    with open(input_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            primary, secondary, conf, notes = labeler.label(entry)
            entry["label"]["primary"] = primary
            entry["label"]["secondary"] = secondary
            entry["label"]["confidence"] = conf
            entry["label"]["notes"] = notes
            entry["label"]["is_hallucination"] = primary is not None
            entry["label"]["level"] = get_level(primary) if primary else None
            updated.append(entry)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for entry in updated:
                fh.write(json.dumps(entry, default=str) + "\n")

    return updated


# ---------------------------------------------------------------------------
# Label quality metrics
# ---------------------------------------------------------------------------

def label_quality_report(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(entries)
    if total == 0:
        return {}

    conf_scores = [e["label"]["confidence"] for e in entries]
    avg_conf = sum(conf_scores) / total

    needs_review = [e for e in entries if e["label"].get("manual_review")]
    ambiguous = [e for e in entries if avg_conf < 0.75]

    label_dist: Dict[str, int] = {}
    for e in entries:
        lab = e["label"]["primary"] or "valid"
        label_dist[lab] = label_dist.get(lab, 0) + 1

    return {
        "total": total,
        "avg_label_confidence": round(avg_conf, 4),
        "needs_manual_review": len(needs_review),
        "potentially_ambiguous": len(ambiguous),
        "label_distribution": label_dist,
    }
