# import jsonschema

# class Validator:
#     def __init__(self, tool_registry):
#         self.tool_registry = tool_registry

#     def validate(self, tool_call):
#         violations = []

#         tool_name = tool_call["name"]
#         params = tool_call.get("parameters", {})

#         # 1. Tool existence check
#         if tool_name not in self.tool_registry:
#             violations.append({
#                 "type": "tool_existence",
#                 "severity": "critical",
#                 "message": f"{tool_name} not found"
#             })
#             return self._build_response(False, False, None, violations)

#         tool_schema = self.tool_registry[tool_name]["schema"]

#         # 2. Schema validation
#         try:
#             jsonschema.validate(instance=params, schema=tool_schema)
#             schema_valid = True
#         except jsonschema.ValidationError as e:
#             schema_valid = False
#             violations.append({
#                 "type": "parameter_schema",
#                 "severity": "high",
#                 "message": str(e)
#             })

#         # 3. Permission check (dummy for now)
#         permission_granted = True

#         return self._build_response(
#             True,
#             schema_valid,
#             permission_granted,
#             violations
#         )

#     def _build_response(self, tool_exists, schema_valid, permission_granted, violations):
#         return {
#             "tool_exists": tool_exists,
#             "schema_valid": schema_valid,
#             "permission_granted": permission_granted,
#             "violations": violations
#         }
"""
Validator — maps raw ValidationResult from the world into a
LabelRecord using the PLH taxonomy.

The labeling is deterministic (rule-based) and relies only on
information available at call time (no ML).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.logging.schema import (
    AgentReasoning,
    ConfidenceSignals,
    LabelRecord,
    ValidationRecord,
    WorldResponseRecord,
)
from src.worlds.base_world import ValidationResult, WorldResponse


# ---------------------------------------------------------------------------
# Taxonomy constants
# ---------------------------------------------------------------------------

EXPLORATORY_TOOLS = {"list_tools", "describe_tool", "get_world_metadata"}

# Confidence word lists (from executive summary)
HIGH_CONFIDENCE_WORDS = {
    "will", "definitely", "clearly", "obviously", "certainly",
    "the correct tool is", "this requires", "must", "always",
    "undoubtedly", "without doubt",
}

LOW_CONFIDENCE_WORDS = {
    "might", "possibly", "could", "perhaps", "maybe",
    "i think", "let me try", "not sure", "unsure",
    "i believe", "probably", "hopefully",
}

# PLH taxonomy mapping
TAXONOMY_LEVELS = {
    # Level 1 – Existence
    "tool_existence_hallucination": 1,
    "capability_existence_hallucination": 1,
    "resource_existence_hallucination": 1,
    # Level 2 – Specification
    "parameter_schema_hallucination": 2,
    "parameter_semantic_hallucination": 2,
    "constraint_violation_hallucination": 2,
    "dependency_hallucination": 2,
    # Level 3 – Contextual
    "permission_violation_hallucination": 3,
    "state_assumption_hallucination": 3,
    "temporal_violation_hallucination": 3,
    "execution_fabrication_hallucination": 3,
    # Level 4 – Semantic
    "intent_misalignment_hallucination": 4,
    "tool_conflation_hallucination": 4,
    "over_generalization_hallucination": 4,
    "under_specification_hallucination": 4,
}


# ---------------------------------------------------------------------------
# Confidence signal extractor
# ---------------------------------------------------------------------------

def extract_confidence_signals(
    reasoning_text: str,
    tool_call_name: str,
    prior_tool_calls: List[str],
) -> ConfidenceSignals:
    """
    Extract confidence signals from the agent's reasoning trace.
    """
    text_lower = reasoning_text.lower()
    tokens = set(re.findall(r"[a-z_' ]+", text_lower))

    high_found = [w for w in HIGH_CONFIDENCE_WORDS if w in text_lower]
    low_found = [w for w in LOW_CONFIDENCE_WORDS if w in text_lower]

    # Simple weighted score: 0 = very low confidence, 1 = very high
    raw_score = 0.5 + 0.1 * len(high_found) - 0.1 * len(low_found)
    score = max(0.0, min(1.0, raw_score))

    exploratory_calls = [tc for tc in prior_tool_calls if tc in EXPLORATORY_TOOLS]
    used_list = any(tc == "list_tools" for tc in prior_tool_calls)
    used_desc = any(tc == "describe_tool" for tc in prior_tool_calls)

    return ConfidenceSignals(
        high_conf_words=high_found,
        low_conf_words=low_found,
        score=score,
        used_list_tools=used_list,
        used_describe_tool=used_desc,
        discovery_before_invocation=len(exploratory_calls) > 0,
        tools_queried=exploratory_calls,
    )


# ---------------------------------------------------------------------------
# Core labeler
# ---------------------------------------------------------------------------

class Validator:
    """
    Converts world ValidationResult + WorldResponse into a structured
    LabelRecord, following the 4-level PLH taxonomy.
    """

    def label(
        self,
        tool_name: str,
        params: Dict[str, Any],
        validation: ValidationResult,
        response: WorldResponse,
        expected_hallucination: Optional[str] = None,
    ) -> LabelRecord:
        # Exploratory tool calls are never labelled as hallucinations
        if tool_name in EXPLORATORY_TOOLS and validation.is_valid:
            return LabelRecord(
                primary=None,
                secondary=[],
                level=None,
                is_hallucination=False,
                confidence=0.99,
                is_exploratory=True,
                notes="Exploratory discovery call.",
            )

        primary, secondary, confidence, notes = self._classify(
            tool_name, params, validation, response, expected_hallucination
        )

        is_hallucination = primary is not None
        level = TAXONOMY_LEVELS.get(primary) if primary else None

        return LabelRecord(
            primary=primary,
            secondary=secondary,
            level=level,
            is_hallucination=is_hallucination,
            confidence=confidence,
            notes=notes,
        )

    def _classify(
        self,
        tool_name: str,
        params: Dict[str, Any],
        validation: ValidationResult,
        response: WorldResponse,
        expected: Optional[str],
    ) -> Tuple[Optional[str], List[str], float, str]:
        """
        Apply labeling rules in priority order (most severe first).
        Returns (primary, secondary, confidence, notes).
        """
        primary: Optional[str] = None
        secondary: List[str] = []
        confidence = 0.9
        notes = ""

        # ---- Level 1: Existence ----
        if not validation.tool_exists:
            v = validation.violations[0] if validation.violations else None
            if v and v.type == "temporal_violation":
                primary = "temporal_violation_hallucination"
                notes = "Agent called a deprecated tool."
            else:
                primary = "tool_existence_hallucination"
                notes = f"Tool '{tool_name}' does not exist in the registry."
            return primary, secondary, confidence, notes

        # ---- Level 3: Permission ----
        if validation.permission_granted is False:
            primary = "permission_violation_hallucination"
            notes = f"Role lacks permission to call '{tool_name}'."

        # ---- Level 2: Schema ----
        schema_violations = [
            v for v in validation.violations if v.type == "parameter_schema"
        ]
        if schema_violations:
            if primary:
                secondary.append("parameter_schema_hallucination")
            else:
                primary = "parameter_schema_hallucination"
                notes = "; ".join(v.message for v in schema_violations)

        # ---- Level 3: State / preconditions ----
        if validation.state_valid is False:
            if primary:
                secondary.append("state_assumption_hallucination")
            else:
                primary = "state_assumption_hallucination"
                notes = "Precondition not met for this tool call."

        if primary:
            return primary, secondary, confidence, notes

        # ---- All validation passed → valid call ----
        # Could still be a Level 4 semantic hallucination, but that
        # requires embedding similarity (deferred to ML layer).
        # For rule-based labeling, a passing call is not hallucination.
        confidence = 0.95
        notes = "Call is structurally valid."
        return None, [], confidence, notes


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_validation_record(vr: ValidationResult) -> ValidationRecord:
    return ValidationRecord(
        tool_exists=vr.tool_exists,
        schema_valid=vr.schema_valid,
        permission_granted=vr.permission_granted,
        state_valid=vr.state_valid,
        violations=[
            {
                "type": v.type,
                "severity": v.severity,
                "message": v.message,
                "field": v.field,
            }
            for v in vr.violations
        ],
    )


def build_world_response_record(wr: WorldResponse) -> WorldResponseRecord:
    return WorldResponseRecord(
        status=wr.status,
        result=wr.result,
        error_type=wr.error_type,
        message=wr.message,
    )
