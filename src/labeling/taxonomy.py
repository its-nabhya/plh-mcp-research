"""
PLH Taxonomy — constants, levels, and metadata for the 4-level,
15-category Protocol-Level Hallucination taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TaxonomyEntry:
    key: str
    level: int
    short_name: str
    description: str
    detection_signal: str
    severity: str   # "critical" | "high" | "medium" | "low"


TAXONOMY: Dict[str, TaxonomyEntry] = {
    # ---- Level 1: Existence ----
    "tool_existence_hallucination": TaxonomyEntry(
        key="tool_existence_hallucination", level=1,
        short_name="Tool Existence",
        description="Agent calls a tool that does not exist in the registry.",
        detection_signal="Tool name not in validated registry.",
        severity="critical",
    ),
    "capability_existence_hallucination": TaxonomyEntry(
        key="capability_existence_hallucination", level=1,
        short_name="Capability Existence",
        description="Tool exists but claimed capability is not supported.",
        detection_signal="Semantic mismatch between tool description and intended use.",
        severity="high",
    ),
    "resource_existence_hallucination": TaxonomyEntry(
        key="resource_existence_hallucination", level=1,
        short_name="Resource Existence",
        description="Agent assumes files, data, or endpoints exist without verification.",
        detection_signal="State validation failures, filesystem checks.",
        severity="high",
    ),
    # ---- Level 2: Specification ----
    "parameter_schema_hallucination": TaxonomyEntry(
        key="parameter_schema_hallucination", level=2,
        short_name="Parameter Schema",
        description="Type, format, or structure violations in parameters.",
        detection_signal="JSON schema validation failures.",
        severity="high",
    ),
    "parameter_semantic_hallucination": TaxonomyEntry(
        key="parameter_semantic_hallucination", level=2,
        short_name="Parameter Semantic",
        description="Parameters match schema but values are nonsensical or out-of-domain.",
        detection_signal="Range checks, semantic validation, injection detection.",
        severity="medium",
    ),
    "constraint_violation_hallucination": TaxonomyEntry(
        key="constraint_violation_hallucination", level=2,
        short_name="Constraint Violation",
        description="Exceeding limits, violating preconditions, or ignoring invariants.",
        detection_signal="Constraint checks in schemas, rate limit violations.",
        severity="high",
    ),
    "dependency_hallucination": TaxonomyEntry(
        key="dependency_hallucination", level=2,
        short_name="Dependency",
        description="Assuming tool X implies tool Y, or unsupported composition.",
        detection_signal="Cross-tool dependency analysis, capability graph validation.",
        severity="medium",
    ),
    # ---- Level 3: Contextual ----
    "permission_violation_hallucination": TaxonomyEntry(
        key="permission_violation_hallucination", level=3,
        short_name="Permission Violation",
        description="Ignoring or misunderstanding access control.",
        detection_signal="Permission checks, role validation.",
        severity="critical",
    ),
    "state_assumption_hallucination": TaxonomyEntry(
        key="state_assumption_hallucination", level=3,
        short_name="State Assumption",
        description="Assuming world state without verification or discovery.",
        detection_signal="State consistency checks, precondition validation.",
        severity="high",
    ),
    "temporal_violation_hallucination": TaxonomyEntry(
        key="temporal_violation_hallucination", level=3,
        short_name="Temporal Violation",
        description="Using deprecated tools, future capabilities, or stale state.",
        detection_signal="Version checks, deprecation tracking, timestamp validation.",
        severity="high",
    ),
    "execution_fabrication_hallucination": TaxonomyEntry(
        key="execution_fabrication_hallucination", level=3,
        short_name="Execution Fabrication",
        description="Reporting success without actual execution.",
        detection_signal="Execution log verification, output validation.",
        severity="critical",
    ),
    # ---- Level 4: Semantic ----
    "intent_misalignment_hallucination": TaxonomyEntry(
        key="intent_misalignment_hallucination", level=4,
        short_name="Intent Misalignment",
        description="Tool is valid but irrelevant or counterproductive to the task.",
        detection_signal="Embedding similarity between task and tool purpose.",
        severity="medium",
    ),
    "tool_conflation_hallucination": TaxonomyEntry(
        key="tool_conflation_hallucination", level=4,
        short_name="Tool Conflation",
        description="Merging capabilities of similar but distinct tools.",
        detection_signal="Tool similarity analysis, capability boundary detection.",
        severity="medium",
    ),
    "over_generalization_hallucination": TaxonomyEntry(
        key="over_generalization_hallucination", level=4,
        short_name="Over-Generalization",
        description="Assuming a tool works in contexts it was not designed for.",
        detection_signal="Context-domain mismatch, scope validation.",
        severity="medium",
    ),
    "under_specification_hallucination": TaxonomyEntry(
        key="under_specification_hallucination", level=4,
        short_name="Under-Specification",
        description="Insufficient information for disambiguation; assuming non-existent defaults.",
        detection_signal="Completeness checks, required context validation.",
        severity="low",
    ),
}

LEVELS: Dict[int, List[str]] = {1: [], 2: [], 3: [], 4: []}
for _key, _entry in TAXONOMY.items():
    LEVELS[_entry.level].append(_key)

ALL_KEYS: List[str] = list(TAXONOMY.keys())
HALLUCINATION_CLASSES: List[str] = ALL_KEYS + ["valid"]   # for multi-class ML


def get_level(key: str) -> Optional[int]:
    entry = TAXONOMY.get(key)
    return entry.level if entry else None


def get_severity(key: str) -> Optional[str]:
    entry = TAXONOMY.get(key)
    return entry.severity if entry else None
