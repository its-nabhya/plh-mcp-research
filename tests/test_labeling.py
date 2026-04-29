"""
Tests for the taxonomy and rule-based labeler.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.labeling.taxonomy import (
    TAXONOMY, ALL_KEYS, LEVELS, get_level, get_severity,
)
from src.labeling.labeler import RuleBasedLabeler


class TestTaxonomy:
    def test_all_levels_populated(self):
        for level in [1, 2, 3, 4]:
            assert len(LEVELS[level]) > 0, f"Level {level} is empty"

    def test_15_categories(self):
        assert len(ALL_KEYS) == 15

    def test_levels_consistent(self):
        for key in ALL_KEYS:
            entry = TAXONOMY[key]
            assert entry.level in (1, 2, 3, 4)
            assert get_level(key) == entry.level

    def test_severity_values(self):
        valid_severities = {"critical", "high", "medium", "low"}
        for key in ALL_KEYS:
            assert get_severity(key) in valid_severities

    def test_unknown_key(self):
        assert get_level("nonexistent_key") is None
        assert get_severity("nonexistent_key") is None


class TestRuleBasedLabeler:
    labeler = RuleBasedLabeler()

    def _entry(self, tool_name, violations=None, tool_exists=True,
               schema_valid=True, permission=True, state=None, category="valid_baseline"):
        return {
            "tool_call": {"name": tool_name, "parameters": {}},
            "validation": {
                "tool_exists": tool_exists,
                "schema_valid": schema_valid,
                "permission_granted": permission,
                "state_valid": state,
                "violations": violations or [],
            },
            "label": {"primary": None, "is_exploratory": False, "confidence": 0.9},
            "prompt": {"category": category},
        }

    def test_valid_call(self):
        entry = self._entry("add_numbers")
        p, sec, conf, notes = self.labeler.label(entry)
        assert p is None
        assert conf > 0.9

    def test_nonexistent_tool(self):
        entry = self._entry(
            "fake_tool", tool_exists=False,
            violations=[{"type": "tool_existence", "severity": "critical",
                         "message": "Not found", "field": None}]
        )
        p, sec, conf, notes = self.labeler.label(entry)
        assert p == "tool_existence_hallucination"

    def test_permission_violation(self):
        entry = self._entry(
            "admin_tool", permission=False,
            violations=[{"type": "permission_violation", "severity": "critical",
                         "message": "No perms", "field": None}]
        )
        p, sec, conf, notes = self.labeler.label(entry)
        assert p == "permission_violation_hallucination"

    def test_schema_violation(self):
        entry = self._entry(
            "add_numbers", schema_valid=False,
            violations=[{"type": "parameter_schema", "severity": "high",
                         "message": "Wrong type for a", "field": "a"}]
        )
        p, sec, conf, notes = self.labeler.label(entry)
        assert p == "parameter_schema_hallucination"

    def test_exploratory_call_not_hallucination(self):
        entry = self._entry("list_tools")
        p, sec, conf, notes = self.labeler.label(entry)
        assert p is None

    def test_api_error_tool(self):
        entry = self._entry("__api_error__", tool_exists=False)
        p, sec, conf, notes = self.labeler.label(entry)
        assert p == "tool_existence_hallucination"
