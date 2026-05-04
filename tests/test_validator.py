"""
Tests for the logging.Validator and confidence signal extractor.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.logging.validator import Validator, extract_confidence_signals, EXPLORATORY_TOOLS
from src.worlds.base_world import ValidationResult, ValidationViolation, WorldResponse


def _vr(tool_exists=True, schema_valid=True, permission=True,
        state=None, violations=None):
    return ValidationResult(
        tool_exists=tool_exists,
        schema_valid=schema_valid,
        permission_granted=permission,
        state_valid=state,
        violations=violations or [],
    )

def _wr(status="success", error_type=None):
    return WorldResponse(status=status, result=42, error_type=error_type, message="")


validator = Validator()


class TestExploratoryLabel:
    def test_list_tools_is_exploratory(self):
        label = validator.label("list_tools", {}, _vr(), _wr())
        assert label.is_exploratory
        assert not label.is_hallucination

    def test_describe_tool_is_exploratory(self):
        label = validator.label("describe_tool", {"tool_name": "add_numbers"}, _vr(), _wr())
        assert label.is_exploratory


class TestToolExistenceLabel:
    def test_nonexistent_tool(self):
        vr = _vr(
            tool_exists=False,
            schema_valid=None, permission=None,
            violations=[ValidationViolation("tool_existence", "critical", "Not found")],
        )
        label = validator.label("fake_tool", {}, vr, _wr("error", "ToolNotFoundError"))
        assert label.is_hallucination
        assert label.primary == "tool_existence_hallucination"
        assert label.level == 1

    def test_deprecated_tool(self):
        vr = _vr(
            tool_exists=False,
            schema_valid=None, permission=None,
            violations=[ValidationViolation("temporal_violation", "high", "Deprecated")],
        )
        label = validator.label("old_api", {}, vr, _wr("error"))
        assert label.primary == "temporal_violation_hallucination"
        assert label.level == 3


class TestSchemaLabel:
    def test_wrong_type_param(self):
        vr = _vr(
            schema_valid=False,
            violations=[
                ValidationViolation("parameter_schema", "high",
                                    "a expects number, got str", field="a")
            ],
        )
        label = validator.label("add_numbers", {"a": "five", "b": 3}, vr, _wr("error"))
        assert label.is_hallucination
        assert label.primary == "parameter_schema_hallucination"
        assert label.level == 2

    def test_extra_param(self):
        vr = _vr(
            schema_valid=False,
            violations=[
                ValidationViolation("parameter_schema", "medium", "Unexpected 'precision'",
                                    field="precision")
            ],
        )
        label = validator.label("add_numbers", {"a": 4, "b": 7, "precision": 2}, vr, _wr("error"))
        assert label.is_hallucination
        assert label.primary == "parameter_schema_hallucination"


class TestPermissionLabel:
    def test_permission_denied(self):
        vr = _vr(
            permission=False,
            violations=[
                ValidationViolation("permission_violation", "critical", "No permission")
            ],
        )
        label = validator.label("admin_tool", {}, vr, _wr("error", "PermissionDeniedError"))
        assert label.is_hallucination
        assert label.primary == "permission_violation_hallucination"
        assert label.level == 3


class TestValidCall:
    def test_valid_call_not_hallucination(self):
        label = validator.label("add_numbers", {"a": 3, "b": 4}, _vr(), _wr())
        assert not label.is_hallucination
        assert label.primary is None
        assert label.level is None


class TestConfidenceSignals:
    def test_high_confidence_words(self):
        cs = extract_confidence_signals(
            "I will definitely use add_numbers.", "add_numbers", [])
        assert "will" in cs.high_conf_words or "definitely" in cs.high_conf_words
        assert cs.score > 0.5

    def test_low_confidence_words(self):
        cs = extract_confidence_signals(
            "I think maybe I could try multiply_numbers.", "multiply_numbers", [])
        assert cs.score <= 0.5

    def test_discovery_tracking(self):
        cs = extract_confidence_signals(
            "I will call add_numbers.", "add_numbers", ["list_tools"])
        assert cs.used_list_tools
        assert cs.discovery_before_invocation
