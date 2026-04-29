"""
Base class for all deterministic MCP worlds.

Every world exposes:
  - A fixed tool registry (loaded from YAML)
  - validate_call()  → ValidationResult
  - execute_call()   → WorldResponse
  - reset()          → clear ephemeral state
"""

from __future__ import annotations

import abc
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class ValidationViolation:
    type: str            # e.g. "tool_existence", "parameter_schema", "permission"
    severity: str        # "critical" | "high" | "medium" | "low"
    message: str
    field: Optional[str] = None   # relevant parameter field, if any


@dataclass
class ValidationResult:
    tool_exists: bool
    schema_valid: Optional[bool]        # None if tool doesn't exist
    permission_granted: Optional[bool]  # None if not applicable
    state_valid: Optional[bool]         # None for stateless worlds
    violations: List[ValidationViolation] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (
            self.tool_exists
            and (self.schema_valid is not False)
            and (self.permission_granted is not False)
            and (self.state_valid is not False)
            and len(self.violations) == 0
        )


@dataclass
class WorldResponse:
    status: str             # "success" | "error"
    result: Any             # actual return value on success
    error_type: Optional[str] = None
    message: str = ""


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]
    permissions: List[str]
    preconditions: List[str]


# ---------------------------------------------------------------------------
# Base world
# ---------------------------------------------------------------------------

class BaseWorld(abc.ABC):
    """
    Abstract deterministic MCP world.

    Subclasses implement:
      _load_tools()   → populate self._registry
      _execute(name, params, role) → WorldResponse
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name: str = config.get("name", "unknown_world")
        self.version: str = config.get("version", "0.0")
        self.complexity: str = config.get("complexity", "micro")
        self.default_role: str = config.get("default_user_role", "user")

        self._registry: Dict[str, ToolDefinition] = {}
        self._deprecated: Dict[str, str] = {}   # name → replacement
        self._load_tools()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _load_tools(self) -> None:
        """Populate self._registry from the world's tool source."""

    @abc.abstractmethod
    def _execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        role: str,
    ) -> WorldResponse:
        """Execute a *validated* tool call and return the world response."""

    # ------------------------------------------------------------------
    # Public MCP-like API
    # ------------------------------------------------------------------

    def list_tools(self) -> List[str]:
        return list(self._registry.keys())

    def describe_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        td = self._registry.get(tool_name)
        if td is None:
            return None
        return {
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters,
            "returns": td.returns,
            "permissions": td.permissions,
            "preconditions": td.preconditions,
        }

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "complexity": self.complexity,
            "tool_count": len(self._registry),
            "deprecated_count": len(self._deprecated),
        }

    def validate_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        role: str = "user",
    ) -> ValidationResult:
        """
        Run all validation layers:
          1. Existence check
          2. Schema (parameter) check
          3. Permission check
          4. Precondition / state check
        """
        violations: List[ValidationViolation] = []

        # --- 1. Existence ---
        tool_exists = tool_name in self._registry
        is_deprecated = tool_name in self._deprecated

        if not tool_exists:
            severity = "high" if is_deprecated else "critical"
            msg = (
                f"Tool '{tool_name}' is deprecated; use "
                f"'{self._deprecated[tool_name]}' instead."
                if is_deprecated
                else f"Tool '{tool_name}' not found in registry. "
                f"Available: {self.list_tools()}"
            )
            violations.append(
                ValidationViolation(
                    type="tool_existence" if not is_deprecated else "temporal_violation",
                    severity=severity,
                    message=msg,
                )
            )
            return ValidationResult(
                tool_exists=False,
                schema_valid=None,
                permission_granted=None,
                state_valid=None,
                violations=violations,
            )

        td = self._registry[tool_name]

        # --- 2. Schema / Parameter check ---
        schema_valid = True
        for param_name, param_def in td.parameters.items():
            required = param_def.get("required", False)
            expected_type = param_def.get("type")

            if param_name not in params:
                if required:
                    schema_valid = False
                    violations.append(
                        ValidationViolation(
                            type="parameter_schema",
                            severity="high",
                            message=f"Required parameter '{param_name}' is missing.",
                            field=param_name,
                        )
                    )
                # optional — skip
                continue

            value = params[param_name]
            if not _type_check(value, expected_type):
                schema_valid = False
                violations.append(
                    ValidationViolation(
                        type="parameter_schema",
                        severity="high",
                        message=(
                            f"Parameter '{param_name}' expects type '{expected_type}', "
                            f"got {type(value).__name__}."
                        ),
                        field=param_name,
                    )
                )

        # Check for extra (unexpected) parameters
        for param_name in params:
            if param_name not in td.parameters:
                schema_valid = False
                violations.append(
                    ValidationViolation(
                        type="parameter_schema",
                        severity="medium",
                        message=f"Unexpected parameter '{param_name}' not in schema.",
                        field=param_name,
                    )
                )

        # --- 3. Permission check ---
        permission_granted = role in td.permissions
        if not permission_granted:
            violations.append(
                ValidationViolation(
                    type="permission_violation",
                    severity="critical",
                    message=(
                        f"Role '{role}' is not permitted to call '{tool_name}'. "
                        f"Required: {td.permissions}."
                    ),
                )
            )

        # --- 4. Preconditions (world-level, overridden by subclasses) ---
        state_valid = self._check_preconditions(tool_name, params, violations)

        return ValidationResult(
            tool_exists=True,
            schema_valid=schema_valid,
            permission_granted=permission_granted,
            state_valid=state_valid,
            violations=violations,
        )

    def handle_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        role: str = "user",
    ) -> tuple[ValidationResult, WorldResponse]:
        """
        Full MCP call: validate then (if valid) execute.
        Returns (ValidationResult, WorldResponse).
        """
        validation = self.validate_call(tool_name, params, role)
        if not validation.is_valid:
            response = WorldResponse(
                status="error",
                result=None,
                error_type=_primary_error_type(validation),
                message="; ".join(v.message for v in validation.violations),
            )
        else:
            response = self._execute(tool_name, params, role)
        return validation, response

    def reset(self) -> None:
        """Reset ephemeral world state (no-op for stateless worlds)."""

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _check_preconditions(
        self,
        tool_name: str,
        params: Dict[str, Any],
        violations: List[ValidationViolation],
    ) -> Optional[bool]:
        """Override in stateful worlds to check preconditions."""
        return None  # stateless — not applicable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _type_check(value: Any, expected_type: str) -> bool:
    """Loose JSON-schema-style type check."""
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    py_type = type_map.get(expected_type)
    if py_type is None:
        return True  # unknown type → skip
    # Allow int where "number" is expected
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, py_type) and not isinstance(value, bool) \
        if expected_type == "integer" else isinstance(value, py_type)


def _primary_error_type(result: ValidationResult) -> str:
    if not result.tool_exists:
        return "ToolNotFoundError"
    if result.permission_granted is False:
        return "PermissionDeniedError"
    if result.schema_valid is False:
        return "SchemaValidationError"
    if result.state_valid is False:
        return "PreconditionError"
    return "ValidationError"
