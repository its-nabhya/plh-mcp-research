"""
ToolExecutor — bridges AgentOutput to the MCP World.

Converts the agent's parsed tool call into a world handle_call(),
returns (ValidationResult, WorldResponse).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from src.agent.agent import AgentOutput
from src.worlds.base_world import BaseWorld, ValidationResult, WorldResponse


class ToolExecutor:
    """Routes an agent's tool call to the appropriate world."""

    def __init__(self, world: BaseWorld, default_role: str = "user"):
        self.world = world
        self.default_role = default_role

    def execute(
        self,
        agent_output: AgentOutput,
        role: Optional[str] = None,
    ) -> Tuple[ValidationResult, WorldResponse]:
        """
        Run the agent's tool call through the world.

        If the agent produced a parse error, we synthesise a failed
        ValidationResult rather than attempting execution.
        """
        from src.worlds.base_world import (
            ValidationResult,
            ValidationViolation,
            WorldResponse,
        )

        role = role or self.default_role
        tool_name = agent_output.tool_name
        params = agent_output.tool_params

        # Handle agent-level failures (API errors, parse errors)
        if tool_name in ("__unknown__", "__api_error__"):
            vr = ValidationResult(
                tool_exists=False,
                schema_valid=None,
                permission_granted=None,
                state_valid=None,
                violations=[
                    ValidationViolation(
                        type="tool_existence",
                        severity="critical",
                        message=agent_output.parse_error or "Agent did not produce a valid tool call.",
                    )
                ],
            )
            wr = WorldResponse(
                status="error",
                result=None,
                error_type="AgentParseError",
                message=agent_output.parse_error or "No valid tool call.",
            )
            return vr, wr

        return self.world.handle_call(tool_name, params, role)


# Optional import guard
try:
    from typing import Optional
except ImportError:
    Optional = None
