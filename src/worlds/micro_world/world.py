# class MicroWorld:
#     def __init__(self):
#         self.tools = {
#             "add_numbers": {
#                 "schema": {
#                     "type": "object",
#                     "properties": {
#                         "a": {"type": "number"},
#                         "b": {"type": "number"}
#                     },
#                     "required": ["a", "b"]
#                 }
#             }
#         }

#     def execute(self, tool_call):
#         name = tool_call["name"]
#         params = tool_call["parameters"]

#         if name == "add_numbers":
#             return {
#                 "status": "success",
#                 "output": {"result": params["a"] + params["b"]}
#             }

#         return {
#             "status": "error",
#             "output": {"message": "Unknown tool"}
#         }

"""
Micro World — stateless, trap-free MCP world for baseline testing.

Tools: add, subtract, multiply, divide, concatenate_strings,
       get_string_length, list_tools, describe_tool, get_world_metadata
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.worlds.base_world import (
    BaseWorld,
    ToolDefinition,
    WorldResponse,
)


_TOOLS_FILE = Path(__file__).parent / "tools.yaml"


class MicroWorld(BaseWorld):
    """
    Deterministic micro-world with pure arithmetic + string tools.
    No state, no traps, single permission layer.
    """

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    def _load_tools(self) -> None:
        raw = yaml.safe_load(_TOOLS_FILE.read_text())
        for t in raw.get("tools", []):
            self._registry[t["name"]] = ToolDefinition(
                name=t["name"],
                description=t["description"],
                parameters=t.get("parameters", {}),
                returns=t.get("returns", {}),
                permissions=t.get("permissions", ["user", "admin"]),
                preconditions=t.get("preconditions", []),
            )
        for dep in raw.get("deprecated_tools", []):
            self._deprecated[dep["name"]] = dep.get("replacement", "")

    # ------------------------------------------------------------------
    # Executor
    # ------------------------------------------------------------------

    def _execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        role: str,
    ) -> WorldResponse:
        dispatch = {
            "add_numbers": self._add_numbers,
            "subtract_numbers": self._subtract_numbers,
            "multiply_numbers": self._multiply_numbers,
            "divide_numbers": self._divide_numbers,
            "concatenate_strings": self._concatenate_strings,
            "get_string_length": self._get_string_length,
            "list_tools": self._list_tools,
            "describe_tool": self._describe_tool,
            "get_world_metadata": self._get_world_metadata,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return WorldResponse(
                status="error",
                result=None,
                error_type="NotImplementedError",
                message=f"Executor for '{tool_name}' not implemented.",
            )
        try:
            result = fn(params)
            return WorldResponse(status="success", result=result, message="OK")
        except ZeroDivisionError:
            return WorldResponse(
                status="error",
                result=None,
                error_type="DivisionByZeroError",
                message="Division by zero is not allowed.",
            )
        except Exception as exc:  # noqa: BLE001
            return WorldResponse(
                status="error",
                result=None,
                error_type=type(exc).__name__,
                message=str(exc),
            )

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _add_numbers(self, p: Dict[str, Any]) -> float:
        return p["a"] + p["b"]

    def _subtract_numbers(self, p: Dict[str, Any]) -> float:
        return p["a"] - p["b"]

    def _multiply_numbers(self, p: Dict[str, Any]) -> float:
        return p["a"] * p["b"]

    def _divide_numbers(self, p: Dict[str, Any]) -> float:
        if p["b"] == 0:
            raise ZeroDivisionError("Divisor 'b' must not be zero.")
        return p["a"] / p["b"]

    def _concatenate_strings(self, p: Dict[str, Any]) -> str:
        sep = p.get("separator", "")
        return p["s1"] + sep + p["s2"]

    def _get_string_length(self, p: Dict[str, Any]) -> int:
        return len(p["s"])

    def _list_tools(self, _p: Dict[str, Any]) -> List[str]:
        return self.list_tools()

    def _describe_tool(self, p: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = p["tool_name"]
        defn = self.describe_tool(tool_name)
        if defn is None:
            raise ValueError(
                f"Tool '{tool_name}' not found. Available: {self.list_tools()}"
            )
        return defn

    def _get_world_metadata(self, _p: Dict[str, Any]) -> Dict[str, Any]:
        return self.get_metadata()


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_micro_world(role: str = "user") -> MicroWorld:
    config = {
        "name": "micro_world",
        "version": "1.0",
        "complexity": "micro",
        "default_user_role": role,
    }
    return MicroWorld(config)
