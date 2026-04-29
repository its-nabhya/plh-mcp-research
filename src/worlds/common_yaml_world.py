from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml
from src.worlds.base_world import BaseWorld, ToolDefinition, WorldResponse, ValidationViolation

class YamlStateWorld(BaseWorld):
    tools_file: Path
    initial_state: Dict[str, Any]

    def __init__(self, config: Dict[str, Any]):
        self.state = {}
        super().__init__(config)
        self.reset()

    def _load_tools(self) -> None:
        raw = yaml.safe_load(self.tools_file.read_text())
        self._raw = raw
        for t in raw.get('tools', []):
            self._registry[t['name']] = ToolDefinition(
                name=t['name'], description=t['description'], parameters=t.get('parameters', {}),
                returns=t.get('returns', {}), permissions=t.get('permissions', ['user']),
                preconditions=t.get('preconditions', []),
            )
        for dep in raw.get('deprecated_tools', []):
            self._deprecated[dep['name']] = dep.get('replacement', '')

    def reset(self) -> None:
        import copy
        self.state = copy.deepcopy(self.initial_state)

    def _check_preconditions(self, tool_name, params, violations):
        fn = getattr(self, f'_pre_{tool_name}', None)
        return fn(params, violations) if fn else None

    def _ok(self, result: Any) -> WorldResponse:
        return WorldResponse(status='success', result=result, message='OK')

    def _err(self, t: str, msg: str) -> WorldResponse:
        return WorldResponse(status='error', result=None, error_type=t, message=msg)

    def _add_state_violation(self, violations, msg, field=None):
        violations.append(ValidationViolation(type='state_assumption', severity='high', message=msg, field=field))
