import jsonschema

class Validator:
    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def validate(self, tool_call):
        violations = []

        tool_name = tool_call["name"]
        params = tool_call.get("parameters", {})

        # 1. Tool existence check
        if tool_name not in self.tool_registry:
            violations.append({
                "type": "tool_existence",
                "severity": "critical",
                "message": f"{tool_name} not found"
            })
            return self._build_response(False, False, None, violations)

        tool_schema = self.tool_registry[tool_name]["schema"]

        # 2. Schema validation
        try:
            jsonschema.validate(instance=params, schema=tool_schema)
            schema_valid = True
        except jsonschema.ValidationError as e:
            schema_valid = False
            violations.append({
                "type": "parameter_schema",
                "severity": "high",
                "message": str(e)
            })

        # 3. Permission check (dummy for now)
        permission_granted = True

        return self._build_response(
            True,
            schema_valid,
            permission_granted,
            violations
        )

    def _build_response(self, tool_exists, schema_valid, permission_granted, violations):
        return {
            "tool_exists": tool_exists,
            "schema_valid": schema_valid,
            "permission_granted": permission_granted,
            "violations": violations
        }