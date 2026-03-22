class MicroWorld:
    def __init__(self):
        self.tools = {
            "add_numbers": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["a", "b"]
                }
            }
        }

    def execute(self, tool_call):
        name = tool_call["name"]
        params = tool_call["parameters"]

        if name == "add_numbers":
            return {
                "status": "success",
                "output": {"result": params["a"] + params["b"]}
            }

        return {
            "status": "error",
            "output": {"message": "Unknown tool"}
        }