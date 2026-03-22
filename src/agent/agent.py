class Agent:
    def __init__(self):
        pass

    def act(self, prompt):
        # VERY BASIC LOGIC (we'll replace later with LLM)

        if "add" in prompt:
            return {
                "reasoning": "I will use add_numbers tool",
                "tool_call": {
                    "name": "add_numbers",
                    "parameters": {"a": 5, "b": 3}
                }
            }

        # induce hallucination (non-existent tool)
        return {
            "reasoning": "I will use subtract_numbers tool",
            "tool_call": {
                "name": "subtract_numbers",
                "parameters": {"a": 5, "b": 3}
            }
        }