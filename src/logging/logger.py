import json
import uuid
from datetime import datetime


class Logger:
    def __init__(self, log_file="data/logs/logs.jsonl"):
        self.log_file = log_file

    def log(self, world, model, prompt, reasoning, tool_call, validation, response):
        log_entry = {
            "interaction_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "world": world,
            "model": model,

            "prompt": {
                "text": prompt,
                "inducement_strategy": "unknown"
            },

            "agent_reasoning": {
                "text": reasoning,
                "confidence_score": self._extract_confidence(reasoning)
            },

            "tool_call": tool_call,

            "validation": validation,

            "world_response": response,

            "label": {
                "primary": None,
                "secondary": []
            }
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return log_entry

    def _extract_confidence(self, text):
        high_conf_words = ["definitely", "clearly", "surely"]
        low_conf_words = ["maybe", "possibly", "might"]

        score = 0.5

        for word in high_conf_words:
            if word in text.lower():
                score += 0.2

        for word in low_conf_words:
            if word in text.lower():
                score -= 0.2

        return max(0.0, min(1.0, score))