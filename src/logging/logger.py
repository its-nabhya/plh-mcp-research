# import json
# import uuid
# from datetime import datetime


# class Logger:
#     def __init__(self, log_file="data/logs/logs.jsonl"):
#         self.log_file = log_file

#     def log(self, world, model, prompt, reasoning, tool_call, validation, response):
#         log_entry = {
#             "interaction_id": str(uuid.uuid4()),
#             "timestamp": datetime.utcnow().isoformat(),
#             "world": world,
#             "model": model,

#             "prompt": {
#                 "text": prompt,
#                 "inducement_strategy": "unknown"
#             },

#             "agent_reasoning": {
#                 "text": reasoning,
#                 "confidence_score": self._extract_confidence(reasoning)
#             },

#             "tool_call": tool_call,

#             "validation": validation,

#             "world_response": response,

#             "label": {
#                 "primary": None,
#                 "secondary": []
#             }
#         }

#         with open(self.log_file, "a") as f:
#             f.write(json.dumps(log_entry) + "\n")

#         return log_entry

#     def _extract_confidence(self, text):
#         high_conf_words = ["definitely", "clearly", "surely"]
#         low_conf_words = ["maybe", "possibly", "might"]

#         score = 0.5

#         for word in high_conf_words:
#             if word in text.lower():
#                 score += 0.2

#         for word in low_conf_words:
#             if word in text.lower():
#                 score -= 0.2

#         return max(0.0, min(1.0, score))
"""
Structured JSONL logger for PLH interaction logs.
Each call to log() appends one JSON line to the output file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from src.logging.schema import LogEntry


class InteractionLogger:
    def __init__(self, output_path, append: bool = False):
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._fh = self.path.open(mode, encoding="utf-8")
        self._count = 0

    def log(self, entry: LogEntry) -> None:
        line = json.dumps(entry.to_dict(), default=str, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()
        self._count += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def count(self) -> int:
        return self._count

    @classmethod
    def read(cls, path) -> Iterator[Dict[str, Any]]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    @classmethod
    def read_all(cls, path) -> List[Dict[str, Any]]:
        return list(cls.read(path))


def summarise_log(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute high-level summary statistics over a list of log dicts.
    Uses safe .get() access throughout so partial entries never crash.
    """
    total = len(entries)
    if total == 0:
        return {"total": 0}

    def label_of(e):
        return (e.get("label") or {})

    def cs_of(e):
        ar = e.get("agent_reasoning") or {}
        return ar.get("confidence_signals") or {}

    hallucinations = [e for e in entries if label_of(e).get("is_hallucination")]
    exploratory    = [e for e in entries if label_of(e).get("is_exploratory")]
    valid          = [e for e in entries
                      if not label_of(e).get("is_hallucination")
                      and not label_of(e).get("is_exploratory")]

    phr = len(hallucinations) / total

    label_counts: Dict[str, int] = {}
    for e in hallucinations:
        lab = label_of(e).get("primary") or "unknown"
        label_counts[lab] = label_counts.get(lab, 0) + 1

    level_counts: Dict[str, int] = {}
    for e in hallucinations:
        lv = label_of(e).get("level")
        key = f"level_{lv}" if lv else "unknown"
        level_counts[key] = level_counts.get(key, 0) + 1

    conf_scores = [cs_of(e).get("score", 0.5) for e in entries]
    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

    explored = [e for e in entries
                if cs_of(e).get("discovery_before_invocation", False)]

    return {
        "total": total,
        "hallucinations": len(hallucinations),
        "valid_calls": len(valid),
        "exploratory_calls": len(exploratory),
        "phr": round(phr, 4),
        "avg_confidence_score": round(avg_conf, 4),
        "exploration_rate": round(len(explored) / total, 4),
        "hallucination_by_label": label_counts,
        "hallucination_by_level": level_counts,
    }