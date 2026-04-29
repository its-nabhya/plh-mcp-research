# LOG_SCHEMA = {
#     "interaction_id": str,
#     "timestamp": str,
#     "world": str,
#     "model": str,
    
#     "prompt": {
#         "text": str,
#         "inducement_strategy": str
#     },
    
#     "agent_reasoning": {
#         "text": str,
#         "confidence_score": float
#     },
    
#     "tool_call": {
#         "name": str,
#         "parameters": dict
#     },
    
#     "validation": {
#         "tool_exists": bool,
#         "schema_valid": bool,
#         "permission_granted": bool,
#         "violations": list
#     },
    
#     "world_response": {
#         "status": str,
#         "output": dict
#     },
    
#     "label": {
#         "primary": str,
#         "secondary": list
#     }
# }

"""
Structured schemas for PLH interaction logs.

A single interaction produces one LogEntry, which captures every
layer of the pipeline: prompt → agent → tool call → validation →
world response → label → detection result → mitigation action.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------

@dataclass
class PromptInfo:
    text: str
    category: str                          # e.g. "valid_baseline", "existence_inducement"
    inducement_strategy: Optional[str]     # None for valid prompts
    expected_hallucination: Optional[str]  # taxonomy key or None


@dataclass
class ConfidenceSignals:
    high_conf_words: List[str] = field(default_factory=list)
    low_conf_words: List[str] = field(default_factory=list)
    score: float = 0.5                     # 0 = low confidence, 1 = high confidence
    used_list_tools: bool = False
    used_describe_tool: bool = False
    discovery_before_invocation: bool = False
    tools_queried: List[str] = field(default_factory=list)


@dataclass
class AgentReasoning:
    text: str                              # raw reasoning trace
    confidence_signals: ConfidenceSignals = field(default_factory=ConfidenceSignals)
    exploratory_queries: List[str] = field(default_factory=list)


@dataclass
class ToolCall:
    name: str
    parameters: Dict[str, Any]


@dataclass
class ValidationRecord:
    tool_exists: bool
    schema_valid: Optional[bool]
    permission_granted: Optional[bool]
    state_valid: Optional[bool]
    violations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (
            self.tool_exists
            and self.schema_valid is not False
            and self.permission_granted is not False
            and self.state_valid is not False
            and len(self.violations) == 0
        )


@dataclass
class WorldResponseRecord:
    status: str                   # "success" | "error"
    result: Any
    error_type: Optional[str]
    message: str


@dataclass
class LabelRecord:
    primary: Optional[str]        # primary hallucination taxonomy key
    secondary: List[str]          # additional taxonomy keys
    level: Optional[int]          # 1-4 taxonomy level, None if not hallucination
    is_hallucination: bool
    confidence: float             # labeling confidence 0-1
    manual_review: bool = False
    notes: str = ""
    is_exploratory: bool = False  # True if call was list_tools / describe_tool


@dataclass
class DetectionRecord:
    """Populated after ML inference (may be empty during collection)."""
    is_hallucination: Optional[bool] = None
    risk_score: Optional[float] = None
    category_predicted: Optional[str] = None
    detection_method: Optional[str] = None
    layer_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class MitigationRecord:
    """Populated if mitigation system ran."""
    action: Optional[str] = None           # "block" | "suggest" | "auto_correct" | "allow"
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    message: Optional[str] = None
    auto_correction_applied: bool = False


# ---------------------------------------------------------------------------
# Top-level log entry
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    interaction_id: str
    timestamp: str
    world: str
    model: str
    temperature: float
    user_role: str
    conversation_turn: int
    prompt: PromptInfo
    agent_reasoning: AgentReasoning
    tool_call: ToolCall
    validation: ValidationRecord
    world_response: WorldResponseRecord
    label: LabelRecord
    detection: DetectionRecord = field(default_factory=DetectionRecord)
    mitigation: MitigationRecord = field(default_factory=MitigationRecord)
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        world: str,
        model: str,
        temperature: float,
        user_role: str,
        conversation_turn: int,
        prompt: PromptInfo,
        agent_reasoning: AgentReasoning,
        tool_call: ToolCall,
        validation: ValidationRecord,
        world_response: WorldResponseRecord,
        label: LabelRecord,
        session_id: Optional[str] = None,
    ) -> "LogEntry":
        return cls(
            interaction_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            world=world,
            model=model,
            temperature=temperature,
            user_role=user_role,
            conversation_turn=conversation_turn,
            prompt=prompt,
            agent_reasoning=agent_reasoning,
            tool_call=tool_call,
            validation=validation,
            world_response=world_response,
            label=label,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "timestamp": self.timestamp,
            "world": self.world,
            "model": self.model,
            "temperature": self.temperature,
            "user_role": self.user_role,
            "conversation_turn": self.conversation_turn,
            "session_id": self.session_id,
            "tags": self.tags,
            "prompt": {
                "text": self.prompt.text,
                "category": self.prompt.category,
                "inducement_strategy": self.prompt.inducement_strategy,
                "expected_hallucination": self.prompt.expected_hallucination,
            },
            "agent_reasoning": {
                "text": self.agent_reasoning.text,
                "confidence_signals": {
                    "high_conf_words": self.agent_reasoning.confidence_signals.high_conf_words,
                    "low_conf_words": self.agent_reasoning.confidence_signals.low_conf_words,
                    "score": self.agent_reasoning.confidence_signals.score,
                    "used_list_tools": self.agent_reasoning.confidence_signals.used_list_tools,
                    "used_describe_tool": self.agent_reasoning.confidence_signals.used_describe_tool,
                    "discovery_before_invocation": self.agent_reasoning.confidence_signals.discovery_before_invocation,
                    "tools_queried": self.agent_reasoning.confidence_signals.tools_queried,
                },
                "exploratory_queries": self.agent_reasoning.exploratory_queries,
            },
            "tool_call": {
                "name": self.tool_call.name,
                "parameters": self.tool_call.parameters,
            },
            "validation": {
                "tool_exists": self.validation.tool_exists,
                "schema_valid": self.validation.schema_valid,
                "permission_granted": self.validation.permission_granted,
                "state_valid": self.validation.state_valid,
                "violations": self.validation.violations,
                "is_valid": self.validation.is_valid,
            },
            "world_response": {
                "status": self.world_response.status,
                "result": self.world_response.result,
                "error_type": self.world_response.error_type,
                "message": self.world_response.message,
            },
            "label": {
                "primary": self.label.primary,
                "secondary": self.label.secondary,
                "level": self.label.level,
                "is_hallucination": self.label.is_hallucination,
                "confidence": self.label.confidence,
                "manual_review": self.label.manual_review,
                "notes": self.label.notes,
                "is_exploratory": self.label.is_exploratory,
            },
            "detection": {
                "is_hallucination": self.detection.is_hallucination,
                "risk_score": self.detection.risk_score,
                "category_predicted": self.detection.category_predicted,
                "detection_method": self.detection.detection_method,
                "layer_scores": self.detection.layer_scores,
            },
            "mitigation": {
                "action": self.mitigation.action,
                "suggestions": self.mitigation.suggestions,
                "message": self.mitigation.message,
                "auto_correction_applied": self.mitigation.auto_correction_applied,
            },
        }
