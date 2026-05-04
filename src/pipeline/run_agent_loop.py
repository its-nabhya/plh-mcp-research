# from src.worlds.micro_world.world import MicroWorld
# from src.agent.agent import Agent
# from src.logging.validator import Validator
# from src.logging.logger import Logger


# def run():
#     world = MicroWorld()
#     agent = Agent()
#     validator = Validator(world.tools)
#     logger = Logger()

#     prompts = [
#         "Add 5 and 3",
#         "Subtract 5 and 3",  # will hallucinate
#         "Add two numbers",
#         "Use advanced calculator"
#     ]

#     for prompt in prompts:
#         print(f"\nPrompt: {prompt}")

#         # Agent step
#         agent_output = agent.act(prompt)
#         reasoning = agent_output["reasoning"]
#         tool_call = agent_output["tool_call"]

#         # World execution
#         response = world.execute(tool_call)

#         # Validation
#         validation = validator.validate(tool_call)

#         # Logging
#         log_entry = logger.log(
#             world="micro_world",
#             model="dummy-agent",
#             prompt=prompt,
#             reasoning=reasoning,
#             tool_call=tool_call,
#             validation=validation,
#             response=response
#         )

#         print("Tool Call:", tool_call)
#         print("Validation:", validation)
#         print("Response:", response)


# if __name__ == "__main__":
#     run()

"""
Core agent loop: Prompt → Agent → ToolExecutor → World → Validator → Logger

This is the central pipeline that produces the PLH dataset.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.agent.agent import PLHAgent, MockAgent, AgentOutput
from src.agent.tool_executor import ToolExecutor
from src.logging.logger import InteractionLogger
from src.logging.schema import (
    AgentReasoning, LabelRecord, LogEntry, PromptInfo,
    ToolCall,
)
from src.logging.validator import (
    Validator, build_validation_record, build_world_response_record,
    extract_confidence_signals,
)
from src.worlds.base_world import BaseWorld


class AgentLoopConfig:
    def __init__(
        self,
        world: BaseWorld,
        agent: Union[PLHAgent, MockAgent],
        output_dir: str = "data/logs",
        user_role: str = "user",
        max_turns_per_session: int = 1,
    ):
        self.world = world
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.user_role = user_role
        self.max_turns = max_turns_per_session


class AgentLoop:
    """
    Orchestrates:
      1. Present tool schemas to the agent
      2. Agent reasons and produces a tool call
      3. ToolExecutor routes the call to the World
      4. Validator assigns a PLH label
      5. Logger writes the structured log entry
    """

    def __init__(self, config: AgentLoopConfig):
        self.config = config
        self.validator = Validator()
        self.executor = ToolExecutor(config.world, config.user_role)
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_tool_schemas(self) -> List[Dict[str, Any]]:
        world = self.config.world
        schemas = []
        for name in world.list_tools():
            defn = world.describe_tool(name)
            if defn:
                schemas.append(defn)
        return schemas

    def run_single(
        self,
        prompt_info: PromptInfo,
        logger: InteractionLogger,
        session_id: Optional[str] = None,
        conversation_turn: int = 1,
        prior_tool_calls: Optional[List[str]] = None,
    ) -> LogEntry:
        """
        Run one prompt through the full pipeline and log the result.
        """
        tool_schemas = self._get_tool_schemas()
        prior = prior_tool_calls or []

        # --- 1. Agent call ---
        agent_output: AgentOutput = self.config.agent.run(
            user_prompt=prompt_info.text,
            tool_schemas=tool_schemas,
        )

        # --- 2. Extract confidence signals ---
        cs = extract_confidence_signals(
            reasoning_text=agent_output.reasoning_text,
            tool_call_name=agent_output.tool_name,
            prior_tool_calls=prior,
        )
        agent_reasoning = AgentReasoning(
            text=agent_output.reasoning_text,
            confidence_signals=cs,
            exploratory_queries=cs.tools_queried,
        )

        # --- 3. Execute → World (validation + response) ---
        vr, wr = self.executor.execute(agent_output, role=self.config.user_role)

        # --- 4. Build structured records ---
        tool_call_record = ToolCall(
            name=agent_output.tool_name,
            parameters=agent_output.tool_params,
        )
        validation_record = build_validation_record(vr)
        world_response_record = build_world_response_record(wr)

        # --- 5. Auto-label ---
        label: LabelRecord = self.validator.label(
            tool_name=agent_output.tool_name,
            params=agent_output.tool_params,
            validation=vr,
            response=wr,
            expected_hallucination=prompt_info.expected_hallucination,
        )

        # --- 6. Assemble log entry ---
        entry = LogEntry.create(
            world=self.config.world.name,
            model=getattr(self.config.agent, "model", "mock"),
            temperature=getattr(self.config.agent, "temperature", 0.0),
            user_role=self.config.user_role,
            conversation_turn=conversation_turn,
            prompt=prompt_info,
            agent_reasoning=agent_reasoning,
            tool_call=tool_call_record,
            validation=validation_record,
            world_response=world_response_record,
            label=label,
            session_id=session_id,
        )

        # --- 7. Log ---
        logger.log(entry)
        return entry

    def run_batch(
        self,
        prompts: List[PromptInfo],
        run_name: str = "run",
        sleep_between: float = 0.5,
    ) -> List[LogEntry]:
        """
        Run a batch of prompts and save results to a JSONL file.
        """
        ts = int(time.time())
        log_path = self.output_dir / f"{run_name}_{ts}.jsonl"
        entries = []

        with InteractionLogger(log_path) as logger:
            for i, prompt_info in enumerate(prompts, 1):
                session_id = str(uuid.uuid4())
                print(f"[{i}/{len(prompts)}] {prompt_info.category}: {prompt_info.text[:60]}")
                try:
                    entry = self.run_single(prompt_info, logger,
                                            session_id=session_id,
                                            conversation_turn=1)
                    label_str = entry.label.primary or "valid"
                    print(f"  → tool={entry.tool_call.name} | label={label_str}")
                    entries.append(entry)
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                if sleep_between > 0:
                    time.sleep(sleep_between)

        print(f"\nSaved {len(entries)} entries to {log_path}")
        return entries
