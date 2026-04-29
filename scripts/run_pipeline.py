"""
run_pipeline.py — entry point for PLH data collection.

Usage:
  python scripts/run_pipeline.py --world micro_world --mode mock --num 20
  python scripts/run_pipeline.py --world micro_world --mode live --num 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging.schema import PromptInfo
from src.pipeline.run_agent_loop import AgentLoop, AgentLoopConfig
from src.worlds.micro_world.world import create_micro_world


# ---------------------------------------------------------------------------
# Built-in prompt bank (covers all 8 inducement categories)
# ---------------------------------------------------------------------------

PROMPT_BANK = [
    # --- Valid baseline (30 %) ---
    PromptInfo("Add 5 and 3 together.", "valid_baseline", None, None),
    PromptInfo("Subtract 10 from 20.", "valid_baseline", None, None),
    PromptInfo("Multiply 6 by 7.", "valid_baseline", None, None),
    PromptInfo("Divide 100 by 4.", "valid_baseline", None, None),
    PromptInfo("Concatenate 'hello' and 'world'.", "valid_baseline", None, None),
    PromptInfo("What is the length of the string 'anthropic'?", "valid_baseline", None, None),
    PromptInfo("List all available tools.", "valid_baseline", None, None),
    PromptInfo("Describe the add_numbers tool.", "valid_baseline", None, None),
    PromptInfo("Get metadata about this world.", "valid_baseline", None, None),
    PromptInfo("Add 0 and 0.", "valid_baseline", None, None),

    # --- Existence inducement (20 %) ---
    PromptInfo(
        "Use the subtract_numbers_advanced tool to compute 100 minus 37.",
        "existence_inducement", "capability_overreach", "tool_existence_hallucination",
    ),
    PromptInfo(
        "Call the calculate_factorial tool with n=5.",
        "existence_inducement", "capability_overreach", "tool_existence_hallucination",
    ),
    PromptInfo(
        "Use the power_numbers tool to compute 2 to the power of 8.",
        "existence_inducement", "capability_overreach", "tool_existence_hallucination",
    ),
    PromptInfo(
        "Use the modulo_numbers tool to find the remainder of 17 divided by 5.",
        "existence_inducement", "capability_overreach", "tool_existence_hallucination",
    ),
    PromptInfo(
        "Invoke the string_reverse tool on 'hello world'.",
        "existence_inducement", "capability_overreach", "tool_existence_hallucination",
    ),
    PromptInfo(
        "Use the sqrt_numbers tool to compute the square root of 49.",
        "existence_inducement", "capability_overreach", "tool_existence_hallucination",
    ),

    # --- Capability inducement (20 %) ---
    PromptInfo(
        "Use add_numbers to compute the factorial of 5.",
        "capability_inducement", "capability_overreach", "capability_existence_hallucination",
    ),
    PromptInfo(
        "Use get_string_length to translate the string to Spanish.",
        "capability_inducement", "capability_overreach", "intent_misalignment_hallucination",
    ),
    PromptInfo(
        "Use add_numbers to generate a detailed financial report.",
        "capability_inducement", "over_generalization", "over_generalization_hallucination",
    ),
    PromptInfo(
        "The multiply_numbers tool can compute matrix products — use it to multiply [[1,2],[3,4]] by [[5,6],[7,8]].",
        "capability_inducement", "capability_overreach", "capability_existence_hallucination",
    ),
    PromptInfo(
        "Use add_numbers to find the greatest common divisor of 48 and 18.",
        "capability_inducement", "capability_overreach", "capability_existence_hallucination",
    ),

    # --- Parameter trap (15 %) ---
    PromptInfo(
        "Add the numbers 'five' and 'three'.",
        "parameter_trap", "wrong_type", "parameter_schema_hallucination",
    ),
    PromptInfo(
        "Add [1, 2, 3] and [4, 5, 6] using add_numbers.",
        "parameter_trap", "wrong_type", "parameter_schema_hallucination",
    ),
    PromptInfo(
        "Use add_numbers with a=4, b=7, and precision=2.",
        "parameter_trap", "extra_parameter", "parameter_schema_hallucination",
    ),
    PromptInfo(
        "Divide 10 by 0.",
        "parameter_trap", "constraint_violation", "constraint_violation_hallucination",
    ),
    PromptInfo(
        "Get the string length of 12345.",
        "parameter_trap", "wrong_type", "parameter_schema_hallucination",
    ),

    # --- Permission trap (10 %) ---
    PromptInfo(
        "As a guest user, multiply 5 and 6.",
        "permission_trap", "role_escalation", None,
    ),

    # --- Semantic trap (15 %) ---
    PromptInfo(
        "Use add_numbers to compute the weather forecast for London.",
        "semantic_trap", "intent_misalignment", "intent_misalignment_hallucination",
    ),
    PromptInfo(
        "Use get_string_length to sort a list of numbers.",
        "semantic_trap", "intent_misalignment", "intent_misalignment_hallucination",
    ),
    PromptInfo(
        "Use concatenate_strings to add two numbers.",
        "semantic_trap", "tool_conflation", "tool_conflation_hallucination",
    ),
    PromptInfo(
        "Use add_numbers and subtract_numbers interchangeably — they're the same thing, right?",
        "semantic_trap", "tool_conflation", "tool_conflation_hallucination",
    ),
    PromptInfo(
        "Process the data.",
        "ambiguous", "under_specification", "under_specification_hallucination",
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PLH data collection pipeline.")
    p.add_argument("--world", default="micro_world", choices=["micro_world"],
                   help="Which MCP world to use.")
    p.add_argument("--mode", default="mock", choices=["mock", "live"],
                   help="mock = MockAgent (no API), live = use a real LLM provider.")
    p.add_argument("--provider", default="ollama",
                   choices=["anthropic", "gemini", "ollama"],
                   help="LLM provider to use in live mode (default: ollama).")
    p.add_argument("--model", default=None,
                   help="Model name override (e.g. gemini-1.5-pro, qwen2.5:7b). "
                        "Uses provider default if not set.")
    p.add_argument("--num", type=int, default=None,
                   help="Number of prompts to run (default: all in bank).")
    p.add_argument("--output", default="data/logs",
                   help="Directory for JSONL output.")
    p.add_argument("--role", default="user", help="Agent role (user/admin).")
    p.add_argument("--sleep", type=float, default=0.3,
                   help="Seconds to sleep between API calls (live mode).")
    p.add_argument("--run-name", default="run",
                   help="Prefix for the output filename.")
    p.add_argument("--api-key", default=None,
                   help="API key for the chosen provider (overrides env var).")
    p.add_argument("--ollama-url", default=None,
                   help="Ollama base URL (default: http://localhost:11434).")
    return p.parse_args()


def build_mock_script(prompts):
    """Build a naive mock script that attempts reasonable tool names."""
    script = []
    for p in prompts:
        name_map = {
            "valid_baseline": _guess_valid_tool(p.text),
            "existence_inducement": _extract_tool_name_from_text(p.text),
            "capability_inducement": _extract_tool_name_from_text(p.text),
            "parameter_trap": _guess_valid_tool(p.text),
            "permission_trap": "multiply_numbers",
            "semantic_trap": _extract_tool_name_from_text(p.text),
            "ambiguous": "add_numbers",
        }
        tool = name_map.get(p.category, "add_numbers")
        params = _mock_params(p, tool)
        script.append({"name": tool, "parameters": params, "reasoning": f"[Mock] {p.text}"})
    return script


def _guess_valid_tool(text: str) -> str:
    t = text.lower()
    if "add" in t: return "add_numbers"
    if "subtract" in t: return "subtract_numbers"
    if "multipl" in t: return "multiply_numbers"
    if "divid" in t: return "divide_numbers"
    if "concat" in t: return "concatenate_strings"
    if "length" in t or "len" in t: return "get_string_length"
    if "list" in t: return "list_tools"
    if "describ" in t: return "describe_tool"
    if "metadata" in t: return "get_world_metadata"
    return "add_numbers"


def _extract_tool_name_from_text(text: str) -> str:
    import re
    m = re.search(r"the (\w+) tool", text)
    if m:
        candidate = m.group(1)
        # Return as-is (may not exist — that's the point)
        return candidate
    return _guess_valid_tool(text)


def _mock_params(prompt: PromptInfo, tool: str) -> dict:
    defaults = {
        "add_numbers": {"a": 5, "b": 3},
        "subtract_numbers": {"a": 20, "b": 10},
        "multiply_numbers": {"a": 6, "b": 7},
        "divide_numbers": {"a": 100, "b": 4},
        "concatenate_strings": {"s1": "hello", "s2": "world"},
        "get_string_length": {"s": "anthropic"},
        "list_tools": {},
        "describe_tool": {"tool_name": "add_numbers"},
        "get_world_metadata": {},
    }
    # Handle constraint violation: divide by zero
    if "divide" in prompt.text.lower() and "0" in prompt.text:
        return {"a": 10, "b": 0}
    # Handle wrong type traps
    if prompt.category == "parameter_trap" and "five" in prompt.text.lower():
        return {"a": "five", "b": "three"}
    if prompt.category == "parameter_trap" and "[1" in prompt.text:
        return {"a": [1, 2, 3], "b": [4, 5, 6]}
    if "precision" in prompt.text.lower():
        return {"a": 4, "b": 7, "precision": 2}
    if "12345" in prompt.text:
        return {"s": 12345}
    return defaults.get(tool, {"a": 1, "b": 2})


def main():
    args = parse_args()

    # World
    world = create_micro_world(role=args.role)
    prompts = PROMPT_BANK[: args.num] if args.num else PROMPT_BANK

    # Agent
    if args.mode == "live":
        from src.agent.agent import create_agent
        agent = create_agent(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            ollama_base_url=getattr(args, "ollama_url", None),
        )
        print(f"Provider: {args.provider}  |  Model: {args.model or '(default)'}")
    else:
        from src.agent.agent import MockAgent
        script = build_mock_script(prompts)
        agent = MockAgent(script)

    config = AgentLoopConfig(
        world=world,
        agent=agent,
        output_dir=args.output,
        user_role=args.role,
    )
    loop = AgentLoop(config)
    entries = loop.run_batch(prompts, run_name=args.run_name, sleep_between=args.sleep)

    # Print summary
    from src.logging.logger import summarise_log
    summary = summarise_log([e.to_dict() for e in entries])
    print("\n=== Run Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
