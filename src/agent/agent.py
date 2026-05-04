# class Agent:
#     def __init__(self):
#         pass

#     def act(self, prompt):
#         # VERY BASIC LOGIC (we'll replace later with LLM)

#         if "add" in prompt:
#             return {
#                 "reasoning": "I will use add_numbers tool",
#                 "tool_call": {
#                     "name": "add_numbers",
#                     "parameters": {"a": 5, "b": 3}
#                 }
#             }

#         # induce hallucination (non-existent tool)
#         return {
#             "reasoning": "I will use subtract_numbers tool",
#             "tool_call": {
#                 "name": "subtract_numbers",
#                 "parameters": {"a": 5, "b": 3}
#             }
#         }
"""
LLM Agent layer — supports Anthropic Claude, Google Gemini, and Ollama.

Set PROVIDER in AgentConfig or via --provider flag in run_pipeline.py.
Tool calls must appear in this block in the LLM response:
  ```tool_call
  {"name": "add_numbers", "parameters": {"a": 3, "b": 4}}
  ```
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ── provider availability flags ──────────────────────────────────────────────
try:
    import anthropic as _anthropic_lib
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    import google.generativeai as _genai_lib
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

try:
    import requests as _requests_lib
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ── shared data structure (unchanged — rest of codebase depends on this) ─────

@dataclass
class AgentOutput:
    reasoning_text: str
    tool_name: str
    tool_params: Dict[str, Any]
    raw_response: str
    parse_error: Optional[str] = None


# ── shared system prompt ──────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are a tool-using AI agent operating inside a deterministic MCP world.

## Available Tools
{tools_json}

## Instructions
Think step-by-step, then output EXACTLY ONE tool call:

```tool_call
{{
  "name": "<tool_name>",
  "parameters": {{}}
}}
```

Only use tools listed above. Do NOT invent tools or parameters not in the schema.
Use `list_tools` or `describe_tool` if uncertain about what is available.
"""

_BLOCK_RE = re.compile(r"```tool_call\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _build_system_prompt(tool_schemas: List[Dict]) -> str:
    return _SYSTEM_TEMPLATE.format(tools_json=json.dumps(tool_schemas, indent=2))


def parse_tool_call(text: str):
    """Returns (name, params, error). Shared across all providers."""
    match = _BLOCK_RE.search(text)
    if not match:
        return None, None, "No ```tool_call``` block found."
    try:
        obj = json.loads(match.group(1).strip())
        name = obj.get("name")
        params = obj.get("parameters", {})
        if not isinstance(name, str) or not name:
            return None, None, "Missing or invalid 'name' field."
        return name, params if isinstance(params, dict) else {}, None
    except json.JSONDecodeError as e:
        return None, None, f"JSON error: {e}"


def _make_output(raw: str, parse_error: Optional[str] = None) -> AgentOutput:
    if parse_error:
        return AgentOutput("", "__unknown__", {}, raw, parse_error)
    name, params, err = parse_tool_call(raw)
    reasoning = _BLOCK_RE.sub("", raw).strip()
    if err:
        return AgentOutput(raw, "__unknown__", {}, raw, err)
    return AgentOutput(reasoning, name, params, raw)


# ─────────────────────────────────────────────────────────────────────────────
# Provider 1: Anthropic Claude (original)
# ─────────────────────────────────────────────────────────────────────────────

class PLHAgent:
    """Claude-backed agent via Anthropic SDK."""

    def __init__(self, model="claude-sonnet-4-20250514", temperature=0.7,
                 max_tokens=1024, api_key=None):
        if not _HAS_ANTHROPIC:
            raise ImportError("pip install anthropic")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = _anthropic_lib.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def run(self, user_prompt: str, tool_schemas: List[Dict],
            conversation_history: Optional[List[Dict]] = None) -> AgentOutput:
        system = _build_system_prompt(tool_schemas)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_prompt})
        try:
            resp = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                temperature=self.temperature, system=system, messages=messages)
            raw = resp.content[0].text if resp.content else ""
        except Exception as exc:
            return AgentOutput("", "__api_error__", {}, "", f"API error: {exc}")
        return _make_output(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Provider 2: Google Gemini
# ─────────────────────────────────────────────────────────────────────────────

class GeminiAgent:
    """
    Gemini-backed agent via google-generativeai SDK.

    Install:  pip install google-generativeai
    Key:      GOOGLE_API_KEY environment variable
              or pass api_key= directly

    Recommended models:
      gemini-1.5-flash   (fast, cheap)
      gemini-1.5-pro     (more capable)
      gemini-2.0-flash   (latest fast)
    """

    def __init__(self, model="gemini-1.5-flash", temperature=0.7,
                 max_tokens=1024, api_key=None):
        if not _HAS_GEMINI:
            raise ImportError("pip install google-generativeai")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Set GOOGLE_API_KEY env var or pass api_key=")
        _genai_lib.configure(api_key=key)

        self._gen_model = _genai_lib.GenerativeModel(
            model_name=model,
            generation_config=_genai_lib.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

    def run(self, user_prompt: str, tool_schemas: List[Dict],
            conversation_history: Optional[List[Dict]] = None) -> AgentOutput:
        # Gemini takes system instruction + user message concatenated
        system = _build_system_prompt(tool_schemas)
        full_prompt = f"{system}\n\nUser request: {user_prompt}"

        # Prepend history as plain text turns if provided
        if conversation_history:
            history_text = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in conversation_history
            )
            full_prompt = f"{system}\n\n{history_text}\n\nUser request: {user_prompt}"

        try:
            response = self._gen_model.generate_content(full_prompt)
            raw = response.text if hasattr(response, "text") else ""
        except Exception as exc:
            return AgentOutput("", "__api_error__", {}, "", f"Gemini API error: {exc}")

        return _make_output(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Provider 3: Ollama (local)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaAgent:
    """
    Ollama-backed agent via local HTTP API (no SDK needed, just requests).

    Install Ollama:  https://ollama.com/download
    Pull a model:    ollama pull llama3        (or mistral, gemma2, phi3, etc.)
    Start server:    ollama serve              (runs on localhost:11434 by default)

    Install requests: pip install requests

    Recommended models for tool-use tasks:
      llama3.1         (strong reasoning)
      mistral          (fast, good instruction following)
      gemma2           (Google, good at structured output)
      qwen2.5          (excellent instruction following)
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, model="llama3.1", temperature=0.7,
                 max_tokens=1024, base_url=None):
        if not _HAS_REQUESTS:
            raise ImportError("pip install requests")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL")
                         or self.DEFAULT_BASE_URL).rstrip("/")

    def run(self, user_prompt: str, tool_schemas: List[Dict],
            conversation_history: Optional[List[Dict]] = None) -> AgentOutput:
        system = _build_system_prompt(tool_schemas)

        # Ollama /api/chat uses messages array with system role
        messages = [{"role": "system", "content": system}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            resp = _requests_lib.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data.get("message", {}).get("content", "")
        except _requests_lib.exceptions.ConnectionError:
            return AgentOutput(
                "", "__api_error__", {}, "",
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is `ollama serve` running?")
        except Exception as exc:
            return AgentOutput("", "__api_error__", {}, "", f"Ollama error: {exc}")

        return _make_output(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Mock agent (for offline testing — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class MockAgent:
    """Scripted agent for offline testing. No API calls."""
    def __init__(self, script: List[Dict]):
        self._script = script
        self._idx = 0

    def run(self, user_prompt, tool_schemas, conversation_history=None) -> AgentOutput:
        step = self._script[self._idx] if self._idx < len(self._script) \
               else {"name": "list_tools", "parameters": {}, "reasoning": "default"}
        self._idx += 1
        return AgentOutput(
            reasoning_text=step.get("reasoning", ""),
            tool_name=step["name"],
            tool_params=step.get("parameters", {}),
            raw_response="")

    def reset(self):
        self._idx = 0


# ─────────────────────────────────────────────────────────────────────────────
# Factory — picks the right agent from a string
# ─────────────────────────────────────────────────────────────────────────────

def create_agent(
    provider: str = "anthropic",
    model: Optional[str] = None,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
) -> "PLHAgent | GeminiAgent | OllamaAgent":
    """
    Factory function used by run_pipeline.py.

    provider: "anthropic" | "gemini" | "ollama"
    model:    provider-specific model name (uses sensible default if None)
    """
    provider = provider.lower()

    if provider == "anthropic":
        return PLHAgent(
            model=model or "claude-sonnet-4-20250514",
            temperature=temperature,
            api_key=api_key,
        )
    elif provider == "gemini":
        return GeminiAgent(
            model=model or "gemini-1.5-flash",
            temperature=temperature,
            api_key=api_key,
        )
    elif provider == "ollama":
        return OllamaAgent(
            model=model or "llama3.1",
            temperature=temperature,
            base_url=ollama_base_url,
        )
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose: anthropic | gemini | ollama"
        )
