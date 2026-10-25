"""Model providers for the planning agent, behind one small interface.

The agent keeps its conversation in a neutral form:

    {"role": "user", "text": ...}
    {"role": "assistant", "text": ..., "tool_calls": [ToolCall, ...]}
    {"role": "tool", "results": [{"id", "name", "content", "is_error"}, ...]}

Each provider turns it into its own request and its reply back into a
ModelTurn. Building and reading payloads are plain functions, tested without
a network; only the client classes send HTTP requests (with httpx).

Configuration comes from environment variables, optionally read from a
git-ignored .env file:

    ANTHROPIC_API_KEY or OPENAI_API_KEY   the key (one is enough)
    SUPPLYPILOT_AGENT_PROVIDER            anthropic or openai (default: the key that is set)
    SUPPLYPILOT_AGENT_MODEL               model name (defaults below)
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-5-mini"}
KEY_VARIABLES = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MAX_TOKENS = 1024
TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ModelTurn:
    """One model reply: text, tool calls, or both."""

    text: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


class ModelError(RuntimeError):
    """The provider could not be reached or refused the request."""


class ModelClient(Protocol):
    provider: str
    model: str

    def complete(
        self, system: str, conversation: list[dict], tools: list[dict]
    ) -> ModelTurn: ...


# Anthropic Messages API


def anthropic_payload(
    model: str, system: str, conversation: list[dict], tools: list[dict]
) -> dict:
    messages = []
    for item in conversation:
        if item["role"] == "user":
            messages.append({"role": "user", "content": item["text"]})
        elif item["role"] == "assistant":
            blocks = (
                [{"type": "text", "text": item["text"]}] if item.get("text") else []
            )
            blocks += [
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in item.get("tool_calls", ())
            ]
            messages.append({"role": "assistant", "content": blocks})
        elif item["role"] == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result["id"],
                            "content": json.dumps(result["content"]),
                            "is_error": result["is_error"],
                        }
                        for result in item["results"]
                    ],
                }
            )
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": messages,
        "tools": [
            {
                "name": spec["name"],
                "description": spec["description"],
                "input_schema": spec["input_schema"],
            }
            for spec in tools
        ],
    }


def parse_anthropic(body: dict) -> ModelTurn:
    text, calls = [], []
    for block in body.get("content", []):
        if block.get("type") == "text":
            text.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            arguments = block.get("input")
            calls.append(
                ToolCall(
                    id=str(block.get("id", "")),
                    name=str(block.get("name", "")),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )
    return ModelTurn(text="".join(text).strip(), tool_calls=tuple(calls))


# OpenAI Chat Completions API


def openai_payload(
    model: str, system: str, conversation: list[dict], tools: list[dict]
) -> dict:
    messages = [{"role": "system", "content": system}]
    for item in conversation:
        if item["role"] == "user":
            messages.append({"role": "user", "content": item["text"]})
        elif item["role"] == "assistant":
            message = {"role": "assistant", "content": item.get("text") or None}
            calls = item.get("tool_calls", ())
            if calls:
                message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in calls
                ]
            messages.append(message)
        elif item["role"] == "tool":
            messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": result["id"],
                    "content": json.dumps(result["content"]),
                }
                for result in item["results"]
            )
    return {
        "model": model,
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["input_schema"],
                },
            }
            for spec in tools
        ],
    }


def parse_openai(body: dict) -> ModelTurn:
    choices = body.get("choices") or [{}]
    message = choices[0].get("message") or {}
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = None
        calls.append(
            ToolCall(
                id=str(call.get("id", "")),
                name=str(function.get("name", "")),
                # Unreadable arguments reach the tool as a non-object and get an error back.
                arguments=arguments
                if isinstance(arguments, dict)
                else {"_unreadable": True},
            )
        )
    return ModelTurn(
        text=(message.get("content") or "").strip(), tool_calls=tuple(calls)
    )


# HTTP clients


def _error_message(body) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
    return "no details"


class _HttpModel:
    provider = ""
    url = ""

    def __init__(self, api_key: str, model: str | None = None, transport=None):
        if not api_key:
            raise ValueError(f"An API key is needed for {self.provider}")
        self._api_key = api_key
        self.model = model or DEFAULT_MODELS[self.provider]
        self._transport = transport

    def _headers(self) -> dict:
        raise NotImplementedError

    def _post(self, payload: dict) -> dict:
        import httpx  # Imported here so the rest of the agent works without it.

        try:
            with httpx.Client(
                transport=self._transport, timeout=TIMEOUT_SECONDS
            ) as http:
                response = http.post(self.url, headers=self._headers(), json=payload)
        except httpx.HTTPError as exc:
            raise ModelError(
                f"Could not reach {self.provider}: {exc.__class__.__name__}"
            ) from exc

        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code >= 400:
            raise ModelError(
                f"{self.provider} answered {response.status_code}: {_error_message(body)}"
            )
        if not isinstance(body, dict):
            raise ModelError(f"{self.provider} sent a reply that is not JSON")
        return body


class AnthropicModel(_HttpModel):
    provider = "anthropic"
    url = ANTHROPIC_URL

    def _headers(self) -> dict:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def complete(
        self, system: str, conversation: list[dict], tools: list[dict]
    ) -> ModelTurn:
        return parse_anthropic(
            self._post(anthropic_payload(self.model, system, conversation, tools))
        )


class OpenAIModel(_HttpModel):
    provider = "openai"
    url = OPENAI_URL

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def complete(
        self, system: str, conversation: list[dict], tools: list[dict]
    ) -> ModelTurn:
        return parse_openai(
            self._post(openai_payload(self.model, system, conversation, tools))
        )


MODEL_CLASSES = {"anthropic": AnthropicModel, "openai": OpenAIModel}


# Configuration


def read_env_file(path) -> dict[str, str]:
    """KEY=VALUE lines from a .env file; comments, blanks and missing files are fine."""
    file = Path(path)
    if not file.is_file():
        return {}
    values = {}
    for line in file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


def model_from_environment(
    environ: Mapping[str, str], transport=None
) -> ModelClient | None:
    """The configured model, or None when no key is set."""
    provider = environ.get("SUPPLYPILOT_AGENT_PROVIDER", "").strip().lower()
    if not provider:
        provider = next(
            (
                name
                for name, variable in KEY_VARIABLES.items()
                if environ.get(variable, "").strip()
            ),
            "",
        )
    if not provider:
        return None
    if provider not in MODEL_CLASSES:
        raise ValueError(
            f"SUPPLYPILOT_AGENT_PROVIDER must be one of {sorted(MODEL_CLASSES)}, not {provider!r}"
        )
    key = environ.get(KEY_VARIABLES[provider], "").strip()
    if not key:
        return None
    model = environ.get("SUPPLYPILOT_AGENT_MODEL", "").strip() or None
    return MODEL_CLASSES[provider](key, model, transport=transport)
