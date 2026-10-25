import json

import pytest

from src.agent_models import (
    DEFAULT_MODELS,
    AnthropicModel,
    ModelError,
    ModelTurn,
    OpenAIModel,
    ToolCall,
    anthropic_payload,
    model_from_environment,
    openai_payload,
    parse_anthropic,
    parse_openai,
    read_env_file,
)

TOOLS = [
    {
        "name": "list_scenarios",
        "description": "What the run covers.",
        "input_schema": {"type": "object", "properties": {}},
    }
]
CALL = ToolCall("call-1", "list_scenarios", {})
CONVERSATION = [
    {"role": "user", "text": "What does the run cover?"},
    {"role": "assistant", "text": "", "tool_calls": [CALL]},
    {
        "role": "tool",
        "results": [
            {
                "id": "call-1",
                "name": "list_scenarios",
                "content": {"series": 2996},
                "is_error": False,
            }
        ],
    },
    {"role": "assistant", "text": "It covers 2,996 series.", "tool_calls": []},
    {"role": "user", "text": "Thanks"},
]


def test_anthropic_payload_uses_content_blocks():
    payload = anthropic_payload("claude-test", "System text", CONVERSATION, TOOLS)

    assert payload["model"] == "claude-test"
    assert payload["system"] == "System text"
    assert payload["max_tokens"] > 0
    messages = payload["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1]["content"] == [
        {"type": "tool_use", "id": "call-1", "name": "list_scenarios", "input": {}}
    ]
    result = messages[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "call-1"
    assert json.loads(result["content"]) == {"series": 2996}
    assert messages[3]["content"] == [
        {"type": "text", "text": "It covers 2,996 series."}
    ]
    assert payload["tools"][0]["input_schema"] == TOOLS[0]["input_schema"]


def test_anthropic_replies_are_read():
    turn = parse_anthropic(
        {
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "list_scenarios",
                    "input": {},
                },
            ]
        }
    )

    assert turn == ModelTurn(
        text="Let me check.", tool_calls=(ToolCall("toolu_1", "list_scenarios", {}),)
    )


def test_openai_payload_uses_tool_messages():
    payload = openai_payload("gpt-test", "System text", CONVERSATION, TOOLS)

    messages = payload["messages"]
    assert messages[0] == {"role": "system", "content": "System text"}
    assert messages[2]["tool_calls"][0]["function"] == {
        "name": "list_scenarios",
        "arguments": "{}",
    }
    assert messages[2]["content"] is None
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": json.dumps({"series": 2996}),
    }
    assert payload["tools"][0]["function"]["parameters"] == TOOLS[0]["input_schema"]


def test_openai_replies_are_read_and_bad_arguments_marked():
    turn = parse_openai(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "a",
                                "type": "function",
                                "function": {
                                    "name": "compare_costs",
                                    "arguments": '{"lead_time_days": 3}',
                                },
                            },
                            {
                                "id": "b",
                                "type": "function",
                                "function": {
                                    "name": "compare_costs",
                                    "arguments": "{not json",
                                },
                            },
                        ],
                    }
                }
            ]
        }
    )

    assert turn.text == ""
    assert turn.tool_calls[0].arguments == {"lead_time_days": 3}
    assert turn.tool_calls[1].arguments == {"_unreadable": True}


def test_env_files_are_read_leniently(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# keys\nANTHROPIC_API_KEY='sk-ant-test'\nexport SUPPLYPILOT_AGENT_MODEL = claude-test\n\nbroken line\n"
    )

    assert read_env_file(env) == {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "SUPPLYPILOT_AGENT_MODEL": "claude-test",
    }
    assert read_env_file(tmp_path / "missing.env") == {}


def test_the_model_comes_from_the_environment():
    assert model_from_environment({}) is None
    assert model_from_environment({"SUPPLYPILOT_AGENT_PROVIDER": "openai"}) is None

    anthropic = model_from_environment({"ANTHROPIC_API_KEY": "k"})
    assert isinstance(anthropic, AnthropicModel)
    assert anthropic.model == DEFAULT_MODELS["anthropic"]

    openai = model_from_environment(
        {"OPENAI_API_KEY": "k", "SUPPLYPILOT_AGENT_MODEL": "gpt-test"}
    )
    assert isinstance(openai, OpenAIModel)
    assert openai.model == "gpt-test"

    chosen = model_from_environment(
        {
            "ANTHROPIC_API_KEY": "a",
            "OPENAI_API_KEY": "o",
            "SUPPLYPILOT_AGENT_PROVIDER": "OpenAI",
        }
    )
    assert isinstance(chosen, OpenAIModel)

    with pytest.raises(ValueError, match="must be one of"):
        model_from_environment(
            {"SUPPLYPILOT_AGENT_PROVIDER": "gemini", "ANTHROPIC_API_KEY": "k"}
        )


def test_http_requests_and_errors():
    httpx = pytest.importorskip("httpx")
    seen = {}

    def answer(request):
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": "Hello."}]}
        )

    model = AnthropicModel(
        "secret-key", "claude-test", transport=httpx.MockTransport(answer)
    )
    turn = model.complete("System", [{"role": "user", "text": "Hi"}], TOOLS)

    assert turn.text == "Hello."
    assert seen["headers"]["x-api-key"] == "secret-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["model"] == "claude-test"

    def refuse(request):
        return httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})

    failing = OpenAIModel("secret-key", transport=httpx.MockTransport(refuse))
    with pytest.raises(
        ModelError, match="openai answered 401: invalid x-api-key"
    ) as error:
        failing.complete("System", [{"role": "user", "text": "Hi"}], TOOLS)
    assert "secret-key" not in str(error.value)

    def drop(request):
        raise httpx.ConnectError("offline", request=request)

    offline = AnthropicModel("secret-key", transport=httpx.MockTransport(drop))
    with pytest.raises(ModelError, match="Could not reach anthropic"):
        offline.complete("System", [{"role": "user", "text": "Hi"}], TOOLS)
