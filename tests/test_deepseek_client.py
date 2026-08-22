from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from backend.app.llm.deepseek import (
    DEEPSEEK_CHAT_URL,
    DeepSeekAPIError,
    DeepSeekClient,
)


class FakeResponse:
    def __init__(self, payload, status_code: int = 200, headers: dict | None = None) -> None:
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self.payload


class FakeHTTP:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class AnswerSchema(BaseModel):
    answer: str


def tool_response():
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": json.dumps({"query": "Mercado Libre"}),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
        },
    }


def test_select_tool_uses_authenticated_non_thinking_deepseek_request() -> None:
    http = FakeHTTP(FakeResponse(tool_response()))
    client = DeepSeekClient("secret", "deepseek-v4-flash", http_client=http)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Busca",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    response = client.select_tool([{"role": "user", "content": "investigá"}], tools)

    url, request = http.calls[0]
    assert url == DEEPSEEK_CHAT_URL
    assert request["headers"]["Authorization"] == "Bearer secret"
    assert request["json"]["model"] == "deepseek-v4-flash"
    assert request["json"]["thinking"] == {"type": "disabled"}
    assert request["json"]["tool_choice"] == "required"
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].arguments == {"query": "Mercado Libre"}
    assert response.usage["cache_hit_tokens"] == 80


def test_generate_json_sends_schema_instruction_and_json_mode() -> None:
    payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"answer":"ok"}'},
            }
        ],
        "usage": {},
    }
    http = FakeHTTP(FakeResponse(payload))
    client = DeepSeekClient("secret", "deepseek-v4-flash", http_client=http)

    result = client.generate_json(
        "Sistema",
        "Respondé",
        AnswerSchema,
        max_tokens=100,
        temperature=0.1,
    )

    body = http.calls[0][1]["json"]
    assert body["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in body["messages"][0]["content"]
    assert result == {"answer": "ok"}


@pytest.mark.parametrize("status_code", [401, 402, 429, 500, 503])
def test_http_errors_are_sanitized(status_code: int) -> None:
    client = DeepSeekClient(
        "secret",
        "deepseek-v4-flash",
        http_client=FakeHTTP(FakeResponse({"secret": "provider body"}, status_code)),
    )

    with pytest.raises(DeepSeekAPIError) as caught:
        client.generate_text("Sistema", "Hola", max_tokens=20, temperature=0.1)

    assert caught.value.status_code == status_code
    assert "provider body" not in str(caught.value)
    assert client.last_usage["provider_requests"] == 1
    assert client.last_usage["failed_request"] == 1
    assert client.last_usage["deepseek_ms"] >= 0


def test_invalid_tool_arguments_are_rejected() -> None:
    payload = tool_response()
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "not-json"
    client = DeepSeekClient(
        "secret", "deepseek-v4-flash", http_client=FakeHTTP(FakeResponse(payload))
    )

    with pytest.raises(DeepSeekAPIError, match="invalid tool arguments"):
        client.select_tool([{"role": "user", "content": "hola"}], [])
