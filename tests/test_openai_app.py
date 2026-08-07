from __future__ import annotations

import json

import httpx
import pytest
import respx

from vision_proxy.app import create_app

CHAT_UPSTREAM = "https://ark.test/api/v3/chat/completions"
DOUBAO = "https://ark.test/api/v3/responses"

OPENAI_YAML = """
listen: 127.0.0.1:8417
doubao:
  base_url: https://ark.test/api/v3/responses
  api_key: ark-test-key
  model: doubao-seed-2-0-lite-260428
  max_output_tokens: 128
  timeout_s: 60
cache_size: 16
unknown_model: default_upstream
default_upstream: ds-ga
upstreams:
  ds-ga:
    base_url: https://ark.test/api/v3
    api_key: ark-ds-key
    model: deepseek-v4-flash-ga-260731
    format: openai
  deepseek-v4-flash:
    base_url: https://api.deepseek.com/anthropic
    api_key: ds-test-key
    model: deepseek-v4-flash
    format: anthropic
"""


@pytest.fixture
def openai_cfg_path(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(OPENAI_YAML)
    return p


def _chat_response(text: str = "it is blue") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _img_part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


@respx.mock
async def test_no_image_chat_forwarded_untouched(openai_cfg_path):
    up = respx.post(CHAT_UPSTREAM).mock(return_value=httpx.Response(200, json=_chat_response("hi back")))
    app = create_app(str(openai_cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "ds-ga", "max_tokens": 50, "stream": False,
                "messages": [{"role": "user", "content": "hi"}]}
        r = await client.post("http://test/v1/chat/completions", json=body)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi back"
    sent = json.loads(up.calls[0].request.read())
    assert sent["messages"][0]["content"] == "hi"  # unchanged
    assert sent["model"] == "deepseek-v4-flash-ga-260731"


@respx.mock
async def test_image_chat_described_and_forwarded(openai_cfg_path):
    respx.post(DOUBAO).mock(
        return_value=httpx.Response(
            200,
            json={"output": [{"type": "message", "content": [{"type": "output_text", "text": "a blue square"}]}]},
        )
    )
    up = respx.post(CHAT_UPSTREAM).mock(return_value=httpx.Response(200, json=_chat_response()))
    app = create_app(str(openai_cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "ds-ga", "max_tokens": 100, "stream": False,
                "messages": [{"role": "user", "content": [
                    _img_part("data:image/png;base64,QUFB"),
                    {"type": "text", "text": "what is it?"},
                ]}]}
        r = await client.post("http://test/v1/chat/completions", json=body)
    assert r.status_code == 200
    sent = json.loads(up.calls[0].request.read())
    parts = sent["messages"][0]["content"]
    assert parts[0]["type"] == "text"
    assert "a blue square" in parts[0]["text"]
    assert parts[1] == {"type": "text", "text": "what is it?"}


@respx.mock
async def test_chat_streaming_passthrough(openai_cfg_path):
    sse = (
        b'data: {"choices":[{"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        b'data: [DONE]\n\n'
    )
    respx.post(CHAT_UPSTREAM).mock(
        return_value=httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    )
    app = create_app(str(openai_cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "ds-ga", "max_tokens": 50, "stream": True,
                "messages": [{"role": "user", "content": "hi"}]}
        r = await client.post("http://test/v1/chat/completions", json=body)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    # Deltas arrive as separate SSE events, so check each substring independently.
    assert b"Hello" in r.content
    assert b"world" in r.content
    assert b"[DONE]" in r.content


@respx.mock
async def test_chat_format_mismatch_rejected(openai_cfg_path):
    # An anthropic-format model must NOT be reachable via the openai chat endpoint.
    app = create_app(str(openai_cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]}
        r = await client.post("http://test/v1/chat/completions", json=body)
    assert r.status_code == 400
    assert "not an openai-format upstream" in r.json()["error"]["message"]


@respx.mock
async def test_messages_format_mismatch_rejected(openai_cfg_path):
    # An openai-format model must NOT be reachable via the anthropic messages endpoint.
    app = create_app(str(openai_cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "ds-ga", "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}]}
        r = await client.post("http://test/v1/messages", json=body)
    assert r.status_code == 400
    assert "not an anthropic-format upstream" in r.json()["error"]["message"]


@respx.mock
async def test_chat_tools_passthrough(openai_cfg_path):
    up = respx.post(CHAT_UPSTREAM).mock(return_value=httpx.Response(200, json=_chat_response("ok")))
    app = create_app(str(openai_cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {
            "model": "ds-ga", "max_tokens": 50, "stream": False,
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        r = await client.post("http://test/v1/chat/completions", json=body)
    assert r.status_code == 200
    sent = json.loads(up.calls[0].request.read())
    assert sent["tools"][0]["function"]["name"] == "get_weather"  # tools forwarded unchanged
