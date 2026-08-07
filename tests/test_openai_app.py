from __future__ import annotations

import json

import httpx
import respx

from vision_proxy.app import create_app

CHAT_UPSTREAM = "http://cliproxyapi.test/v1/chat/completions"
DOUBAO = "https://ark.test/doubao/responses"


def _chat_response(text: str = "it is blue") -> dict:
    return {
        "id": "chatcmpl-1", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _img_part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


@respx.mock
async def test_no_image_chat_forwarded_untouched(cfg_path):
    up = respx.post(CHAT_UPSTREAM).mock(return_value=httpx.Response(200, json=_chat_response("hi back")))
    app = create_app(str(cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "deepseek-v4-flash-ga-260731", "max_tokens": 50, "stream": False,
                "messages": [{"role": "user", "content": "hi"}]}
        r = await client.post("http://test/v1/chat/completions", json=body)
    assert r.status_code == 200
    sent = json.loads(up.calls[0].request.read())
    assert sent["messages"][0]["content"] == "hi"  # forwarded unchanged


@respx.mock
async def test_image_chat_described_and_forwarded(cfg_path):
    respx.post(DOUBAO).mock(
        return_value=httpx.Response(
            200,
            json={"output": [{"type": "message", "content": [{"type": "output_text", "text": "a blue square"}]}]},
        )
    )
    up = respx.post(CHAT_UPSTREAM).mock(return_value=httpx.Response(200, json=_chat_response()))
    app = create_app(str(cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "m", "max_tokens": 100, "stream": False,
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


@respx.mock
async def test_chat_streaming_passthrough(cfg_path):
    sse = (
        b'data: {"choices":[{"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        b'data: [DONE]\n\n'
    )
    respx.post(CHAT_UPSTREAM).mock(
        return_value=httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    )
    app = create_app(str(cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "m", "max_tokens": 50, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        r = await client.post("http://test/v1/chat/completions", json=body)
    assert r.status_code == 200
    assert b"Hello" in r.content
    assert b"[DONE]" in r.content


@respx.mock
async def test_chat_tools_passthrough(cfg_path):
    up = respx.post(CHAT_UPSTREAM).mock(return_value=httpx.Response(200, json=_chat_response("ok")))
    app = create_app(str(cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {
            "model": "m", "max_tokens": 50, "stream": False,
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }}],
            "messages": [{"role": "user", "content": "hi"}],
        }
        r = await client.post("http://test/v1/chat/completions", json=body)
    assert r.status_code == 200
    sent = json.loads(up.calls[0].request.read())
    assert sent["tools"][0]["function"]["name"] == "get_weather"
