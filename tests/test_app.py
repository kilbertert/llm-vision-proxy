from __future__ import annotations

import json

import httpx
import respx

from vision_proxy.app import create_app

UPSTREAM = "http://cliproxyapi.test/v1/messages"
COUNT = "http://cliproxyapi.test/v1/messages/count_tokens"
DOUBAO = "https://ark.test/doubao/responses"


def _upstream_response(text: str = "it is a red circle") -> dict:
    return {
        "id": "msg_1", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _b64_block(data: str = "QUFB") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


@respx.mock
async def test_no_image_forwarded_untouched(cfg_path):
    up = respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=_upstream_response("hi back")))
    app = create_app(str(cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "deepseek-v4-flash", "max_tokens": 50, "stream": False,
                "messages": [{"role": "user", "content": "hi"}]}
        r = await client.post("http://test/v1/messages", json=body)
    assert r.status_code == 200
    sent = json.loads(up.calls[0].request.read())
    assert sent["messages"][0]["content"] == "hi"  # unchanged, forwarded as-is
    # forward auth uses the configured cliproxyapi key
    assert up.calls[0].request.headers["authorization"] == "Bearer cpa-test-key"


@respx.mock
async def test_image_described_and_forwarded(cfg_path):
    respx.post(DOUBAO).mock(
        return_value=httpx.Response(
            200,
            json={"output": [{"type": "message", "content": [{"type": "output_text", "text": "a red circle"}]}]},
        )
    )
    up = respx.post(UPSTREAM).mock(return_value=httpx.Response(200, json=_upstream_response()))
    app = create_app(str(cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "any-model", "max_tokens": 100, "stream": False,
                "messages": [{"role": "user", "content": [_b64_block(), {"type": "text", "text": "what is it?"}]}]}
        r = await client.post("http://test/v1/messages", json=body)
    assert r.status_code == 200
    sent = json.loads(up.calls[0].request.read())
    contents = sent["messages"][0]["content"]
    assert contents[0]["type"] == "text"
    assert "a red circle" in contents[0]["text"]
    assert contents[1] == {"type": "text", "text": "what is it?"}
    assert sent["model"] == "any-model"  # model untouched; cliproxyapi routes


@respx.mock
async def test_streaming_passthrough(cfg_path):
    sse = b'event: message_start\ndata: {"type":"message_start"}\n\nevent: content_block_delta\ndata: {"type":"text_delta","text":"hello"}\n\n'
    respx.post(UPSTREAM).mock(
        return_value=httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    )
    app = create_app(str(cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "m", "max_tokens": 50, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        r = await client.post("http://test/v1/messages", json=body)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert b"hello" in r.content


async def test_healthz(cfg_path):
    app = create_app(str(cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        h = await client.get("http://test/healthz")
        assert h.status_code == 200
        assert h.json()["forward"] == "http://cliproxyapi.test"


@respx.mock
async def test_count_tokens_strips_images(cfg_path):
    respx.post(DOUBAO).mock(
        return_value=httpx.Response(
            200,
            json={"output": [{"type": "message", "content": [{"type": "output_text", "text": "a circle"}]}]},
        )
    )
    ct = respx.post(COUNT).mock(return_value=httpx.Response(200, json={"input_tokens": 42}))
    app = create_app(str(cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "m", "messages": [{"role": "user", "content": [_b64_block(), {"type": "text", "text": "q"}]}]}
        r = await client.post("http://test/v1/messages/count_tokens", json=body)
    assert r.status_code == 200
    sent = json.loads(ct.calls[0].request.read())
    assert sent["messages"][0]["content"][0]["type"] == "text"
