from __future__ import annotations

import json

import httpx
import pytest
import respx

from vision_proxy.app import create_app

RESP_UPSTREAM = "https://ark.test/api/v3/responses"
DOUBAO = "https://ark.test/api/v3/responses"  # same host, different role (describer)

RESPONSES_YAML = """
listen: 127.0.0.1:8417
doubao:
  base_url: https://ark.test/doubao/responses
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
    format: openai_responses
  ds-chat:
    base_url: https://ark.test/api/v3
    api_key: ark-ds-key
    model: deepseek-v4-flash-ga-260731
    format: openai
"""

DOUBAO_URL = "https://ark.test/doubao/responses"


@pytest.fixture
def resp_cfg_path(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(RESPONSES_YAML)
    return p


def _resp_response(text: str = "it is blue") -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "deepseek-v4-flash-ga-260731",
        "output": [
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": text}]}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def _img_part(url: str) -> dict:
    return {"type": "input_image", "image_url": url}


@respx.mock
async def test_no_image_responses_forwarded_untouched(resp_cfg_path):
    up = respx.post(RESP_UPSTREAM).mock(return_value=httpx.Response(200, json=_resp_response("hi back")))
    app = create_app(str(resp_cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "ds-ga", "stream": False,
                "input": [{"role": "user", "content": "hi"}]}
        r = await client.post("http://test/v1/responses", json=body)
    assert r.status_code == 200
    sent = json.loads(up.calls[0].request.read())
    assert sent["input"][0]["content"] == "hi"  # unchanged
    assert sent["model"] == "deepseek-v4-flash-ga-260731"


@respx.mock
async def test_image_responses_described_and_forwarded(resp_cfg_path):
    respx.post(DOUBAO_URL).mock(
        return_value=httpx.Response(
            200,
            json={"output": [{"type": "message", "content": [{"type": "output_text", "text": "a blue square"}]}]},
        )
    )
    up = respx.post(RESP_UPSTREAM).mock(return_value=httpx.Response(200, json=_resp_response()))
    app = create_app(str(resp_cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "ds-ga", "stream": False,
                "input": [{"role": "user", "content": [
                    _img_part("data:image/png;base64,QUFB"),
                    {"type": "input_text", "text": "what is it?"},
                ]}]}
        r = await client.post("http://test/v1/responses", json=body)
    assert r.status_code == 200
    sent = json.loads(up.calls[0].request.read())
    parts = sent["input"][0]["content"]
    assert parts[0]["type"] == "input_text"
    assert "a blue square" in parts[0]["text"]
    assert parts[1] == {"type": "input_text", "text": "what is it?"}


@respx.mock
async def test_responses_streaming_passthrough(resp_cfg_path):
    sse = (
        b'event: response.created\n'
        b'data: {"type":"response.created","response":{"id":"r1"}}\n\n'
        b'event: response.output_text.delta\n'
        b'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
        b'event: response.completed\n'
        b'data: {"type":"response.completed","response":{"id":"r1"}}\n\n'
    )
    respx.post(RESP_UPSTREAM).mock(
        return_value=httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    )
    app = create_app(str(resp_cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "ds-ga", "stream": True,
                "input": [{"role": "user", "content": "hi"}]}
        r = await client.post("http://test/v1/responses", json=body)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert b"Hello" in r.content
    assert b"response.completed" in r.content


@respx.mock
async def test_responses_format_mismatch_rejected(resp_cfg_path):
    # An openai (chat) model must NOT be reachable via the responses endpoint.
    app = create_app(str(resp_cfg_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        body = {"model": "ds-chat", "input": [{"role": "user", "content": "hi"}]}
        r = await client.post("http://test/v1/responses", json=body)
    assert r.status_code == 400
    assert "not an openai_responses-format upstream" in r.json()["error"]["message"]
