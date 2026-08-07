from __future__ import annotations

import httpx
import pytest
import respx

from vision_proxy.config import DoubaoConfig
from vision_proxy.describer import Describer, _extract_text


def _dc() -> DoubaoConfig:
    return DoubaoConfig(
        base_url="https://ark.test/api/v3/responses",
        api_key="ark-test-key",
        model="doubao-seed-2-0-lite-260428",
        max_output_tokens=128,
        timeout_s=10.0,
    )


def _resp(*output) -> dict:
    return {"output": list(output)}


def test_extract_text_prefers_message():
    resp = _resp(
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]},
        {"type": "message", "content": [{"type": "output_text", "text": "a blue square"}]},
    )
    assert _extract_text(resp) == "a blue square"


def test_extract_text_reasoning_fallback():
    resp = _resp(
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "only reasoning here"}]},
    )
    assert _extract_text(resp) == "only reasoning here"


def test_extract_text_empty():
    assert _extract_text(_resp()) == ""
    assert _extract_text({}) == ""


@respx.mock
async def test_describe_calls_api_and_caches():
    route = respx.post("https://ark.test/api/v3/responses").mock(
        return_value=httpx.Response(
            200,
            json=_resp({"type": "message", "content": [{"type": "output_text", "text": "a red circle"}]}),
        )
    )
    d = Describer(_dc(), cache_size=8)
    img = "data:image/png;base64,QUFB"

    r1 = await d.describe(img)
    r2 = await d.describe(img)
    assert r1 == "a red circle"
    assert r2 == "a red circle"
    assert route.call_count == 1  # second call served from cache
    await d.aclose()


@respx.mock
async def test_describe_sends_correct_body():
    route = respx.post("https://ark.test/api/v3/responses").mock(
        return_value=httpx.Response(
            200, json=_resp({"type": "message", "content": [{"type": "output_text", "text": "x"}]})
        )
    )
    d = Describer(_dc())
    await d.describe("https://e.com/a.png")
    await d.aclose()

    assert route.called
    req = route.calls[0].request
    assert req.headers["authorization"] == "Bearer ark-test-key"
    body = req.read()
    import json

    payload = json.loads(body)
    assert payload["model"] == "doubao-seed-2-0-lite-260428"
    assert payload["store"] is False
    content = payload["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert content[1] == {"type": "input_image", "image_url": "https://e.com/a.png"}


@respx.mock
async def test_describe_returns_none_on_http_error():
    respx.post("https://ark.test/api/v3/responses").mock(
        return_value=httpx.Response(500, text="upstream boom")
    )
    d = Describer(_dc())
    r = await d.describe("data:image/png;base64,QUFB")
    assert r is None
    await d.aclose()


@respx.mock
async def test_describe_returns_none_on_empty_output():
    respx.post("https://ark.test/api/v3/responses").mock(
        return_value=httpx.Response(200, json=_resp())
    )
    d = Describer(_dc())
    r = await d.describe("data:image/png;base64,QUFB")
    assert r is None
    await d.aclose()
