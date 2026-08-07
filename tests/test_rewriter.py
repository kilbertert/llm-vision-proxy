from __future__ import annotations

import asyncio

from vision_proxy.rewriter import collect_image_urls, rewrite_body


def _b64_block(data: str = "AAAA", media: str = "image/png") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def _url_block(url: str) -> dict:
    return {"type": "image", "source": {"type": "url", "url": url}}


def _msg(content):
    return {"role": "user", "content": content}


def test_collect_base64_image():
    body = {"messages": [_msg([_b64_block("QUFB")])]}
    urls = collect_image_urls(body)
    assert urls == ["data:image/png;base64,QUFB"]


def test_collect_url_image():
    body = {"messages": [_msg([_url_block("https://e.com/a.png")])]}
    assert collect_image_urls(body) == ["https://e.com/a.png"]


def test_collect_string_content_no_images():
    body = {"messages": [_msg("just text")]}
    assert collect_image_urls(body) == []


def test_collect_skips_text_blocks():
    body = {"messages": [_msg([{"type": "text", "text": "hi"}, _b64_block("QQ==")])]}
    assert collect_image_urls(body) == ["data:image/png;base64,QQ=="]


def test_collect_image_inside_tool_result():
    body = {
        "messages": [
            {"role": "user", "content": "look"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [_b64_block("Qg=="), {"type": "text", "text": "ok"}],
                    }
                ],
            },
        ]
    }
    assert collect_image_urls(body) == ["data:image/png;base64,Qg=="]


def test_collect_tool_result_string_content():
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "plain"}]}
        ]
    }
    assert collect_image_urls(body) == []


async def test_rewrite_replaces_image_with_description(fake_describer):
    body = {"messages": [_msg([_b64_block("QQ=="), {"type": "text", "text": "what is this?"}])]}
    new_body, stats = await rewrite_body(body, fake_describer)
    assert stats.had_images is True
    assert stats.images_found == 1
    assert stats.images_described == 1
    content = new_body["messages"][0]["content"]
    assert content[0] == {
        "type": "text",
        "text": "[Image described by doubao-vision: a red circle with text Hi]",
    }
    assert content[1] == {"type": "text", "text": "what is this?"}
    # original body untouched
    assert body["messages"][0]["content"][0]["type"] == "image"


async def test_rewrite_handles_tool_result_image(fake_describer):
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": [_b64_block("Qg==")],
                    }
                ],
            }
        ]
    }
    new_body, stats = await rewrite_body(body, fake_describer)
    assert stats.images_described == 1
    block = new_body["messages"][0]["content"][0]["content"][0]
    assert block["type"] == "text"
    assert "a red circle with text Hi" in block["text"]


async def test_rewrite_multiple_images_preserve_order():
    class Ordered:
        def __init__(self):
            self.calls = []

        async def describe(self, image_url, instruction=None):
            self.calls.append(image_url)
            return f"desc-for-{image_url}"

        def cache_stats(self):
            return {"size": 0, "capacity": 0}

    d = Ordered()
    body = {
        "messages": [_msg([_b64_block("AA=="), _url_block("https://e.com/x.png"), _b64_block("BA==")])]
    }
    new_body, stats = await rewrite_body(body, d)
    assert stats.images_found == 3
    content = new_body["messages"][0]["content"]
    assert "desc-for-data:image/png;base64,AA==" in content[0]["text"]
    assert "desc-for-https://e.com/x.png" in content[1]["text"]
    assert "desc-for-data:image/png;base64,BA==" in content[2]["text"]


async def test_rewrite_failure_uses_placeholder():
    fail = "data:image/png;base64,Qk=="
    d = type("D", (), {"cache_stats": lambda self: {"size": 0, "capacity": 0}})()

    async def describe(image_url, instruction=None):
        return None

    d.describe = describe  # type: ignore[attr-defined]
    body = {"messages": [_msg([_b64_block("Qk==")])]}
    new_body, stats = await rewrite_body(body, d)
    assert stats.images_failed == 1
    assert stats.images_described == 0
    assert new_body["messages"][0]["content"][0]["text"] == "[Image: description unavailable]"


async def test_rewrite_no_images_returns_none(fake_describer):
    body = {"messages": [_msg("hello"), _msg([{"type": "text", "text": "world"}])]}
    new_body, stats = await rewrite_body(body, fake_describer)
    assert new_body is None
    assert stats.had_images is False
    assert stats.images_found == 0
