from __future__ import annotations

from vision_proxy.openai_rewriter import collect_image_urls, rewrite_body


def _img_part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _msg(content) -> dict:
    return {"role": "user", "content": content}


def test_collect_data_uri_image():
    body = {"messages": [_msg([_img_part("data:image/png;base64,QUFB")])]}
    assert collect_image_urls(body) == ["data:image/png;base64,QUFB"]


def test_collect_http_url_image():
    body = {"messages": [_msg([_img_part("https://e.com/a.png")])]}
    assert collect_image_urls(body) == ["https://e.com/a.png"]


def test_collect_string_content_no_images():
    assert collect_image_urls({"messages": [_msg("just text")]}) == []


def test_collect_skips_text_parts():
    body = {"messages": [_msg([{"type": "text", "text": "hi"}, _img_part("https://e.com/a.png")])]}
    assert collect_image_urls(body) == ["https://e.com/a.png"]


def test_collect_multiple_images_across_messages():
    body = {"messages": [
        _msg([_img_part("https://e.com/a.png")]),
        _msg("intermediate text"),
        _msg([{"type": "text", "text": "q"}, _img_part("https://e.com/b.png")]),
    ]}
    assert collect_image_urls(body) == ["https://e.com/a.png", "https://e.com/b.png"]


async def test_rewrite_replaces_image_with_description(fake_describer):
    body = {"messages": [_msg([
        _img_part("data:image/png;base64,QQ=="),
        {"type": "text", "text": "what is this?"},
    ])]}
    new_body, stats = await rewrite_body(body, fake_describer)
    assert stats.had_images is True
    assert stats.images_described == 1
    parts = new_body["messages"][0]["content"]
    assert parts[0] == {
        "type": "text",
        "text": "[Image described by doubao-vision: a red circle with text Hi]",
    }
    assert parts[1] == {"type": "text", "text": "what is this?"}
    # original untouched
    assert body["messages"][0]["content"][0]["type"] == "image_url"


async def test_rewrite_multiple_preserve_order():
    class Ordered:
        async def describe(self, image_url, instruction=None):
            return f"desc-{image_url}"
        def cache_stats(self): return {"size": 0, "capacity": 0}
    d = Ordered()
    body = {"messages": [_msg([
        _img_part("https://e.com/a.png"), _img_part("https://e.com/b.png")
    ])]}
    new_body, stats = await rewrite_body(body, d)
    assert stats.images_found == 2
    parts = new_body["messages"][0]["content"]
    assert "desc-https://e.com/a.png" in parts[0]["text"]
    assert "desc-https://e.com/b.png" in parts[1]["text"]


async def test_rewrite_failure_uses_placeholder():
    class D:
        async def describe(self, image_url, instruction=None): return None
        def cache_stats(self): return {"size": 0, "capacity": 0}
    body = {"messages": [_msg([_img_part("data:image/png;base64,Qk==")])]}
    new_body, stats = await rewrite_body(body, D())
    assert stats.images_failed == 1
    assert new_body["messages"][0]["content"][0]["text"] == "[Image: description unavailable]"


async def test_rewrite_no_images_returns_none(fake_describer):
    body = {"messages": [_msg("hello"), _msg([{"type": "text", "text": "world"}])]}
    new_body, stats = await rewrite_body(body, fake_describer)
    assert new_body is None
    assert stats.had_images is False
