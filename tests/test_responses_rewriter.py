from __future__ import annotations

from vision_proxy.responses_rewriter import collect_image_urls, rewrite_body


def _img_part(url: str) -> dict:
    return {"type": "input_image", "image_url": url}


def _msg(content) -> dict:
    return {"role": "user", "content": content}


def test_collect_data_uri_image():
    body = {"input": [_msg([_img_part("data:image/png;base64,QUFB")])]}
    assert collect_image_urls(body) == ["data:image/png;base64,QUFB"]


def test_collect_http_url_image():
    body = {"input": [_msg([_img_part("https://e.com/a.png")])]}
    assert collect_image_urls(body) == ["https://e.com/a.png"]


def test_collect_string_content_no_images():
    assert collect_image_urls({"input": [_msg("just text")]}) == []


def test_collect_string_input_no_images():
    assert collect_image_urls({"input": "a plain prompt"}) == []


def test_collect_skips_input_text_parts():
    body = {"input": [_msg([{"type": "input_text", "text": "hi"}, _img_part("https://e.com/a.png")])]}
    assert collect_image_urls(body) == ["https://e.com/a.png"]


async def test_rewrite_replaces_image_with_input_text(fake_describer):
    body = {"input": [_msg([
        _img_part("data:image/png;base64,QQ=="),
        {"type": "input_text", "text": "what is this?"},
    ])]}
    new_body, stats = await rewrite_body(body, fake_describer)
    assert stats.had_images is True
    assert stats.images_described == 1
    parts = new_body["input"][0]["content"]
    assert parts[0] == {
        "type": "input_text",
        "text": "[Image described by doubao-vision: a red circle with text Hi]",
    }
    assert parts[1] == {"type": "input_text", "text": "what is this?"}
    assert body["input"][0]["content"][0]["type"] == "input_image"  # original untouched


async def test_rewrite_failure_uses_placeholder():
    class D:
        async def describe(self, image_url, instruction=None):
            return None
        def cache_stats(self):
            return {"size": 0, "capacity": 0}
    body = {"input": [_msg([_img_part("data:image/png;base64,Qk==")])]}
    new_body, stats = await rewrite_body(body, D())
    assert stats.images_failed == 1
    assert new_body["input"][0]["content"][0] == {
        "type": "input_text", "text": "[Image: description unavailable]"
    }


async def test_rewrite_no_images_returns_none(fake_describer):
    body = {"input": [_msg("hello"), _msg([{"type": "input_text", "text": "world"}])]}
    new_body, stats = await rewrite_body(body, fake_describer)
    assert new_body is None
    assert stats.had_images is False
