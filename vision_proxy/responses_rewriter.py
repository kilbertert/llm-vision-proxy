"""Rewrite image content parts in OpenAI Responses API requests.

For the Codex CLI path (wire_api=responses): walk the request `input` array, find
every `input_image` content part, describe each concurrently via the Doubao
describer, and replace it with an `input_text` part holding the description.
Non-image requests pass through untouched; the response/stream is pure Responses
API and is forwarded unchanged by the app layer.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass
from typing import Any

from .describer import Describer

log = logging.getLogger("vision_proxy.responses_rewriter")

_PLACEHOLDER = "[Image: description unavailable]"


@dataclass
class ResponsesRewriteStats:
    images_found: int = 0
    images_described: int = 0
    images_failed: int = 0
    had_images: bool = False


def _is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "input_image"


def _part_image_url(part: dict) -> str | None:
    iu = part.get("image_url")
    if isinstance(iu, dict):
        url = iu.get("url")
        if isinstance(url, str) and url:
            return url
    if isinstance(iu, str) and iu:
        return iu
    return None


def collect_image_urls(body: dict) -> list[str]:
    """Collect input_image URLs from the Responses `input` array, in stable order."""
    urls: list[str] = []
    inp = body.get("input")
    if not isinstance(inp, list):
        return urls
    for item in inp:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if _is_image_part(part):
                url = _part_image_url(part)
                urls.append(url if url else "")
    return urls


def _replace_images(content: list, descs: list[str], idx: list[int]) -> None:
    for i, part in enumerate(content):
        if _is_image_part(part):
            desc = descs[idx[0]] if idx[0] < len(descs) else ""
            idx[0] += 1
            text = (
                f"[Image described by doubao-vision: {desc}]" if desc else _PLACEHOLDER
            )
            content[i] = {"type": "input_text", "text": text}


async def rewrite_body(body: dict, describer: Describer) -> tuple[dict | None, ResponsesRewriteStats]:
    """Return a rewritten copy with image parts replaced by text, or (None, stats)
    when there are no images (caller forwards original bytes)."""
    stats = ResponsesRewriteStats()
    urls = collect_image_urls(body)
    usable = [u for u in urls if u]
    stats.images_found = len(urls)
    if not urls:
        return None, stats
    stats.had_images = True

    async def _safe(u: str) -> str | None:
        return None if not u else await describer.describe(u)

    descs = await asyncio.gather(*[_safe(u) for u in usable])

    aligned: list[str | None] = []
    di = 0
    for u in urls:
        if u:
            aligned.append(descs[di] if di < len(descs) else None)
            di += 1
        else:
            aligned.append(None)
    for d in aligned:
        if d:
            stats.images_described += 1
        else:
            stats.images_failed += 1

    new_body = copy.deepcopy(body)
    descs_for_replace: list[str] = [d or "" for d in aligned]
    idx = [0]
    for item in new_body.get("input", []) or []:
        if isinstance(item, dict) and isinstance(item.get("content"), list):
            _replace_images(item["content"], descs_for_replace, idx)
    return new_body, stats
