"""Rewrite image content blocks into text description blocks.

Walks an Anthropic Messages API request body, finds every `image` content block
(including those nested inside `tool_result` blocks), describes each concurrently
via the Doubao describer, and replaces each image block with a `text` block
holding the description. Non-image requests are passed through untouched.

Only the Anthropic content-block format is handled here (Claude Code target).
"""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass
from typing import Any

from .describer import Describer

log = logging.getLogger("vision_proxy.rewriter")

_PLACEHOLDER = "[Image: description unavailable]"


@dataclass
class RewriteStats:
    images_found: int = 0
    images_described: int = 0
    images_failed: int = 0
    had_images: bool = False


def _is_image_block(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == "image"


def _block_image_url(block: dict) -> str | None:
    """Turn an Anthropic image block into a value Doubao accepts as `image_url`."""
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    stype = source.get("type")
    if stype == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        if not data:
            return None
        return f"data:{media_type};base64,{data}"
    if stype == "url":
        url = source.get("url")
        if isinstance(url, str) and url:
            return url
    # Unknown source shape (e.g. {"type":"file","file_id":...}) - unsupported in v1.
    return None


def _collect_image_urls(content: Any, out: list[str]) -> None:
    """Walk a message's content (string or list of blocks) collecting image URLs."""
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if _is_image_block(block):
            url = _block_image_url(block)
            if url is not None:
                out.append(url)
            else:
                # Mark as an image we cannot handle so counts stay accurate.
                out.append("")
        elif block.get("type") == "tool_result":
            _collect_image_urls(block.get("content"), out)


def collect_image_urls(body: dict) -> list[str]:
    """Collect image source URLs for every message, in stable order."""
    urls: list[str] = []
    for msg in body.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        _collect_image_urls(msg.get("content"), urls)
    return urls


def _replace_images(content: Any, descs: list[str], idx: list[int]) -> None:
    """Replace image blocks in `content` with text blocks, consuming `descs` in order.

    `idx` is a one-element list holding the current index into `descs` (mutable
    counter so recursion shares state).
    """
    if not isinstance(content, list):
        return
    for i, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        if _is_image_block(block):
            if idx[0] < len(descs):
                desc = descs[idx[0]]
            else:
                desc = ""
            idx[0] += 1
            text = (
                f"[Image described by doubao-vision: {desc}]"
                if desc
                else _PLACEHOLDER
            )
            content[i] = {"type": "text", "text": text}
        elif block.get("type") == "tool_result":
            _replace_images(block.get("content"), descs, idx)


async def rewrite_body(body: dict, describer: Describer) -> tuple[dict | None, RewriteStats]:
    """Return a rewritten copy of `body` with images replaced by text.

    Returns (None, stats) when the body contains no images, signalling the caller
    to forward the original request bytes unchanged.
    """
    stats = RewriteStats()
    urls = collect_image_urls(body)
    # Drop empty entries (unsupported image sources) but still count them as found.
    usable = [u for u in urls if u]
    stats.images_found = len(urls)
    if not urls:
        return None, stats
    stats.had_images = True

    # Describe all usable images concurrently. Empty entries -> None (placeholder).
    async def _safe_describe(u: str) -> str | None:
        if not u:
            return None
        return await describer.describe(u)

    descs = await asyncio.gather(*[_safe_describe(u) for u in usable])

    # Re-align descs to the original `urls` order (empty entries have no desc).
    aligned: list[str] = []
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
    # Walk new_body and replace, mapping each found image to its aligned desc.
    # We must replace exactly len(urls) image blocks in order.
    descs_for_replace: list[str] = [d or "" for d in aligned]
    idx = [0]
    for msg in new_body.get("messages", []) or []:
        if isinstance(msg, dict):
            _replace_images(msg.get("content"), descs_for_replace, idx)
    return new_body, stats
