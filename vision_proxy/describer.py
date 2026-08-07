"""Doubao vision describer.

Calls the Volcano Engine Ark Responses API to turn an image into a text description.
The description replaces the image block in the forwarded request so the text-only
backend can "see" it. Includes an in-memory LRU cache keyed by the image source, so
the same image (resent every turn by Claude Code) is described only once.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

import httpx

from .config import DoubaoConfig

log = logging.getLogger("vision_proxy.describer")

# Fixed instruction so descriptions are stable and cacheable across turns. The
# downstream text model gets a rich description and can answer any follow-up.
_DEFAULT_INSTRUCTION = (
    "You are a vision assistant for a text-only language model that cannot see images. "
    "Describe this image in detail: main subjects and objects, any visible text "
    "(quote it exactly), colors, spatial layout, and other notable details. "
    "Be concise but complete. Output only the description, no preamble."
)


def _extract_text(resp: dict[str, Any]) -> str:
    """Parse an Ark Responses API response into description text.

    Prefers `message`/`output_text` items; falls back to `reasoning`/`summary_text`
    when the model spent all tokens on reasoning and produced no final message.
    """
    message_texts: list[str] = []
    reasoning_texts: list[str] = []
    for item in resp.get("output", []) or []:
        itype = item.get("type")
        if itype == "message":
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text":
                    message_texts.append(c.get("text", ""))
        elif itype == "reasoning":
            for s in item.get("summary", []) or []:
                if s.get("type") == "summary_text":
                    reasoning_texts.append(s.get("text", ""))
    desc = "".join(message_texts).strip()
    if desc:
        return desc
    return "".join(reasoning_texts).strip()


class Describer:
    def __init__(self, config: DoubaoConfig, cache_size: int = 256) -> None:
        self._config = config
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_size = max(1, cache_size)
        # A shared client is reused across requests for connection pooling.
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._config.timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _cache_key(self, image_url: str) -> str:
        # For data URIs (base64 images) hash the embedded bytes; for http URLs hash
        # the URL string. Same image across turns -> same key -> cache hit.
        if image_url.startswith("data:"):
            try:
                payload = image_url.split(",", 1)[1]
                return hashlib.sha256(payload.encode("ascii")).hexdigest()
            except IndexError:
                return hashlib.sha256(image_url.encode("utf-8")).hexdigest()
        return hashlib.sha256(image_url.encode("utf-8")).hexdigest()

    def _remember(self, key: str, value: str) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    async def describe(self, image_url: str, instruction: str | None = None) -> str | None:
        """Return a text description of the image, or None on failure.

        image_url may be a `data:<media>;base64,...` URI or an http(s) URL.
        Returns None (and logs) on any error so the caller can substitute a
        placeholder without blocking the main request.
        """
        instr = instruction or _DEFAULT_INSTRUCTION
        key = self._cache_key(image_url)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        body = {
            "model": self._config.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instr},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            "max_output_tokens": self._config.max_output_tokens,
            "store": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }

        client = await self._get_client()
        try:
            resp = await client.post(self._config.base_url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            snippet = exc.response.text[:300] if exc.response is not None else ""
            log.warning("doubao HTTP %s: %s", exc.response.status_code if exc.response else "?", snippet)
            return None
        except (httpx.RequestError, ValueError) as exc:
            log.warning("doubao request/parse error: %s", exc)
            return None

        desc = _extract_text(data)
        if not desc:
            log.warning("doubao returned no usable text; raw keys=%s", list(data.keys()))
            return None
        self._remember(key, desc)
        return desc

    def cache_stats(self) -> dict[str, int]:
        return {"size": len(self._cache), "capacity": self._cache_size}
