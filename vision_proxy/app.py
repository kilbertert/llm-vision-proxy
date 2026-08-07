"""FastAPI application: the thin vision shim.

The proxy no longer routes models. It accepts Anthropic Messages
(/v1/messages, /v1/messages/count_tokens), OpenAI Chat Completions
(/v1/chat/completions), and OpenAI Responses (/v1/responses) requests, strips image
content from them (describing each image via Doubao), and forwards the text-only
request unchanged to a single downstream gateway (cliproxyapi) which owns all model
routing. The response/stream is pure passthrough - no protocol conversion, no model
logic. Model supply is managed in exactly one place: cliproxyapi.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import openai_rewriter, responses_rewriter, rewriter
from ._version import __version__
from .config import ProxyConfig, load_config
from .describer import Describer

log = logging.getLogger("vision_proxy")

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}
_REQ_STRIP = _HOP_BY_HOP | {"host", "content-length", "authorization", "x-api-key", "x-amzn-trace-id"}
_RESP_STRIP = _HOP_BY_HOP | {"content-encoding", "content-length"}


def _forward_req_headers(src: dict[str, str], api_key: str) -> dict[str, str]:
    out = {k: v for k, v in src.items() if k.lower() not in _REQ_STRIP}
    out["authorization"] = f"Bearer {api_key}"
    return out


def _forward_resp_headers(src: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in src.items() if k.lower() not in _RESP_STRIP}


def _error(message: str, status: int = 502, openai: bool = False) -> JSONResponse:
    if openai:
        return JSONResponse(status_code=status, content={"error": {"message": message, "type": "proxy_error"}})
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": "proxy_error", "message": message}},
    )


def create_app(config_path: str | None = None) -> FastAPI:
    config: ProxyConfig = load_config(config_path)
    describer = Describer(config.doubao, cache_size=config.cache_size)
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await describer.aclose()
            await client.aclose()

    app = FastAPI(title="llm-vision-proxy", version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "forward": config.forward.url,
            "cache": describer.cache_stats(),
        }

    async def _send_forward(request: Request, fwd_bytes: bytes, path: str, is_stream: bool) -> Response:
        url = config.forward.url + path
        headers = _forward_req_headers(dict(request.headers), config.forward.api_key)
        if is_stream:
            req = client.build_request("POST", url, headers=headers, content=fwd_bytes)
            resp = await client.send(req, stream=True)

            async def gen():
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                finally:
                    await resp.aclose()

            return StreamingResponse(
                gen(),
                status_code=resp.status_code,
                headers=_forward_resp_headers(dict(resp.headers)),
                media_type=resp.headers.get("content-type"),
            )
        try:
            resp = await client.post(url, headers=headers, content=fwd_bytes)
        except httpx.RequestError as exc:
            log.warning("forward request error: %s", exc)
            return _error(f"downstream request failed: {exc}", status=502)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=_forward_resp_headers(dict(resp.headers)),
            media_type=resp.headers.get("content-type"),
        )

    async def _handle(
        request: Request, path: str, fmt: str, openai_err: bool
    ) -> Response:
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            return _error("request body is not valid JSON", status=400, openai=openai_err)
        if not isinstance(body, dict):
            return _error("request body must be a JSON object", status=400, openai=openai_err)

        if fmt == "anthropic":
            new_body, stats = await rewriter.rewrite_body(body, describer)
        elif fmt == "openai":
            new_body, stats = await openai_rewriter.rewrite_body(body, describer)
        else:
            new_body, stats = await responses_rewriter.rewrite_body(body, describer)

        if stats.had_images:
            log.info(
                "%s rewrite model=%s images found=%d described=%d failed=%d",
                fmt, body.get("model"), stats.images_found,
                stats.images_described, stats.images_failed,
            )
        fwd_bytes = json.dumps(new_body).encode("utf-8") if new_body is not None else body_bytes
        return await _send_forward(request, fwd_bytes, path, bool(body.get("stream")))

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        return await _handle(request, "/v1/messages", "anthropic", openai_err=False)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Response:
        return await _handle(request, "/v1/messages/count_tokens", "anthropic", openai_err=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _handle(request, "/v1/chat/completions", "openai", openai_err=True)

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        return await _handle(request, "/v1/responses", "openai_responses", openai_err=True)

    @app.api_route("/v1/models", methods=["GET", "POST"])
    async def models_proxy(request: Request) -> Response:
        # Proxy /v1/models straight through to cliproxyapi (it owns the catalog).
        url = config.forward.url + "/v1/models"
        headers = _forward_req_headers(dict(request.headers), config.forward.api_key)
        try:
            resp = await client.request(request.method, url, headers=headers,
                                        content=await request.body() if request.method == "POST" else None)
        except httpx.RequestError as exc:
            return _error(f"downstream request failed: {exc}", status=502, openai=True)
        return Response(
            content=resp.content, status_code=resp.status_code,
            headers=_forward_resp_headers(dict(resp.headers)),
            media_type=resp.headers.get("content-type"),
        )

    return app
