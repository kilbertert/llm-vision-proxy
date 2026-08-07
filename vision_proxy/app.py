"""FastAPI application: the transparent vision proxy.

Exposes two wire formats:
  - Anthropic Messages API (/v1/messages, /v1/messages/count_tokens) for Claude Code.
  - OpenAI Chat Completions API (/v1/chat/completions) for Codex CLI.

For each request it routes by `model` to a real text-only backend (whose `format`
must match the incoming wire format), rewrites image content into text descriptions
via Doubao, forwards the now text-only request, and streams the response back
unchanged. Both formats are pure passthrough on the response side (no cross-format
conversion), so tool calls, reasoning, and SSE events pass through untouched.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import openai_rewriter
from ._version import __version__
from .config import ProxyConfig, load_config
from .describer import Describer
from .rewriter import rewrite_body
from .routing import RoutingError, resolve_route

log = logging.getLogger("vision_proxy")

# Headers we must not blindly copy from the client request or from the upstream
# response (hop-by-hop / length / auth - we manage these ourselves).
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
_REQ_STRIP = _HOP_BY_HOP | {
    "host",
    "content-length",
    "authorization",
    "x-api-key",
    "x-amzn-trace-id",
}
_RESP_STRIP = _HOP_BY_HOP | {"content-encoding", "content-length"}


def _forward_req_headers(src: dict[str, str], upstream_api_key: str) -> dict[str, str]:
    out = {k: v for k, v in src.items() if k.lower() not in _REQ_STRIP}
    # Set auth in both common forms so any backend accepts it.
    out["x-api-key"] = upstream_api_key
    out["authorization"] = f"Bearer {upstream_api_key}"
    return out


def _forward_resp_headers(src: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in src.items() if k.lower() not in _RESP_STRIP}


def _error_json(message: str, status: int = 502) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": "proxy_error", "message": message}},
    )


def _error_openai(message: str, status: int = 502) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "proxy_error"}},
    )


def create_app(config_path: str | None = None) -> FastAPI:
    config: ProxyConfig = load_config(config_path)
    describer = Describer(config.doubao, cache_size=config.cache_size)
    upstream_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await describer.aclose()
            await upstream_client.aclose()

    app = FastAPI(title="llm-vision-proxy", version=__version__, lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "upstreams": {
                name: {"format": u.format, "model": u.model}
                for name, u in config.upstreams.items()
            },
            "default_upstream": config.default_upstream,
            "cache": describer.cache_stats(),
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "llm-vision-proxy"}
                for name in config.upstreams
            ],
            "object": "list",
        }

    def _resolve(model: str | None, want_format: str, error_fn):
        try:
            route = resolve_route(config, model)
        except RoutingError as exc:
            return None, error_fn(str(exc), status=404)
        if route.upstream.format != want_format:
            return None, error_fn(
                f"model '{model}' is not an {want_format}-format upstream "
                f"(it is '{route.upstream.format}'); use the matching endpoint",
                status=400,
            )
        if route.fell_back:
            log.warning(
                "model '%s' not in upstreams; falling back to default '%s'",
                model,
                config.default_upstream,
            )
        return route, None

    def _prepare_bytes(body_bytes: bytes, body: dict, new_body: dict | None, route) -> bytes:
        incoming_model = body.get("model")
        need_override = (not route.fell_back) and route.upstream.model != incoming_model
        if new_body is not None:
            if need_override:
                new_body["model"] = route.upstream.model
            return json.dumps(new_body).encode("utf-8")
        if need_override:
            light = dict(body)
            light["model"] = route.upstream.model
            return json.dumps(light).encode("utf-8")
        return body_bytes

    async def _send_upstream(
        request: Request, fwd_bytes: bytes, route, path: str, is_stream: bool
    ) -> Response:
        url = route.upstream.base_url + path
        headers = _forward_req_headers(dict(request.headers), route.upstream.api_key)

        if is_stream:
            req = upstream_client.build_request("POST", url, headers=headers, content=fwd_bytes)
            upstream_resp = await upstream_client.send(req, stream=True)

            async def gen():
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        yield chunk
                finally:
                    await upstream_resp.aclose()

            return StreamingResponse(
                gen(),
                status_code=upstream_resp.status_code,
                headers=_forward_resp_headers(dict(upstream_resp.headers)),
                media_type=upstream_resp.headers.get("content-type"),
            )

        try:
            upstream_resp = await upstream_client.post(url, headers=headers, content=fwd_bytes)
        except httpx.RequestError as exc:
            log.warning("upstream request error: %s", exc)
            return _error_json(f"upstream request failed: {exc}", status=502)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=_forward_resp_headers(dict(upstream_resp.headers)),
            media_type=upstream_resp.headers.get("content-type"),
        )

    async def _handle_anthropic(
        request: Request, body_bytes: bytes, body: dict, path: str
    ) -> Response:
        model = body.get("model")
        route, err = _resolve(model, "anthropic", _error_json)
        if err is not None:
            return err
        new_body, stats = await rewrite_body(body, describer)
        if stats.had_images:
            log.info(
                "anthropic rewrite model=%s upstream=%s images found=%d described=%d failed=%d",
                model, route.upstream.model, stats.images_found,
                stats.images_described, stats.images_failed,
            )
        fwd_bytes = _prepare_bytes(body_bytes, body, new_body, route)
        return await _send_upstream(request, fwd_bytes, route, path, bool(body.get("stream")))

    async def _handle_openai(request: Request, body_bytes: bytes, body: dict) -> Response:
        model = body.get("model")
        route, err = _resolve(model, "openai", _error_openai)
        if err is not None:
            return err
        new_body, stats = await openai_rewriter.rewrite_body(body, describer)
        if stats.had_images:
            log.info(
                "openai rewrite model=%s upstream=%s images found=%d described=%d failed=%d",
                model, route.upstream.model, stats.images_found,
                stats.images_described, stats.images_failed,
            )
        fwd_bytes = _prepare_bytes(body_bytes, body, new_body, route)
        return await _send_upstream(
            request, fwd_bytes, route, "/chat/completions", bool(body.get("stream"))
        )

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            return _error_json("request body is not valid JSON", status=400)
        if not isinstance(body, dict):
            return _error_json("request body must be a JSON object", status=400)
        return await _handle_anthropic(request, body_bytes, body, "/v1/messages")

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Response:
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            return _error_json("request body is not valid JSON", status=400)
        if not isinstance(body, dict):
            return _error_json("request body must be a JSON object", status=400)
        return await _handle_anthropic(
            request, body_bytes, body, "/v1/messages/count_tokens"
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            return _error_openai("request body is not valid JSON", status=400)
        if not isinstance(body, dict):
            return _error_openai("request body must be a JSON object", status=400)
        return await _handle_openai(request, body_bytes, body)

    return app
