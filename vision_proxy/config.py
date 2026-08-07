"""Configuration loading and validation for llm-vision-proxy (thin vision shim).

The proxy no longer manages model supplies - it is a thin vision shim that strips
image content from requests, describes each image via Doubao, and forwards the
text-only request to a single downstream gateway (cliproxyapi) which owns all model
routing. Model supply is managed in exactly one place: cliproxyapi.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised when the configuration file is missing required or invalid values."""


@dataclass(frozen=True)
class DoubaoConfig:
    base_url: str
    api_key: str
    model: str
    max_output_tokens: int
    timeout_s: float


@dataclass(frozen=True)
class Forward:
    """The downstream gateway (cliproxyapi) that owns model routing."""

    url: str
    api_key: str


@dataclass(frozen=True)
class ProxyConfig:
    listen_host: str
    listen_port: int
    doubao: DoubaoConfig
    cache_size: int
    forward: Forward


def _default_config_path() -> Path:
    env = os.environ.get("VISION_PROXY_CONFIG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "config.yaml"


def _require(d: dict, key: str, ctx: str) -> object:
    if key not in d:
        raise ConfigError(f"missing required field '{key}' in {ctx}")
    return d[key]


def _build_doubao(d: dict) -> DoubaoConfig:
    base_url = str(_require(d, "base_url", "doubao"))
    api_key = str(_require(d, "api_key", "doubao"))
    if api_key.startswith("REPLACE_WITH"):
        raise ConfigError("doubao.api_key is still a placeholder; fill config.yaml")
    return DoubaoConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=str(_require(d, "model", "doubao")),
        max_output_tokens=int(d.get("max_output_tokens", 1024)),
        timeout_s=float(d.get("timeout_s", 60)),
    )


def _build_forward(d: dict) -> Forward:
    url = str(_require(d, "url", "forward")).rstrip("/")
    api_key = str(_require(d, "api_key", "forward"))
    if api_key.startswith("REPLACE_WITH"):
        raise ConfigError("forward.api_key is still a placeholder; fill config.yaml")
    return Forward(url=url, api_key=api_key)


def load_config(path: Path | str | None = None) -> ProxyConfig:
    cfg_path = Path(path) if path else _default_config_path()
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    listen = str(raw.get("listen", "127.0.0.1:8417"))
    if ":" not in listen:
        raise ConfigError("listen must be 'host:port'")
    host, _, port_s = listen.rpartition(":")
    try:
        port = int(port_s)
    except ValueError as exc:
        raise ConfigError(f"listen port not an integer: {port_s}") from exc

    return ProxyConfig(
        listen_host=host,
        listen_port=port,
        doubao=_build_doubao(dict(_require(raw, "doubao", "root"))),  # type: ignore[arg-type]
        cache_size=int(raw.get("cache_size", 256)),
        forward=_build_forward(dict(_require(raw, "forward", "root"))),  # type: ignore[arg-type]
    )
