"""Configuration loading and validation for llm-vision-proxy."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
class Upstream:
    """A real text-only backend the proxy forwards to."""

    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class ProxyConfig:
    listen_host: str
    listen_port: int
    doubao: DoubaoConfig
    cache_size: int
    unknown_model: Literal["default_upstream", "reject"]
    default_upstream: str
    upstreams: dict[str, Upstream] = field(default_factory=dict)


def _default_config_path() -> Path:
    env = os.environ.get("VISION_PROXY_CONFIG")
    if env:
        return Path(env)
    # Default to config.yaml next to the package (project root).
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
    model = str(_require(d, "model", "doubao"))
    max_output_tokens = int(d.get("max_output_tokens", 1024))
    timeout_s = float(d.get("timeout_s", 60))
    return DoubaoConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        max_output_tokens=max_output_tokens,
        timeout_s=timeout_s,
    )


def _build_upstreams(d: dict) -> dict[str, Upstream]:
    out: dict[str, Upstream] = {}
    for name, raw in d.items():
        if not isinstance(raw, dict):
            raise ConfigError(f"upstream '{name}' must be a mapping")
        base_url = str(_require(raw, "base_url", f"upstream.{name}"))
        api_key = str(_require(raw, "api_key", f"upstream.{name}"))
        if api_key.startswith("REPLACE_WITH"):
            raise ConfigError(
                f"upstream.{name}.api_key is still a placeholder; fill config.yaml"
            )
        model = str(raw.get("model", name))
        out[name] = Upstream(base_url=base_url.rstrip("/"), api_key=api_key, model=model)
    return out


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

    doubao = _build_doubao(dict(_require(raw, "doubao", "root")))  # type: ignore[arg-type]
    upstreams = _build_upstreams(dict(_require(raw, "upstreams", "root")))  # type: ignore[arg-type]
    if not upstreams:
        raise ConfigError("at least one upstream must be configured")

    default_upstream = str(raw.get("default_upstream", next(iter(upstreams))))
    if default_upstream not in upstreams:
        raise ConfigError(
            f"default_upstream '{default_upstream}' is not present in upstreams"
        )

    unknown_model = str(raw.get("unknown_model", "default_upstream"))
    if unknown_model not in ("default_upstream", "reject"):
        raise ConfigError(
            f"unknown_model must be 'default_upstream' or 'reject', got '{unknown_model}'"
        )

    return ProxyConfig(
        listen_host=host,
        listen_port=port,
        doubao=doubao,
        cache_size=int(raw.get("cache_size", 256)),
        unknown_model=unknown_model,  # type: ignore[arg-type]
        default_upstream=default_upstream,
        upstreams=upstreams,
    )
