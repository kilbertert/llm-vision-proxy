#!/usr/bin/env python3
"""Extract Claude-provider upstreams from the cc-switch database into config.yaml.

Reads each `claude` app_type provider in ~/.cc-switch/cc-switch.db, pulls its real
base_url / auth token / model, and merges them into the proxy's `upstreams` table so
the proxy can route by model. Real credentials flow db -> config.yaml without being
printed. Providers whose base_url is localhost (e.g. the cli-proxy-api gateway, which
would double-proxy) are skipped.

Run with the project venv python (needs pyyaml):
    .venv/bin/python scripts/extract-upstreams.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml

PROJECT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT / "config.yaml"
DB = Path.home() / ".cc-switch" / "cc-switch.db"

# env vars that may carry the model name, in priority order.
_MODEL_ENVS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)
_TOKEN_ENVS = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")


def _is_local(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return True
    return host in ("127.0.0.1", "localhost", "::1", "") or host.startswith("169.254.")


def extract_from_db() -> dict[str, dict]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, name, settings_config FROM providers WHERE app_type='claude'"
    ).fetchall()
    con.close()

    out: dict[str, dict] = {}
    for row in rows:
        try:
            cfg = json.loads(row["settings_config"])
        except (TypeError, json.JSONDecodeError):
            continue
        env = cfg.get("env") or {}
        base_url = env.get("ANTHROPIC_BASE_URL")
        if not base_url or _is_local(base_url):
            continue  # skip local gateways / empty
        token = next((env.get(k) for k in _TOKEN_ENVS if env.get(k)), None)
        if not token:
            continue
        model = next((env.get(k) for k in _MODEL_ENVS if env.get(k)), None)
        if not model:
            continue
        if model in out:
            print(f"  ! duplicate model '{model}' (provider {row['name']}); overwriting")
        out[model] = {
            "base_url": base_url.rstrip("/"),
            "api_key": token,
            "model": model,
            "_source": row["name"],
        }
    return out


def merge_into_config(extracted: dict[str, dict]) -> None:
    if not CONFIG.exists():
        sys.exit(f"config.yaml not found at {CONFIG}")
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    upstreams = data.setdefault("upstreams", {})
    before = set(upstreams)
    for model, entry in extracted.items():
        src = entry.pop("_source", None)
        upstreams[model] = {k: v for k, v in entry.items() if not k.startswith("_")}
        print(f"  + {model}  (from cc-switch provider: {src})")
    # Backup then write (mode 600).
    bak = CONFIG.with_suffix(".yaml.bak")
    bak.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    CONFIG.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    CONFIG.chmod(0o600)
    print(f"\nmerged {len(extracted)} upstream(s). before={sorted(before)} after={sorted(upstreams)}")
    print(f"backup: {bak}")


def main() -> None:
    if not DB.exists():
        sys.exit(f"cc-switch db not found: {DB}")
    print(f"reading providers from {DB}")
    extracted = extract_from_db()
    if not extracted:
        print("no extractable claude providers found; nothing to do")
        return
    merge_into_config(extracted)


if __name__ == "__main__":
    main()
