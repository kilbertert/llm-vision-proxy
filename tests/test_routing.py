from __future__ import annotations

import pytest

from vision_proxy.routing import RoutingError, resolve_route


def test_exact_match_uses_upstream(cfg):
    route = resolve_route(cfg, "deepseek-v4-flash")
    assert route.upstream.base_url == "https://api.deepseek.com/anthropic"
    assert route.upstream.model == "deepseek-v4-flash"
    assert route.fell_back is False


def test_unknown_model_falls_back_to_default(cfg):
    route = resolve_route(cfg, "some-unknown-model")
    assert route.upstream.model == "deepseek-v4-flash"  # default
    assert route.fell_back is True


def test_none_model_falls_back(cfg):
    route = resolve_route(cfg, None)
    assert route.fell_back is True
    assert route.upstream.model == "deepseek-v4-flash"


def test_reject_policy_raises(tmp_path, cfg_path):
    from vision_proxy.config import load_config

    # Reuse the valid yaml but flip the policy to reject.
    text = cfg_path.read_text().replace("unknown_model: default_upstream", "unknown_model: reject")
    cfg_path.write_text(text)
    cfg = load_config(cfg_path)

    # Exact match still works.
    assert resolve_route(cfg, "glm-latest").fell_back is False
    # Unknown model now raises.
    with pytest.raises(RoutingError):
        resolve_route(cfg, "nope")


def test_glm_route(cfg):
    route = resolve_route(cfg, "glm-latest")
    assert route.upstream.base_url == "https://ark.cn-beijing.volces.com/api/plan"
    assert route.upstream.model == "glm-latest"
