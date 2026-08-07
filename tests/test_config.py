from __future__ import annotations

import pytest

from vision_proxy.config import ConfigError, load_config


def test_load_valid_config(cfg):
    assert cfg.listen_host == "127.0.0.1"
    assert cfg.listen_port == 8417
    assert cfg.default_upstream == "deepseek-v4-flash"
    assert set(cfg.upstreams) == {"deepseek-v4-flash", "glm-latest"}
    assert cfg.doubao.model == "doubao-seed-2-0-lite-260428"
    assert cfg.unknown_model == "default_upstream"


def test_reject_placeholder_doubao_key(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
listen: 127.0.0.1:8417
doubao:
  base_url: https://x
  api_key: REPLACE_WITH_ARK_API_KEY
  model: m
upstreams:
  a: {base_url: https://y, api_key: real-key}
"""
    )
    with pytest.raises(ConfigError, match="placeholder"):
        load_config(p)


def test_reject_placeholder_upstream_key(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
listen: 127.0.0.1:8417
doubao: {base_url: https://x, api_key: real, model: m}
upstreams:
  a: {base_url: https://y, api_key: REPLACE_WITH_DEEPSEEK_KEY}
"""
    )
    with pytest.raises(ConfigError, match="placeholder"):
        load_config(p)


def test_missing_upstream_field(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
listen: 127.0.0.1:8417
doubao: {base_url: https://x, api_key: real, model: m}
upstreams:
  a: {api_key: real}
"""
    )
    with pytest.raises(ConfigError, match="base_url"):
        load_config(p)


def test_default_upstream_must_exist(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
listen: 127.0.0.1:8417
doubao: {base_url: https://x, api_key: real, model: m}
default_upstream: nope
upstreams:
  a: {base_url: https://y, api_key: real}
"""
    )
    with pytest.raises(ConfigError, match="default_upstream"):
        load_config(p)


def test_unknown_model_policy_validation(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
listen: 127.0.0.1:8417
doubao: {base_url: https://x, api_key: real, model: m}
unknown_model: bogus
upstreams:
  a: {base_url: https://y, api_key: real}
"""
    )
    with pytest.raises(ConfigError, match="unknown_model"):
        load_config(p)


def test_config_file_not_found(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
