from __future__ import annotations

import pytest

from vision_proxy.config import ConfigError, load_config


def test_load_valid_config(cfg):
    assert cfg.listen_host == "127.0.0.1"
    assert cfg.listen_port == 8417
    assert cfg.forward.url == "http://cliproxyapi.test"
    assert cfg.forward.api_key == "cpa-test-key"
    assert cfg.doubao.model == "doubao-seed-2-0-lite-260428"


def test_reject_placeholder_doubao_key(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
listen: 127.0.0.1:8417
doubao: {base_url: https://x, api_key: REPLACE_WITH_ARK_API_KEY, model: m}
forward: {url: http://c, api_key: real}
"""
    )
    with pytest.raises(ConfigError, match="placeholder"):
        load_config(p)


def test_reject_placeholder_forward_key(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
listen: 127.0.0.1:8417
doubao: {base_url: https://x, api_key: real, model: m}
forward: {url: http://c, api_key: REPLACE_WITH_CPA_KEY}
"""
    )
    with pytest.raises(ConfigError, match="placeholder"):
        load_config(p)


def test_missing_forward(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
listen: 127.0.0.1:8417
doubao: {base_url: https://x, api_key: real, model: m}
"""
    )
    with pytest.raises(ConfigError, match="forward"):
        load_config(p)


def test_config_file_not_found(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
