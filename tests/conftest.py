"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import pytest

from vision_proxy.config import load_config

VALID_YAML = """
listen: 127.0.0.1:8417
doubao:
  base_url: https://ark.cn-beijing.volces.com/api/v3/responses
  api_key: ark-test-key
  model: doubao-seed-2-0-lite-260428
  max_output_tokens: 1024
  timeout_s: 60
cache_size: 16
unknown_model: default_upstream
default_upstream: deepseek-v4-flash
upstreams:
  deepseek-v4-flash:
    base_url: https://api.deepseek.com/anthropic
    api_key: ds-test-key
    model: deepseek-v4-flash
  glm-latest:
    base_url: https://ark.cn-beijing.volces.com/api/plan
    api_key: ark-glm-key
    model: glm-latest
"""


class FakeDescriber:
    """Duck-typed stand-in for Describer used by rewriter tests."""

    def __init__(self, result: str = "a red circle with text Hi", fail_urls: set[str] | None = None) -> None:
        self.result = result
        self.fail_urls = fail_urls or set()
        self.calls: list[str] = []

    async def describe(self, image_url: str, instruction: str | None = None) -> str | None:
        self.calls.append(image_url)
        if image_url in self.fail_urls:
            return None
        return self.result

    def cache_stats(self) -> dict[str, int]:
        return {"size": 0, "capacity": 0}

    async def aclose(self) -> None:
        pass


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(VALID_YAML)
    return p


@pytest.fixture
def cfg(cfg_path):
    return load_config(cfg_path)


@pytest.fixture
def fake_describer():
    return FakeDescriber()
