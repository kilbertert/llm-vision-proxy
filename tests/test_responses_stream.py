from __future__ import annotations

import pytest

from vision_proxy.responses_stream import _is_reasoning_summary, normalize


async def _aiter(lines):
    for l in lines:
        yield l


async def test_normalize_strips_reasoning_summary_keeps_rest():
    lines = [
        "event: response.created",
        'data: {"type":"response.created"}',
        "",
        "event: response.reasoning_summary_part.added",
        'data: {"type":"response.reasoning_summary_part.added"}',
        "",
        "event: response.reasoning_summary_text.delta",
        'data: {"type":"response.reasoning_summary_text.delta","delta":"thinking"}',
        "",
        "event: response.output_text.delta",
        'data: {"type":"response.output_text.delta","delta":"Hello"}',
        "",
        "event: response.completed",
        'data: {"type":"response.completed"}',
        "",
        "data: [DONE]",
        "",
    ]
    out = []
    async for chunk in normalize(_aiter(lines)):
        out.append(chunk)
    joined = "".join(out)

    # reasoning_summary events dropped entirely
    assert "response.reasoning_summary" not in joined
    assert "thinking" not in joined
    # everything else preserved
    assert "event: response.created" in joined
    assert "event: response.output_text.delta" in joined
    assert "Hello" in joined
    assert "event: response.completed" in joined
    assert "data: [DONE]" in joined


async def test_normalize_preserves_multi_data_events():
    lines = [
        "event: response.output_text.delta",
        'data: {"type":"response.output_text.delta","delta":"hi"}',
        "",
    ]
    out = []
    async for chunk in normalize(_aiter(lines)):
        out.append(chunk)
    assert "".join(out) == 'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hi"}\n\n'


def test_is_reasoning_summary():
    assert _is_reasoning_summary(["event: response.reasoning_summary_part.added", "data: {}"])
    assert _is_reasoning_summary(["event: response.reasoning_summary_text.done", "data: {}"])
    assert not _is_reasoning_summary(["event: response.output_text.delta", "data: {}"])
    assert not _is_reasoning_summary(["event: response.completed", "data: {}"])
    assert not _is_reasoning_summary(["data: [DONE]"])


async def test_normalize_empty():
    out = [c async for c in normalize(_aiter([]))]
    assert out == []
