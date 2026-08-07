"""Responses stream normalizer.

Codex CLI (fallback metadata mode for custom models) emits non-fatal
`"... without active item"` errors and can occasionally stall when it receives
Ark's `response.reasoning_summary_*` SSE events. This normalizer strips those
events from the Responses stream (between cliproxyapi and Codex) so Codex only
sees the events it tracks cleanly (output_text deltas, item lifecycle, completion).
The model still reasons server-side; Codex just doesn't receive the reasoning
summary events it can't parse.

Applied to the /v1/responses streaming path only. Non-streaming and other formats
are unaffected (pure passthrough).
"""

from __future__ import annotations

from typing import AsyncIterator


def _is_reasoning_summary(lines: list[str]) -> bool:
    """True if this SSE event is a response.reasoning_summary_* event."""
    for l in lines:
        if l.startswith("event:"):
            if l[6:].strip().startswith("response.reasoning_summary"):
                return True
    for l in lines:
        if l.startswith("data:") and '"type":"response.reasoning_summary' in l:
            return True
    return False


async def normalize(aiter_lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield SSE events from `aiter_lines`, dropping reasoning_summary events.

    `aiter_lines` yields lines without trailing newlines; blank lines separate
    events. Kept events are re-emitted as `line\\n...\\n\\n`.
    """
    buf: list[str] = []
    async for line in aiter_lines:
        if line == "":
            if buf and not _is_reasoning_summary(buf):
                yield "\n".join(buf) + "\n\n"
            buf = []
        else:
            buf.append(line)
    if buf and not _is_reasoning_summary(buf):
        yield "\n".join(buf) + "\n\n"
