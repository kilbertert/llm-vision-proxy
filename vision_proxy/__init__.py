"""llm-vision-proxy: transparent vision gateway for text-only LLMs.

Intercepts image content blocks in Anthropic Messages API requests, describes each
image via the Doubao vision model (Volcano Engine Ark Responses API), replaces the
image blocks with text, then forwards the now text-only request to the real backend
and streams the response back unchanged.
"""

from ._version import __version__

__all__ = ["__version__"]
