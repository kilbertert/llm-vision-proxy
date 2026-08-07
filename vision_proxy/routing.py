"""Model -> upstream routing.

The proxy routes each request to a real backend based on the `model` field Claude
Code sends. Claude Code's configured model (set by scripts/vision-use) selects the
backend; the real upstream credentials live only in the proxy config.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ProxyConfig, Upstream


class RoutingError(LookupError):
    """Raised when a model cannot be routed and the policy is `reject`."""


@dataclass(frozen=True)
class Route:
    upstream: Upstream
    fell_back: bool  # True when the model was unknown and we used default_upstream


def resolve_route(config: ProxyConfig, model: str | None) -> Route:
    """Resolve which upstream to use for a given incoming model name.

    - Exact match in config.upstreams -> use it.
    - No match + policy `default_upstream` -> use default_upstream, forward the
      incoming model string as-is (best effort), flag fell_back=True.
    - No match + policy `reject` -> raise RoutingError.
    - model is None -> default_upstream, fell_back=True.
    """
    if model and model in config.upstreams:
        return Route(upstream=config.upstreams[model], fell_back=False)

    if config.unknown_model == "reject":
        raise RoutingError(
            f"model '{model}' is not configured and unknown_model=reject"
        )

    return Route(upstream=config.upstreams[config.default_upstream], fell_back=True)
