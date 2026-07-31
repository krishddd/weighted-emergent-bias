"""The shared graph-state schema threaded through a LangGraph run.

A ``BiasState`` sub-schema carries everything the control plane needs alongside the user's own
state. The channels use reducers so concurrent node updates merge instead of clobbering:
``committed`` and ``unverified_output`` merge by key, and the ``audit`` trail appends.

The **staging buffer** (`unverified_output`) is the mechanism for edge-level interception: a node's
output lands here first, and only the breaker promotes it into ``committed`` — the channel
downstream nodes read. On a halt it is never promoted, so a biased payload cannot reach the next
node's context.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _merge(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Reducer: shallow-merge dict channel updates by key."""
    return {**existing, **update}


class BiasState(TypedDict, total=False):
    """Bias-control channels for a LangGraph state. Compose into your own state ``TypedDict``."""

    committed: Annotated[dict[str, Any], _merge]
    """Promoted, breaker-cleared node outputs — the channel downstream nodes should read."""

    unverified_output: Annotated[dict[str, Any], _merge]
    """Staging buffer: node outputs awaiting a breaker check. Never read downstream directly."""

    local_bias: Annotated[dict[str, float], _merge]
    """Per-node bias magnitude (from ``node_magnitude``)."""

    audit: Annotated[list[str], operator.add]
    """Append-only causal trail (a placeholder for the full M5 SARIF trail)."""

    network_fast: float
    """The fast-scale ``B_net`` at the last breaker check."""

    network_slow: float
    """The slow-scale ``B_net`` at the last breaker check."""

    breaker_state: str
    """The control machine's state (``BreakerState`` value) at the last check."""

    frozen: Any
    """The compromised payload, snapshotted on halt; ``None`` while the run is healthy."""
