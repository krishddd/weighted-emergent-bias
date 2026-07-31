"""LangGraph reference adapter (optional).

Requires ``langgraph`` (``pip install weighted-emergent-bias[langgraph]``). Not imported by the
top-level package, so the core library never depends on LangGraph.
"""

from __future__ import annotations

from .adapter import BreakerOutcome, decide, guarded_node, make_breaker_node
from .skeptic_fanout import SkepticFanoutState, build_skeptic_graph
from .state import BiasState

__all__ = [
    "BiasState",
    "BreakerOutcome",
    "SkepticFanoutState",
    "build_skeptic_graph",
    "decide",
    "guarded_node",
    "make_breaker_node",
]
