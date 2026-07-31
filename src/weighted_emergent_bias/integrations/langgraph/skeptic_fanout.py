"""LangGraph ``Send``-based parallel skeptic fan-out (optional M4 adapter).

The pure :class:`~...intervention.skeptics.SkepticPanel` already runs skeptics concurrently via
``asyncio.gather``; this module is the LangGraph-native equivalent, dispatching one worker per
skeptic with ``Send`` and joining their verdicts through a reducer channel before the trust graph
aggregates. Requires ``langgraph``; imported lazily so the core never depends on it.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from ...intervention.skeptics import SkepticAgent
from ...intervention.trust import TrustGraph


class SkepticFanoutState(TypedDict, total=False):
    """State for the skeptic fan-out subgraph."""

    disputed: Any
    agent_index: int
    verdicts: Annotated[list[Any], operator.add]
    result: Any


def build_skeptic_graph(agents: Sequence[SkepticAgent], trust: TrustGraph | None = None) -> Any:
    """Compile a subgraph that fans out ``agents`` in parallel via ``Send``, then trust-aggregates.

    The workers are async, so drive it with ``await compiled.ainvoke({"disputed": payload})``; the
    returned state's ``result`` is a ``PanelResult``.
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    resolved_trust = trust if trust is not None else TrustGraph()

    async def _worker(state: Any) -> dict[str, Any]:
        verdict = await agents[state["agent_index"]].review(state["disputed"])
        return {"verdicts": [verdict]}

    def _dispatch(state: Any) -> list[Any]:
        return [
            Send("worker", {"disputed": state["disputed"], "agent_index": i})
            for i in range(len(agents))
        ]

    def _aggregate(state: Any) -> dict[str, Any]:
        return {"result": resolved_trust.aggregate(state["verdicts"])}

    graph: Any = StateGraph(SkepticFanoutState)
    graph.add_node("worker", _worker)
    graph.add_node("aggregate", _aggregate)
    graph.add_conditional_edges(START, _dispatch, ["worker"])
    graph.add_edge("worker", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()
