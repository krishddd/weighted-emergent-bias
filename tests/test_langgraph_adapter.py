"""Tests for the LangGraph reference adapter.

The `decide` core is tested framework-free; the full graph run is behind `importorskip`, so the
default suite passes without LangGraph. The key property: on a halt the biased payload is frozen and
never promoted to `committed` (the channel a downstream node reads) -- edge-level interception.
"""

from __future__ import annotations

from typing import Any

import pytest

from weighted_emergent_bias import CircuitBreaker, ControlMachine
from weighted_emergent_bias.accumulation import NetworkAccumulator
from weighted_emergent_bias.integrations.langgraph.adapter import decide, guarded_node


def _machine() -> ControlMachine:
    return ControlMachine(CircuitBreaker(tau_enter=0.3, tau_exit=0.15, tau_warn=0.1))


class TestGuardedNode:
    def test_stages_and_scores(self) -> None:
        node = guarded_node("agent", lambda _s: {"text": "out"}, lambda _o: 0.4)
        update = node({})
        assert update["unverified_output"] == {"agent": {"text": "out"}}
        assert update["local_bias"] == {"agent": 0.4}
        assert update["audit"]  # a trail entry was recorded


class TestDecideCore:
    def test_proceed_promotes_staged_output(self) -> None:
        staged = {"agent": {"text": "clean"}}
        outcome = decide({"agent": 0.0}, {"agent": 1.0}, staged, NetworkAccumulator(), _machine())
        assert outcome.promote
        assert outcome.update["committed"] == staged
        assert "frozen" not in outcome.update

    def test_halt_freezes_and_does_not_promote(self) -> None:
        staged = {"agent": {"text": "biased"}}
        outcome = decide({"agent": 0.9}, {"agent": 1.0}, staged, NetworkAccumulator(), _machine())
        assert not outcome.promote
        # Edge-level guarantee: the biased output is frozen, never committed.
        assert outcome.update["frozen"] == staged
        assert "committed" not in outcome.update


class TestFullGraphRun:
    def test_halt_keeps_biased_output_out_of_committed(self) -> None:
        pytest.importorskip("langgraph")
        from langgraph.graph import END, START, StateGraph

        from weighted_emergent_bias.integrations.langgraph import BiasState, make_breaker_node

        weights = {"agent": 1.0}
        machine = _machine()

        def agent(_state: Any) -> Any:
            return {"answer": "strongly biased text"}

        graph: Any = StateGraph(BiasState)  # langgraph's node overloads are out of scope here
        graph.add_node("agent", guarded_node("agent", agent, lambda _o: 0.9))
        graph.add_node(
            "breaker",
            make_breaker_node(weights, machine, proceed_to="downstream", halt_to=END),
        )
        graph.add_node("downstream", lambda _s: {"audit": ["downstream ran"]})
        graph.add_edge(START, "agent")
        graph.add_edge("agent", "breaker")
        graph.add_edge("downstream", END)
        compiled = graph.compile()

        result = compiled.invoke({})

        # The biased output was staged and frozen, but never committed, and downstream never ran.
        assert result.get("committed", {}) == {}
        assert result["frozen"] == {"agent": {"answer": "strongly biased text"}}
        assert "downstream ran" not in result.get("audit", [])
        assert result["breaker_state"] == "intervention"

    def test_clean_run_promotes_and_reaches_downstream(self) -> None:
        pytest.importorskip("langgraph")
        from langgraph.graph import END, START, StateGraph

        from weighted_emergent_bias.integrations.langgraph import BiasState, make_breaker_node

        weights = {"agent": 1.0}
        machine = _machine()

        graph: Any = StateGraph(BiasState)  # langgraph's node overloads are out of scope here
        graph.add_node(
            "agent", guarded_node("agent", lambda _s: {"answer": "fine"}, lambda _o: 0.0)
        )
        graph.add_node(
            "breaker",
            make_breaker_node(weights, machine, proceed_to="downstream", halt_to=END),
        )
        graph.add_node("downstream", lambda _s: {"audit": ["downstream ran"]})
        graph.add_edge(START, "agent")
        graph.add_edge("agent", "breaker")
        graph.add_edge("downstream", END)
        compiled = graph.compile()

        result = compiled.invoke({})

        assert result["committed"] == {"agent": {"answer": "fine"}}
        assert "downstream ran" in result["audit"]
