"""LangGraph node adapters wiring M1-M3 into a graph.

Two building blocks:

- :func:`guarded_node` wraps a user node so its output is **staged** (not committed) and scored.
- :func:`make_breaker_node` builds the control node: it accumulates ``B_net`` from the staged
  scores (M2), runs the :class:`~...breaker.ControlMachine` (M3), and returns a LangGraph
  ``Command`` that either **promotes** the staged output and proceeds, or **freezes** it and
  reroutes -- so a biased payload never reaches the downstream node.

The M3 decision logic lives in the well-typed pure helper :func:`decide`; the LangGraph-specific
part (returning a ``Command``) is a thin shell. LangGraph is imported lazily so the top-level
package never requires it and the core stays framework-free.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ...accumulation import NetworkAccumulator
from ...breaker import ControlMachine
from ...types import BreakerDecision, NodeId

NodeFn = Callable[[Any], Any]
ScoreFn = Callable[[Any], float]


def guarded_node(
    name: NodeId, node_fn: NodeFn, score_fn: ScoreFn
) -> Callable[[Any], dict[str, Any]]:
    """Wrap a node so its output is staged (not committed) and its bias magnitude recorded.

    ``node_fn(state) -> output`` runs the user's node; ``score_fn(output) -> magnitude`` scores it
    (e.g. via a probe + ``node_magnitude``). The returned update writes only to staging channels.
    """

    def _run(state: Any) -> dict[str, Any]:
        output = node_fn(state)
        magnitude = float(score_fn(output))
        return {
            "unverified_output": {name: output},
            "local_bias": {name: magnitude},
            "audit": [f"{name}: staged, magnitude={magnitude:.3f}"],
        }

    return _run


@dataclass(frozen=True)
class BreakerOutcome:
    """The pure result of a breaker evaluation: the decision plus the state update to apply."""

    decision: BreakerDecision
    update: dict[str, Any]
    promote: bool


def decide(
    local_bias: Mapping[str, float],
    weights: Mapping[str, float],
    staged: Mapping[str, Any],
    accumulator: NetworkAccumulator,
    machine: ControlMachine,
) -> BreakerOutcome:
    """Framework-free core of the breaker node: accumulate ``B_net``, run the machine, build update.

    On proceed, the update promotes ``staged`` into ``committed``; on halt it freezes ``staged`` and
    promotes nothing -- the edge-level guarantee. Testable without LangGraph.
    """
    state = accumulator.update(dict(local_bias), dict(weights))
    decision = machine.step(state.fast, state.slow, payload=dict(staged))

    common: dict[str, Any] = {
        "network_fast": state.fast,
        "network_slow": state.slow,
        "breaker_state": decision.state.value,
        "audit": [f"breaker: {decision.state.value} — {decision.reason}"],
    }
    if decision.halted:
        return BreakerOutcome(
            decision, {**common, "frozen": decision.frozen_payload}, promote=False
        )
    return BreakerOutcome(decision, {**common, "committed": dict(staged)}, promote=True)


def make_breaker_node(
    weights: Mapping[str, float],
    machine: ControlMachine,
    *,
    proceed_to: str,
    halt_to: str,
    accumulator: NetworkAccumulator | None = None,
) -> Callable[[Any], Any]:
    """Build the LangGraph breaker node: evaluates :func:`decide` and returns a routing ``Command``.

    ``proceed_to`` / ``halt_to`` are the node names to route to on proceed vs. halt.
    """
    from langgraph.types import Command  # lazy: optional dependency

    acc = accumulator if accumulator is not None else NetworkAccumulator()

    def _breaker(state: Any) -> Any:
        outcome = decide(
            state.get("local_bias", {}),
            weights,
            state.get("unverified_output", {}),
            acc,
            machine,
        )
        goto = proceed_to if outcome.promote else halt_to
        return Command(goto=goto, update=outcome.update)

    return _breaker
