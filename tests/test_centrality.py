"""Tests for dependency weighting (Katz blast radius).

The orientation test is the one that catches the classic bug: on a chain A->B->C, the *upstream*
node A must outrank the terminal leaf C -- A's bias reaches everything, C's reaches nothing.
Getting the transpose backwards silently inverts this.
"""

from __future__ import annotations

import numpy as np
import pytest

from weighted_emergent_bias import AgentDAG, dependency_weights, katz_weight


class TestOrientation:
    def test_upstream_outranks_leaf_on_chain(self) -> None:
        dag = AgentDAG([("a", "b"), ("b", "c")])
        w = dependency_weights(dag).weights
        assert w["a"] > w["b"] > w["c"]

    def test_katz_weight_matches_hand_computation(self) -> None:
        # A->B->C, alpha=0.5: (I-0.5A)^-1 row sums = [1.75, 1.5, 1.0].
        dag = AgentDAG([("a", "b"), ("b", "c")])
        assert katz_weight(dag, "a", attenuation=0.5) == pytest.approx(1.75)
        assert katz_weight(dag, "b", attenuation=0.5) == pytest.approx(1.5)
        assert katz_weight(dag, "c", attenuation=0.5) == pytest.approx(1.0)

    def test_central_router_outranks_workers(self) -> None:
        # router fans out to three workers; router should carry the most weight.
        dag = AgentDAG([("router", "w1"), ("router", "w2"), ("router", "w3")])
        w = dependency_weights(dag).weights
        assert w["router"] > w["w1"]
        assert w["w1"] == pytest.approx(w["w2"]) == pytest.approx(w["w3"])


class TestNormalization:
    def test_weights_sum_to_one(self) -> None:
        dag = AgentDAG([("a", "b"), ("b", "c"), ("a", "c")])
        assert sum(dependency_weights(dag).weights.values()) == pytest.approx(1.0)

    def test_single_node(self) -> None:
        dag = AgentDAG([], nodes=["solo"])
        result = dependency_weights(dag)
        assert result.weights == {"solo": pytest.approx(1.0)}

    def test_empty_graph(self) -> None:
        dag = AgentDAG([])
        result = dependency_weights(dag)
        assert result.weights == {}


class TestNilpotency:
    def test_large_alpha_ok_on_dag(self) -> None:
        # DAG adjacency is nilpotent, so even a large alpha converges and does not clamp.
        dag = AgentDAG([("a", "b"), ("b", "c")])
        result = dependency_weights(dag, attenuation=5.0)
        assert result.acyclic
        assert not result.clamp_applied
        assert result.attenuation == 5.0
        assert all(np.isfinite(v) for v in result.weights.values())


class TestCyclicClamp:
    def test_cycle_triggers_clamp(self) -> None:
        dag = AgentDAG([("a", "b"), ("b", "c"), ("c", "a")])
        result = dependency_weights(dag, attenuation=0.9)
        assert not result.acyclic
        assert result.clamp_applied
        assert result.attenuation < 0.9
        assert all(np.isfinite(v) and v > 0 for v in result.weights.values())

    def test_small_alpha_on_cycle_not_clamped(self) -> None:
        # rho of a 3-cycle is 1, so max alpha ~ 0.5; a request below that is honored.
        dag = AgentDAG([("a", "b"), ("b", "c"), ("c", "a")])
        result = dependency_weights(dag, attenuation=0.1)
        assert not result.clamp_applied
        assert result.attenuation == pytest.approx(0.1)

    def test_self_loop_is_handled(self) -> None:
        dag = AgentDAG([("a", "a"), ("a", "b")])
        result = dependency_weights(dag, attenuation=0.9)
        assert result.clamp_applied
        assert all(np.isfinite(v) for v in result.weights.values())


class TestOutDegreeFallback:
    def test_out_degree_ranks_by_fanout(self) -> None:
        dag = AgentDAG([("hub", "a"), ("hub", "b"), ("a", "c")])
        w = dependency_weights(dag, method="out_degree").weights
        assert w["hub"] > w["a"]  # hub has 2 successors, a has 1
        assert w["b"] < w["a"]  # b is a leaf (0), a has 1

    def test_out_degree_never_zero(self) -> None:
        # 1 + out-degree, so even a leaf keeps a positive self-weight.
        dag = AgentDAG([("a", "leaf")])
        w = dependency_weights(dag, method="out_degree").weights
        assert w["leaf"] > 0.0

    def test_out_degree_works_on_cycle(self) -> None:
        dag = AgentDAG([("a", "b"), ("b", "a")])
        result = dependency_weights(dag, method="out_degree")
        assert not result.clamp_applied  # out-degree needs no spectral bound
        assert sum(result.weights.values()) == pytest.approx(1.0)


class TestValidation:
    def test_unknown_method_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            dependency_weights(AgentDAG([("a", "b")]), method="pagerank")

    def test_katz_weight_unknown_node(self) -> None:
        with pytest.raises(KeyError, match="not in the graph"):
            katz_weight(AgentDAG([("a", "b")]), "z")
