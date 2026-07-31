"""Tests for the AgentDAG protocol.

The load-bearing property is deterministic node ordering: the adjacency matrix's row/column
order must be identical run to run, or every centrality number downstream is non-reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from weighted_emergent_bias import AgentDAG


class TestConstruction:
    def test_nodes_in_insertion_order(self) -> None:
        dag = AgentDAG([("a", "b"), ("b", "c"), ("a", "c")])
        assert dag.nodes == ("a", "b", "c")

    def test_explicit_nodes_come_first(self) -> None:
        dag = AgentDAG([("b", "c")], nodes=["a", "b", "c", "d"])
        assert dag.nodes == ("a", "b", "c", "d")  # includes isolated a, d

    def test_from_adjacency(self) -> None:
        dag = AgentDAG.from_adjacency({"a": ["b", "c"], "b": ["c"], "c": []})
        assert dag.nodes == ("a", "b", "c")
        assert set(dag.edges) == {("a", "b"), ("a", "c"), ("b", "c")}

    def test_duplicate_edges_deduped(self) -> None:
        dag = AgentDAG([("a", "b"), ("a", "b")])
        assert dag.edges == (("a", "b"),)

    def test_len_and_contains(self) -> None:
        dag = AgentDAG([("a", "b")])
        assert len(dag) == 2
        assert "a" in dag
        assert "z" not in dag


class TestDeterministicOrdering:
    def test_ordering_is_reproducible(self) -> None:
        # Same construction, built twice, must give identical node order and matrix — this is the
        # property a set-backed graph would violate under string hash randomization.
        edges = [("router", "worker"), ("worker", "judge"), ("router", "judge")]
        d1 = AgentDAG(edges)
        d2 = AgentDAG(edges)
        assert d1.nodes == d2.nodes
        np.testing.assert_array_equal(d1.adjacency_matrix(), d2.adjacency_matrix())

    def test_index_matches_node_position(self) -> None:
        dag = AgentDAG([("a", "b"), ("b", "c")])
        assert [dag.index(n) for n in dag.nodes] == [0, 1, 2]


class TestAdjacencyMatrix:
    def test_matrix_orientation(self) -> None:
        # A[i, j] = 1 for edge nodes[i] -> nodes[j].
        dag = AgentDAG([("a", "b"), ("b", "c")])
        a = dag.adjacency_matrix()
        expected = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float64)
        np.testing.assert_array_equal(a, expected)

    def test_isolated_node_row_is_zero(self) -> None:
        dag = AgentDAG([("a", "b")], nodes=["a", "b", "island"])
        a = dag.adjacency_matrix()
        assert a[dag.index("island")].sum() == 0.0
        assert a[:, dag.index("island")].sum() == 0.0


class TestSuccessorsPredecessors:
    def test_successors_and_predecessors(self) -> None:
        dag = AgentDAG([("a", "c"), ("b", "c")])
        assert set(dag.successors("a")) == {"c"}
        assert set(dag.predecessors("c")) == {"a", "b"}
        assert dag.successors("c") == ()


class TestTopologyAndCycles:
    def test_acyclic_chain(self) -> None:
        dag = AgentDAG([("a", "b"), ("b", "c")])
        assert dag.is_acyclic()
        assert dag.topological_order() == ("a", "b", "c")

    def test_topological_order_respects_dependencies(self) -> None:
        dag = AgentDAG([("a", "c"), ("b", "c"), ("c", "d")])
        order = dag.topological_order()
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")
        assert order.index("c") < order.index("d")

    def test_cycle_detected(self) -> None:
        dag = AgentDAG([("a", "b"), ("b", "c"), ("c", "a")])
        assert not dag.is_acyclic()

    def test_self_loop_is_cyclic(self) -> None:
        dag = AgentDAG([("a", "a")])
        assert not dag.is_acyclic()

    def test_topological_order_raises_on_cycle(self) -> None:
        dag = AgentDAG([("a", "b"), ("b", "a")])
        with pytest.raises(ValueError, match="cycle"):
            dag.topological_order()


class TestNetworkxAdapter:
    def test_from_networkx_roundtrips_edges(self) -> None:
        nx = pytest.importorskip("networkx")
        g = nx.DiGraph()
        g.add_edges_from([("a", "b"), ("b", "c")])
        dag = AgentDAG.from_networkx(g)
        assert set(dag.edges) == {("a", "b"), ("b", "c")}
        assert dag.is_acyclic()
