"""A framework-agnostic directed graph of agent nodes.

M2's math (Katz centrality, accumulation) needs a stable adjacency matrix, and a matrix needs a
*canonical, reproducible* node ordering. Python randomizes ``set`` iteration of strings across
processes, so a graph backed by a set would silently produce different centrality numbers run to
run. ``AgentDAG`` fixes the ordering to first-seen (insertion) order and exposes it explicitly.

The type is deliberately thin and depends only on numpy: it can be built from a plain adjacency
dict or adapted from a ``networkx.DiGraph``, but nothing downstream is coupled to networkx (or to
LangGraph). Cycle detection and topological order are implemented here (Kahn's algorithm) so the
module has no graph-library dependency and so WP2 has the reverse-topological order Katz needs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..types import FloatArray, NodeId


class AgentDAG:
    """A directed graph with deterministic, insertion-ordered nodes.

    Edges point in the direction of information flow (upstream ``->`` downstream). The adjacency
    matrix ``A`` has ``A[i, j] = 1`` iff there is an edge from ``nodes[i]`` to ``nodes[j]``; M2's
    blast-radius centrality transposes it (a node's influence flows along outgoing edges).
    """

    def __init__(
        self,
        edges: Iterable[tuple[NodeId, NodeId]],
        *,
        nodes: Iterable[NodeId] | None = None,
    ) -> None:
        # First-seen ordering: explicit `nodes` first (so isolated nodes keep their place),
        # then any endpoints discovered in edges. A dict preserves insertion order and dedupes.
        order: dict[NodeId, None] = {}
        if nodes is not None:
            for n in nodes:
                order.setdefault(n, None)

        seen_edges: dict[tuple[NodeId, NodeId], None] = {}
        for src, dst in edges:
            order.setdefault(src, None)
            order.setdefault(dst, None)
            seen_edges.setdefault((src, dst), None)

        self._nodes: tuple[NodeId, ...] = tuple(order)
        self._edges: tuple[tuple[NodeId, NodeId], ...] = tuple(seen_edges)
        self._index: dict[NodeId, int] = {n: i for i, n in enumerate(self._nodes)}

        self._succ: dict[NodeId, list[NodeId]] = {n: [] for n in self._nodes}
        self._pred: dict[NodeId, list[NodeId]] = {n: [] for n in self._nodes}
        for src, dst in self._edges:
            self._succ[src].append(dst)
            self._pred[dst].append(src)

    # --- construction adapters ---------------------------------------------------------------

    @classmethod
    def from_adjacency(cls, adjacency: Mapping[NodeId, Sequence[NodeId]]) -> AgentDAG:
        """Build from ``{node: [successors]}``. Keys fix the initial node order."""
        edges = [(src, dst) for src, succs in adjacency.items() for dst in succs]
        return cls(edges, nodes=list(adjacency.keys()))

    @classmethod
    def from_networkx(cls, graph: Any) -> AgentDAG:
        """Adapt a ``networkx.DiGraph``. Node order follows the graph's node iteration order."""
        return cls(list(graph.edges()), nodes=list(graph.nodes()))

    # --- accessors ---------------------------------------------------------------------------

    @property
    def nodes(self) -> tuple[NodeId, ...]:
        """Nodes in canonical (insertion) order — the row/column order of the adjacency matrix."""
        return self._nodes

    @property
    def edges(self) -> tuple[tuple[NodeId, NodeId], ...]:
        return self._edges

    def index(self, node: NodeId) -> int:
        """Position of ``node`` in the canonical ordering."""
        return self._index[node]

    def successors(self, node: NodeId) -> tuple[NodeId, ...]:
        return tuple(self._succ[node])

    def predecessors(self, node: NodeId) -> tuple[NodeId, ...]:
        return tuple(self._pred[node])

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node: object) -> bool:
        return node in self._index

    # --- matrices and structure --------------------------------------------------------------

    def adjacency_matrix(self) -> FloatArray:
        """The ``(n, n)`` adjacency matrix in canonical node order (``A[i, j] = 1`` for i->j)."""
        n = len(self._nodes)
        matrix: FloatArray = np.zeros((n, n), dtype=np.float64)
        for src, dst in self._edges:
            matrix[self._index[src], self._index[dst]] = 1.0
        return matrix

    def topological_order(self) -> tuple[NodeId, ...]:
        """Nodes in a topological order (Kahn's algorithm). Raises ``ValueError`` if cyclic.

        Ties are broken by canonical node order, so the result is deterministic.
        """
        indegree = {n: len(self._pred[n]) for n in self._nodes}
        ready = [n for n in self._nodes if indegree[n] == 0]  # canonical order preserved
        ordered: list[NodeId] = []
        while ready:
            node = ready.pop(0)
            ordered.append(node)
            for succ in self._succ[node]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
        if len(ordered) != len(self._nodes):
            raise ValueError("graph has a cycle; no topological order exists")
        return tuple(ordered)

    def is_acyclic(self) -> bool:
        """True if the graph is a DAG (has a topological order)."""
        try:
            self.topological_order()
        except ValueError:
            return False
        return True
