"""Dependency weighting -- how much does a node's bias matter to the rest of the run?

A node's bias score means little without its **blast radius**: a biased central router contaminates
everything downstream, while a biased terminal leaf harms nothing. This module computes that blast
radius as a Katz-style weighted count of the nodes reachable *from* each node.

**Orientation (the trap from DESIGN section 2a).** Blast radius counts walks a node can *reach*, so
it is the row-sums of the Katz walk matrix ``(I - alpha*A)^-1`` using ``A`` directly -- i.e. Katz
centrality on the *reversed* (transposed) graph. ``networkx.katz_centrality`` measures the opposite
(influence arriving along incoming edges); using it as-is ranks terminal leaves as maximally
critical, exactly inverted. The A->B->C orientation test guards this.

**Attenuation alpha.** On a DAG the adjacency is nilpotent, so the series converges for *every*
alpha -- a free "how far downstream do I care" knob. On a cyclic graph (LangGraph permits them) that
guarantee is gone and alpha must stay below ``1 / rho(A)`` (rho = spectral radius). We detect cycles
and clamp alpha, recording that the clamp fired -- never silently returning a divergent number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..types import FloatArray, NodeId
from .dag import AgentDAG

_CLAMP_SAFETY = 0.5  # keep alpha at half the 1/rho stability bound on cyclic graphs


@dataclass(frozen=True)
class WeightResult:
    """Normalized dependency weights plus the provenance a silent dict would hide.

    ``weights`` sum to 1 (a share of total blast radius). ``attenuation`` is the alpha actually
    used -- lower than requested when ``clamp_applied`` is True because the graph was cyclic.
    """

    weights: dict[NodeId, float]
    method: str
    attenuation: float
    clamp_applied: bool
    acyclic: bool
    prior_applied: bool = False


def _katz_vector(matrix: FloatArray, alpha: float) -> FloatArray:
    """Row-sums of ``(I - alpha*A)^-1`` -- blast radius (incl. self) per node, in matrix order."""
    n = matrix.shape[0]
    ones: FloatArray = np.ones(n, dtype=np.float64)
    solved = np.linalg.solve(np.eye(n, dtype=np.float64) - alpha * matrix, ones)
    return np.asarray(solved, dtype=np.float64)


def _spectral_radius(matrix: FloatArray) -> float:
    if matrix.size == 0:
        return 0.0
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigenvalues)))


def _resolve_alpha(dag: AgentDAG, matrix: FloatArray, requested: float) -> tuple[float, bool, bool]:
    """Return ``(alpha, clamp_applied, acyclic)``, clamping alpha on cycles for convergence."""
    acyclic = dag.is_acyclic()
    if acyclic:
        return requested, False, True
    rho = _spectral_radius(matrix)
    if rho <= 0.0:
        return requested, False, False
    max_alpha = _CLAMP_SAFETY / rho
    if requested >= max_alpha:
        return max_alpha, True, False
    return requested, False, False


def katz_weight(dag: AgentDAG, node: NodeId, *, attenuation: float = 0.5) -> float:
    """Raw (unnormalized) Katz blast radius of a single node. See :func:`dependency_weights`."""
    if node not in dag:
        raise KeyError(f"node {node!r} is not in the graph")
    matrix = dag.adjacency_matrix()
    alpha, _, _ = _resolve_alpha(dag, matrix, attenuation)
    return float(_katz_vector(matrix, alpha)[dag.index(node)])


def dependency_weights(
    dag: AgentDAG,
    *,
    method: str = "katz",
    attenuation: float = 0.5,
    prior: Mapping[NodeId, float] | None = None,
) -> WeightResult:
    """Normalized dependency weight ``w_i`` for every node, in one pass.

    ``method="katz"`` uses blast-radius centrality; ``method="out_degree"`` is the cheap fallback
    (``1 + out-degree``, so no node -- not even a leaf -- has zero influence over itself). Weights
    are normalized to sum to 1. Recompute only on topology change; the result is deterministic.

    ``prior`` is an optional, externally-supplied per-node multiplier folded in *before*
    normalization -- e.g. a verified error-history score ``P(biased | history_i)`` (higher = more
    scrutiny). Missing nodes default to 1.0 (neutral). It is **injected only**: M2 never updates a
    prior from its own detections, because a self-reinforcing error history is exactly the
    minority-suppression loop the external review warned about (see docs/plans/PHASE-2.md R3).
    """
    n = len(dag)
    if n == 0:
        return WeightResult({}, method, attenuation, clamp_applied=False, acyclic=True)

    matrix = dag.adjacency_matrix()

    if method == "katz":
        alpha, clamp_applied, acyclic = _resolve_alpha(dag, matrix, attenuation)
        raw = _katz_vector(matrix, alpha)
    elif method == "out_degree":
        alpha, clamp_applied, acyclic = attenuation, False, dag.is_acyclic()
        raw = 1.0 + matrix.sum(axis=1)
    else:
        raise ValueError(f"unknown method {method!r}; expected 'katz' or 'out_degree'")

    if prior is not None:
        if any(v < 0.0 for v in prior.values()):
            raise ValueError("prior values must be non-negative")
        factors: FloatArray = np.array(
            [prior.get(node, 1.0) for node in dag.nodes], dtype=np.float64
        )
        raw = raw * factors

    total = float(raw.sum())
    normalized = raw / total if total > 0.0 else np.full(n, 1.0 / n, dtype=np.float64)
    weights = {node: float(normalized[dag.index(node)]) for node in dag.nodes}
    return WeightResult(
        weights, method, alpha, clamp_applied, acyclic, prior_applied=prior is not None
    )
