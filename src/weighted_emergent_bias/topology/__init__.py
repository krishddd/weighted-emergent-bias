"""Propagation-layer topology: the agent DAG and centrality weighting (M2)."""

from __future__ import annotations

from .centrality import WeightResult, dependency_weights, katz_weight
from .dag import AgentDAG

__all__ = ["AgentDAG", "WeightResult", "dependency_weights", "katz_weight"]
