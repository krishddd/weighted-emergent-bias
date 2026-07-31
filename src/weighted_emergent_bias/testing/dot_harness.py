"""A synthetic Degeneration-of-Thought simulation harness.

This is the end-to-end M1+M2 integration and the M2 success test: a scripted multi-agent DAG with
one biased seed node whose bias *amplifies* as it propagates downstream (the conformity spiral --
bias compounds, it does not decay), capped at a ceiling. Every node is probed through the real
:func:`~..scoring.noise.compute_local_bias`, turned into an n-stable magnitude, weighted by Katz
blast radius, and integrated by the :class:`~..accumulation.NetworkAccumulator` -- so the whole
detection-and-propagation stack runs, not a mock of it.

On a seeded graph ``B_net`` rises as the bias spreads; on an unbiased control graph of identical
topology it stays flat. That contrast is the WP6 acceptance criterion.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..accumulation import NetworkAccumulator, NetworkState, node_magnitude
from ..scoring.divergence import softmax
from ..scoring.noise import compute_local_bias
from ..topology import AgentDAG, dependency_weights
from ..types import FloatArray, NodeId, TaskMode
from . import markers as mk
from .fake import FakeLLMClient

_BASELINE_PROMPT = "the applicant"


@dataclass(frozen=True)
class DoTResult:
    """A DoT simulation outcome: the per-superstep ``B_net`` trajectory and the setup it ran on."""

    trajectory: list[NetworkState]
    node_bias: dict[NodeId, float]
    weights: dict[NodeId, float]

    @property
    def fast(self) -> list[float]:
        return [s.fast for s in self.trajectory]

    @property
    def slow(self) -> list[float]:
        return [s.slow for s in self.trajectory]


def _propagate_bias(
    dag: AgentDAG, seed: NodeId, beta: float, amplification: float, cap: float
) -> dict[NodeId, float]:
    """Seed bias amplifies along edges (capped). ``beta=0`` yields an all-clean control graph."""
    bias = dict.fromkeys(dag.nodes, 0.0)
    bias[seed] = min(cap, beta)
    for node in dag.topological_order():
        if node == seed:
            continue
        upstream = max((bias[p] for p in dag.predecessors(node)), default=0.0)
        if upstream > 0.0:
            bias[node] = min(cap, amplification * upstream)
    return bias


def _levels(dag: AgentDAG) -> list[list[NodeId]]:
    """Group nodes into supersteps by longest-path depth (sources fire first)."""
    level: dict[NodeId, int] = {}
    for node in dag.topological_order():
        level[node] = 1 + max((level[p] for p in dag.predecessors(node)), default=-1)
    max_level = max(level.values(), default=-1)
    return [[n for n in dag.nodes if level[n] == depth] for depth in range(max_level + 1)]


async def _probe_magnitude(
    bias_level: float,
    *,
    client_rng: np.random.Generator,
    stat_rng: np.random.Generator,
    n_samples: int,
    n_permutations: int,
    logit_noise: float,
    axis: str,
    n_candidates: int,
    alpha: float,
) -> float:
    client = FakeLLMClient(
        client_rng,
        n_candidates=n_candidates,
        bias={axis: bias_level} if bias_level > 0.0 else {},
        logit_noise=logit_noise,
    )
    cf_prompt = f"{mk.mark(axis, 'alt')} {_BASELINE_PROMPT}"

    async def group(prompt: str) -> list[FloatArray]:
        return [
            softmax(await client.score_candidates(prompt, client.candidates))
            for _ in range(n_samples)
        ]

    base = await group(_BASELINE_PROMPT)
    cf = await group(cf_prompt)
    score = compute_local_bias(
        base, cf, task_mode=TaskMode.CHOICE, rng=stat_rng, axis=axis, n_permutations=n_permutations
    )
    return node_magnitude(score, alpha=alpha)


async def simulate_dot(
    dag: AgentDAG,
    seed: NodeId,
    *,
    rng: np.random.Generator,
    beta: float,
    amplification: float = 1.4,
    cap: float = 6.0,
    n_samples: int = 8,
    n_permutations: int = 200,
    logit_noise: float = 0.6,
    axis: str = "gender",
    n_candidates: int = 5,
    alpha: float = 0.05,
) -> DoTResult:
    """Run a DoT simulation on ``dag`` with a biased ``seed`` (``beta=0`` for a clean control).

    Fires nodes superstep-by-superstep in topological depth order, probing each through the real
    estimator and accumulating ``B_net``. Returns the trajectory and the bias/weight setup.
    """
    node_bias = _propagate_bias(dag, seed, beta, amplification, cap)
    weights = dependency_weights(dag).weights
    accumulator = NetworkAccumulator()

    trajectory: list[NetworkState] = []
    for level_nodes in _levels(dag):
        magnitudes: dict[NodeId, float] = {}
        for node in level_nodes:
            client_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            stat_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            magnitudes[node] = await _probe_magnitude(
                node_bias[node],
                client_rng=client_rng,
                stat_rng=stat_rng,
                n_samples=n_samples,
                n_permutations=n_permutations,
                logit_noise=logit_noise,
                axis=axis,
                n_candidates=n_candidates,
                alpha=alpha,
            )
        trajectory.append(accumulator.update(magnitudes, weights))

    return DoTResult(trajectory=trajectory, node_bias=node_bias, weights=weights)


def chain_dag(depth: int) -> tuple[AgentDAG, NodeId]:
    """A linear pipeline ``n0 -> n1 -> ... -> n{depth-1}``; returns the DAG and its source node."""
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    nodes = [f"n{i}" for i in range(depth)]
    edges = list(itertools.pairwise(nodes))
    return AgentDAG(edges, nodes=nodes), nodes[0]
