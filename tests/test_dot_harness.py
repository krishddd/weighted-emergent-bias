"""The M2 acceptance test: end-to-end DoT propagation.

Runs the full M1+M2 stack (probe -> magnitude -> Katz weight -> multi-scale accumulator) on a
scripted pipeline. The success criterion from PHASE-2.md: B_net rises as a seeded bias propagates,
and stays flat on an unbiased control graph of identical topology.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from weighted_emergent_bias.testing import DoTResult, chain_dag, simulate_dot

SEED = 7


async def _sim(beta: float, *, depth: int = 4, seed: int = SEED) -> DoTResult:
    """Run the harness with a small (fast, budget-robust) permutation count."""
    dag, source = chain_dag(depth)
    return await simulate_dot(
        dag,
        source,
        rng=np.random.default_rng(seed),
        beta=beta,
        n_permutations=150,
        n_samples=8,
    )


class TestDoTPropagation:
    async def test_bias_amplifies_downstream(self) -> None:
        result = await _sim(2.0)
        biases = [result.node_bias[n] for n in ("n0", "n1", "n2", "n3")]
        assert biases == sorted(biases)  # non-decreasing downstream (conformity spiral)
        assert biases[0] == pytest.approx(2.0)

    async def test_upstream_carries_more_weight(self) -> None:
        result = await _sim(2.0)
        assert result.weights["n0"] > result.weights["n3"]  # blast radius

    async def test_bnet_rises_on_seeded_graph(self) -> None:
        fast = (await _sim(2.0)).fast
        # Non-decreasing (small tolerance for probe sampling noise), and a clear net rise.
        for prev, cur in itertools.pairwise(fast):
            assert cur >= prev - 0.02
        assert fast[-1] > fast[0]

    async def test_control_graph_stays_flat(self) -> None:
        control = await _sim(0.0)
        assert max(control.fast) < 0.05
        assert max(control.slow) < 0.05

    async def test_seeded_clearly_separates_from_control(self) -> None:
        seeded = await _sim(2.0)
        control = await _sim(0.0)
        assert seeded.fast[-1] > max(control.fast) + 0.1

    async def test_deterministic_under_seed(self) -> None:
        r1 = await _sim(2.0, depth=3)
        r2 = await _sim(2.0, depth=3)
        assert r1.fast == r2.fast


class TestHarnessHelpers:
    def test_chain_dag_shape(self) -> None:
        dag, seed = chain_dag(3)
        assert seed == "n0"
        assert dag.nodes == ("n0", "n1", "n2")
        assert dag.is_acyclic()

    def test_chain_dag_rejects_zero_depth(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            chain_dag(0)
