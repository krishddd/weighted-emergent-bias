"""Tests for the intervention runner and the end-to-end M3+M4 halt -> repair -> recover loop."""

from __future__ import annotations

import pytest

from weighted_emergent_bias import (
    BreakerState,
    CircuitBreaker,
    ControlMachine,
    InterventionContext,
    InterventionRunner,
    MaderaEditor,
    SkepticPanel,
    SkepticStance,
    TrustGraph,
)
from weighted_emergent_bias.testing import (
    DecayEditor,
    FakeDiagnoser,
    FakeRetriever,
    FakeSkeptic,
    bias_field_scorer,
)


def _madera() -> MaderaEditor:
    return MaderaEditor(
        FakeDiagnoser(), FakeRetriever(), DecayEditor(0.3), bias_field_scorer, threshold=0.1
    )


class TestRouting:
    async def test_concentrated_routes_to_madera(self) -> None:
        runner = InterventionRunner(madera=_madera())
        ctx = InterventionContext(biased_node_count=1, total_node_count=5, max_node_magnitude=0.8)
        result = await runner.run(ctx, {"bias": 0.8})
        assert result.strategy == "madera"
        assert result.converged

    async def test_broad_routes_to_skeptics(self) -> None:
        panel = SkepticPanel(
            [
                FakeSkeptic("auditor", SkepticStance.REVISE, revision={"bias": 0.0}),
                FakeSkeptic("advocate", SkepticStance.REVISE, revision={"bias": 0.0}),
            ]
        )
        runner = InterventionRunner(panel=panel, trust=TrustGraph(weighting="trust"))
        ctx = InterventionContext(biased_node_count=3, total_node_count=4, max_node_magnitude=0.7)
        result = await runner.run(ctx, {"bias": 0.8})
        assert result.strategy == "skeptics"
        assert result.repaired_payload == {"bias": 0.0}

    async def test_missing_strategy_impl_raises(self) -> None:
        runner = InterventionRunner()  # no panel, no madera
        ctx = InterventionContext(biased_node_count=1, total_node_count=5, max_node_magnitude=0.8)
        with pytest.raises(ValueError, match="madera"):
            await runner.run(ctx, {"bias": 0.8})


class TestEndToEndRecovery:
    async def test_halt_repair_recover(self) -> None:
        # The full M3+M4 loop: a biased payload halts the breaker, MADERA repairs it, and the
        # re-measured B_net drops below tau_exit so the control machine returns to Normal.
        runner = InterventionRunner(madera=_madera())
        machine = ControlMachine(CircuitBreaker(tau_enter=0.3, tau_exit=0.15), cooldown_steps=0)
        payload = {"bias": 0.8}

        score = await bias_field_scorer(payload)
        halted = machine.step(score, 0.0, payload=payload)
        assert halted.state is BreakerState.INTERVENTION

        ctx = InterventionContext(biased_node_count=1, total_node_count=5, max_node_magnitude=score)
        result = await runner.run(ctx, payload)
        assert result.converged
        assert result.repaired_payload is not None

        repaired_score = await bias_field_scorer(result.repaired_payload)
        assert repaired_score < 0.15  # below tau_exit
        recovered = machine.step(repaired_score, 0.0)
        assert recovered.state is BreakerState.NORMAL
