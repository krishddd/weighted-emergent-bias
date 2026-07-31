"""Tests for the M4 intervention value types."""

from __future__ import annotations

import pytest

from weighted_emergent_bias import (
    InterventionResult,
    PanelDecision,
    PanelResult,
    SkepticStance,
    SkepticVerdict,
    TrustScore,
)
from weighted_emergent_bias.intervention.types import PrunedAgent


class TestSkepticVerdict:
    def test_valid(self) -> None:
        v = SkepticVerdict("auditor", SkepticStance.REVISE, "no evidence", 0.8, has_evidence=True)
        assert v.stance is SkepticStance.REVISE
        assert v.confidence == 0.8

    def test_empty_agent_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="agent_type"):
            SkepticVerdict("", SkepticStance.STANDS, "", 0.5)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_confidence_range(self, bad: float) -> None:
        with pytest.raises(ValueError, match="confidence"):
            SkepticVerdict("a", SkepticStance.STANDS, "", bad)


class TestTrustScore:
    def test_value_formula(self) -> None:
        # (1 + 2 + 3) / 2 = 3.0
        assert TrustScore(1.0, 2.0, 3.0, 2.0).value == pytest.approx(3.0)

    def test_denominator_clamp_prevents_blowup(self) -> None:
        # S = 0 must not divide by zero; clamp keeps it finite.
        v = TrustScore(1.0, 1.0, 1.0, 0.0).value
        assert v == pytest.approx(3.0 / 1e-3)

    def test_negative_component_rejected(self) -> None:
        with pytest.raises(ValueError, match="credibility"):
            TrustScore(-1.0, 1.0, 1.0, 1.0)


class TestPanelResult:
    def test_valid(self) -> None:
        r = PanelResult(PanelDecision.REVISED, {"answer": "fixed"}, 0.9)
        assert r.decision is PanelDecision.REVISED
        assert r.corrected_payload == {"answer": "fixed"}

    def test_confidence_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            PanelResult(PanelDecision.STANDS, None, 1.5)

    def test_carries_pruned(self) -> None:
        r = PanelResult(
            PanelDecision.STANDS, None, 0.5, pruned=(PrunedAgent("x", "overconfident"),)
        )
        assert r.pruned[0].reason == "overconfident"


class TestInterventionResult:
    def test_valid(self) -> None:
        r = InterventionResult("skeptics", {"answer": "ok"}, converged=True, trail=("ran panel",))
        assert r.strategy == "skeptics"
        assert r.converged
