"""Tests for trust-graph aggregation.

The minority-suppression test is the load-bearing safety property: a correct, evidence-backed
dissenter against a biased majority is never pruned, and can even win once the overconfident
majority is pruned. Pruning fires on overconfidence, never on dissent.
"""

from __future__ import annotations

import pytest

from weighted_emergent_bias import (
    PanelDecision,
    Payload,
    SkepticStance,
    SkepticVerdict,
    TrustGraph,
)


def _v(
    agent: str, stance: SkepticStance, *, conf: float, evidence: bool, rev: Payload | None = None
) -> SkepticVerdict:
    return SkepticVerdict(agent, stance, "", conf, has_evidence=evidence, proposed_payload=rev)


class TestUniformBaseline:
    def test_majority_wins_no_pruning(self) -> None:
        graph = TrustGraph(weighting="uniform")
        verdicts = [
            _v("a", SkepticStance.STANDS, conf=0.9, evidence=False),
            _v("b", SkepticStance.STANDS, conf=0.9, evidence=False),
            _v("c", SkepticStance.REVISE, conf=0.9, evidence=True, rev="fixed"),
        ]
        result = graph.aggregate(verdicts)
        assert result.decision is PanelDecision.STANDS  # 2 vs 1 majority
        assert result.pruned == ()  # uniform never prunes


class TestOverconfidencePruning:
    def test_overconfident_agent_pruned_in_trust_mode(self) -> None:
        graph = TrustGraph(weighting="trust")
        verdicts = [
            _v("loud", SkepticStance.STANDS, conf=0.99, evidence=False),  # overconfident
            _v("careful", SkepticStance.REVISE, conf=0.6, evidence=True, rev="fixed"),
        ]
        result = graph.aggregate(verdicts)
        assert [p.agent_type for p in result.pruned] == ["loud"]

    def test_pruning_is_about_overconfidence_not_dissent(self) -> None:
        # An overconfident agent that AGREES with the majority is still pruned -> not about dissent.
        graph = TrustGraph(weighting="trust")
        verdicts = [
            _v("loud1", SkepticStance.STANDS, conf=0.99, evidence=False),
            _v("loud2", SkepticStance.STANDS, conf=0.99, evidence=False),
            _v("careful", SkepticStance.STANDS, conf=0.5, evidence=True),
        ]
        result = graph.aggregate(verdicts)
        assert {p.agent_type for p in result.pruned} == {"loud1", "loud2"}


class TestMinoritySuppressionGuarantee:
    def test_evidence_backed_dissenter_never_pruned(self) -> None:
        graph = TrustGraph(weighting="trust")
        verdicts = [
            _v("majority1", SkepticStance.STANDS, conf=0.99, evidence=False),
            _v("majority2", SkepticStance.STANDS, conf=0.99, evidence=False),
            _v("dissenter", SkepticStance.REVISE, conf=0.95, evidence=True, rev="unbiased"),
        ]
        result = graph.aggregate(verdicts)
        pruned_types = {p.agent_type for p in result.pruned}
        assert "dissenter" not in pruned_types  # the guarantee
        # And with the overconfident majority pruned, the correct dissenter wins.
        assert result.decision is PanelDecision.REVISED
        assert result.corrected_payload == "unbiased"


class TestTrustWeighting:
    def test_evidence_backed_verdict_outweighs_bare_one(self) -> None:
        # Trust weighting (not pruning) should favor the evidence-backed verdict even below the
        # overconfidence cutoff.
        graph = TrustGraph(weighting="trust")
        verdicts = [
            _v("bare", SkepticStance.STANDS, conf=0.5, evidence=False),
            _v("backed", SkepticStance.REVISE, conf=0.5, evidence=True, rev="fixed"),
        ]
        result = graph.aggregate(verdicts)
        assert result.decision is PanelDecision.REVISED


class TestValidation:
    def test_bad_weighting_rejected(self) -> None:
        with pytest.raises(ValueError, match="weighting"):
            TrustGraph(weighting="bayesian")

    def test_all_pruned_yields_stands(self) -> None:
        graph = TrustGraph(weighting="trust")
        verdicts = [
            _v("a", SkepticStance.REVISE, conf=0.99, evidence=False),
            _v("b", SkepticStance.REJECT, conf=0.99, evidence=False),
        ]
        result = graph.aggregate(verdicts)
        assert result.decision is PanelDecision.STANDS
        assert len(result.pruned) == 2
