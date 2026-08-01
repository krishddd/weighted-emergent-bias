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
        assert result.confidence == 0.0  # zero confidence is the "no usable review" marker


class TestTrustIsMonotonicInEvidence:
    """Regression: ``S`` (self-orientation) once took the raw confidence, so a no-evidence
    verdict at ``confidence=0.0`` landed on ``S=0``, hit the ``_S_FLOOR`` clamp and was handed
    ~2300x weight -- the clamp meant to prevent divergence caused it. An unevidenced verdict must
    never outweigh an evidence-backed one, and must lose weight as it grows more assertive."""

    def test_unevidenced_never_outweighs_backed(self) -> None:
        graph = TrustGraph(weighting="trust")
        backed = graph.trust(_v("backed", SkepticStance.REVISE, conf=0.9, evidence=True)).value
        for conf in (0.0, 0.25, 0.5, 0.75, 1.0):
            bare = graph.trust(_v("bare", SkepticStance.STANDS, conf=conf, evidence=False)).value
            assert bare < backed, f"unevidenced verdict at conf={conf} outweighed an evidenced one"

    def test_weight_decreases_as_unevidenced_confidence_rises(self) -> None:
        graph = TrustGraph(weighting="trust")
        weights = [
            graph.trust(_v("bare", SkepticStance.STANDS, conf=c, evidence=False)).value
            for c in (0.0, 0.3, 0.6, 0.9)
        ]
        assert weights == sorted(weights, reverse=True)

    def test_clueless_verdict_cannot_override_evidenced_majority(self) -> None:
        graph = TrustGraph(weighting="trust")
        verdicts = [
            _v(f"backed{i}", SkepticStance.REVISE, conf=0.9, evidence=True, rev="fixed")
            for i in range(3)
        ] + [_v("clueless", SkepticStance.STANDS, conf=0.0, evidence=False)]
        result = graph.aggregate(verdicts)
        assert result.decision is PanelDecision.REVISED
        assert result.corrected_payload == "fixed"


class TestTieBreaking:
    """An exact split resolves toward the more cautious stance, not toward whichever enum member
    happened to be declared first (which silently favoured STANDS -- fail-open)."""

    def test_stands_reject_tie_goes_to_reject(self) -> None:
        graph = TrustGraph(weighting="uniform")
        result = graph.aggregate(
            [
                _v("a", SkepticStance.STANDS, conf=0.5, evidence=True),
                _v("b", SkepticStance.REJECT, conf=0.5, evidence=True),
            ]
        )
        assert result.decision is PanelDecision.REJECTED

    def test_stands_revise_tie_goes_to_revise(self) -> None:
        graph = TrustGraph(weighting="uniform")
        result = graph.aggregate(
            [
                _v("a", SkepticStance.STANDS, conf=0.5, evidence=True),
                _v("b", SkepticStance.REVISE, conf=0.5, evidence=True, rev="fixed"),
            ]
        )
        assert result.decision is PanelDecision.REVISED
