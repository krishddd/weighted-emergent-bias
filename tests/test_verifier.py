"""Tests for the optional pre-trigger verifier.

The load-bearing property: UNVERIFIED never blocks. An oracle that failed to reach a conclusion is
not evidence of error, and treating its silence as refutation would halt healthy runs.
"""

from __future__ import annotations

import pytest

from weighted_emergent_bias.intervention.verifier import (
    GateOutcome,
    VerificationResult,
    VerificationStatus,
    gate_decision,
)


class _Oracle:
    def __init__(self, result: VerificationResult) -> None:
        self._result = result

    async def verify(self, payload: object, context: str = "") -> VerificationResult:
        return self._result


def _result(status: VerificationStatus, *, detail: str = "") -> VerificationResult:
    return VerificationResult(status=status, confidence=0.9, detail=detail)


class TestVerificationResult:
    def test_confidence_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            VerificationResult(status=VerificationStatus.VERIFIED, confidence=1.5)

    def test_only_refuted_blocks(self) -> None:
        assert _result(VerificationStatus.REFUTED).blocks_promotion
        assert not _result(VerificationStatus.VERIFIED).blocks_promotion
        assert not _result(VerificationStatus.UNVERIFIED).blocks_promotion


class TestGateDecision:
    async def test_no_verifier_is_passthrough(self) -> None:
        clear = await gate_decision({"x": 1}, bias_halts=False)
        halted = await gate_decision({"x": 1}, bias_halts=True)
        assert not clear.halts
        assert halted.halts
        assert clear.verification is None

    async def test_refutation_halts_a_clean_payload(self) -> None:
        # The whole point of the fork: factually wrong but demographically invariant.
        outcome = await gate_decision(
            {"x": 1},
            bias_halts=False,
            verifier=_Oracle(_result(VerificationStatus.REFUTED, detail="contradicted by source")),
        )
        assert outcome.halts
        assert "factually refuted" in outcome.reason

    async def test_unverified_does_not_halt(self) -> None:
        outcome = await gate_decision(
            {"x": 1},
            bias_halts=False,
            verifier=_Oracle(_result(VerificationStatus.UNVERIFIED)),
        )
        assert not outcome.halts
        assert "unverified" in outcome.reason

    async def test_signals_stay_separable(self) -> None:
        outcome = await gate_decision(
            {"x": 1},
            bias_halts=True,
            verifier=_Oracle(_result(VerificationStatus.REFUTED, detail="bad cite")),
        )
        # Both causes are reported, not merged into one score.
        assert "counterfactual bias" in outcome.reason
        assert "factually refuted" in outcome.reason
        assert outcome.verification is not None
        assert outcome.bias_halts


class TestGateOutcome:
    def test_reason_when_clear(self) -> None:
        assert GateOutcome(bias_halts=False, verification=None).reason == "clear"
