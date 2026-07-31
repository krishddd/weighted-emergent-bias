"""Tests for the M3 control value types."""

from __future__ import annotations

import pytest

from weighted_emergent_bias import (
    BreakerAction,
    BreakerDecision,
    BreakerState,
)


def _decision(**overrides: object) -> BreakerDecision:
    kwargs: dict[str, object] = {
        "state": BreakerState.NORMAL,
        "action": BreakerAction.PROCEED,
        "mixing_ratio": 0.0,
        "b_net": 0.1,
        "reason": "below thresholds",
    }
    kwargs.update(overrides)
    return BreakerDecision(**kwargs)  # type: ignore[arg-type]


class TestBreakerDecision:
    def test_valid(self) -> None:
        d = _decision()
        assert d.state is BreakerState.NORMAL
        assert d.frozen_payload is None

    def test_frozen(self) -> None:
        d = _decision()
        with pytest.raises((AttributeError, TypeError)):
            d.action = BreakerAction.HALT  # type: ignore[misc]

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
    def test_mixing_ratio_range(self, bad: float) -> None:
        with pytest.raises(ValueError, match="mixing_ratio"):
            _decision(mixing_ratio=bad)

    def test_halted_property(self) -> None:
        assert not _decision(action=BreakerAction.PROCEED).halted
        assert _decision(action=BreakerAction.HALT).halted
        assert _decision(action=BreakerAction.REROUTE).halted
        assert _decision(action=BreakerAction.ESCALATE).halted

    def test_carries_frozen_payload(self) -> None:
        payload = {"answer": "biased text"}
        d = _decision(action=BreakerAction.HALT, frozen_payload=payload)
        assert d.frozen_payload == payload


class TestEnums:
    def test_states_distinct(self) -> None:
        assert len(set(BreakerState)) == 5

    def test_actions_distinct(self) -> None:
        assert len(set(BreakerAction)) == 4

    def test_str_valued(self) -> None:
        assert BreakerState.INTERVENTION.value == "intervention"
        assert BreakerAction.REROUTE.value == "reroute"
