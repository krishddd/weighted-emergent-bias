"""Tests for the append-only audit trail."""

from __future__ import annotations

from weighted_emergent_bias import (
    AuditKind,
    AuditTrail,
    BiasScore,
    BreakerAction,
    BreakerDecision,
    BreakerState,
    DivergenceMethod,
    PanelDecision,
    PanelResult,
    TaskMode,
)
from weighted_emergent_bias.intervention.types import PrunedAgent


def _score() -> BiasScore:
    return BiasScore(
        effect_size=4.0,
        ci_low=2.0,
        ci_high=6.0,
        p_value=0.01,
        method=DivergenceMethod.JENSEN_SHANNON,
        task_mode=TaskMode.CHOICE,
        n_samples=8,
        raw_divergence=0.5,
        null_mean=0.1,
        null_std=0.1,
        axis="gender",
    )


class TestAppendOnly:
    def test_seq_is_monotonic(self) -> None:
        trail = AuditTrail()
        e0 = trail.record(AuditKind.NOTE, detail={"m": "a"})
        e1 = trail.record(AuditKind.NOTE, detail={"m": "b"})
        assert (e0.seq, e1.seq) == (0, 1)
        assert len(trail) == 2

    def test_events_snapshot_is_immutable(self) -> None:
        trail = AuditTrail()
        trail.record(AuditKind.NOTE)
        snapshot = trail.events
        trail.record(AuditKind.NOTE)
        assert len(snapshot) == 1  # snapshot did not grow with the trail
        assert isinstance(snapshot, tuple)

    def test_no_wallclock_by_default(self) -> None:
        trail = AuditTrail()
        assert trail.record(AuditKind.NOTE).time is None

    def test_injected_clock(self) -> None:
        ticks = iter([10.0, 20.0])
        trail = AuditTrail(clock=lambda: next(ticks))
        assert trail.record(AuditKind.NOTE).time == 10.0
        assert trail.record(AuditKind.NOTE).time == 20.0


class TestConvenienceRecorders:
    def test_record_score(self) -> None:
        trail = AuditTrail()
        ev = trail.record_score(_score(), node="agent")
        assert ev.kind is AuditKind.SCORE
        assert ev.node == "agent"
        assert ev.detail["axis"] == "gender"
        assert ev.detail["significant"] is True

    def test_record_decision(self) -> None:
        trail = AuditTrail()
        decision = BreakerDecision(
            BreakerState.INTERVENTION, BreakerAction.REROUTE, 0.7, 0.5, "spike"
        )
        ev = trail.record_decision(decision, node="router")
        assert ev.kind is AuditKind.BREAKER
        assert ev.detail["action"] == "reroute"

    def test_record_panel(self) -> None:
        trail = AuditTrail()
        result = PanelResult(
            PanelDecision.REVISED,
            {"fixed": True},
            0.9,
            pruned=(PrunedAgent("loud", "overconfident"),),
        )
        ev = trail.record_panel(result)
        assert ev.kind is AuditKind.INTERVENTION
        assert ev.detail["pruned"][0]["agent"] == "loud"

    def test_of_kind_filter(self) -> None:
        trail = AuditTrail()
        trail.record(AuditKind.NOTE)
        trail.record_score(_score(), node="a")
        assert len(trail.of_kind(AuditKind.SCORE)) == 1
        assert len(trail.of_kind(AuditKind.NOTE)) == 1
