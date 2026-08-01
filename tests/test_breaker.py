"""Tests for the M3 control plane: hysteresis breaker and recovery state machine.

The anti-thrash test is the reason hysteresis exists: a B_net oscillating near the threshold must
not flip the breaker on and off. The lifecycle tests pin the Normal->Warning->Intervention->Recovery
machine, including cool-down gating and escalation after repeated failure.
"""

from __future__ import annotations

import pytest

from weighted_emergent_bias import (
    BreakerAction,
    BreakerDecision,
    BreakerState,
    CircuitBreaker,
    ControlMachine,
    freeze,
)


def _breaker(**kw: float) -> CircuitBreaker:
    params: dict[str, float] = {"tau_enter": 1.0, "tau_exit": 0.5, "tau_warn": 0.4}
    params.update(kw)
    return CircuitBreaker(**params)


class TestCircuitBreakerValidation:
    def test_enter_must_exceed_exit(self) -> None:
        with pytest.raises(ValueError, match="tau_enter"):
            CircuitBreaker(tau_enter=0.5, tau_exit=0.5)

    def test_negative_exit_rejected(self) -> None:
        with pytest.raises(ValueError, match="tau_exit"):
            CircuitBreaker(tau_enter=1.0, tau_exit=-0.1)


class TestCircuitBreakerDecisions:
    def test_normal_below_all(self) -> None:
        d = _breaker().check(fast=0.1, slow=0.1)
        assert d.state is BreakerState.NORMAL
        assert d.action is BreakerAction.PROCEED

    def test_warning_on_slow_drift(self) -> None:
        d = _breaker().check(fast=0.1, slow=0.6)
        assert d.state is BreakerState.WARNING
        assert d.action is BreakerAction.PROCEED

    def test_trip_on_fast_spike(self) -> None:
        d = _breaker().check(fast=1.2, slow=0.1, payload={"x": "biased"})
        assert d.state is BreakerState.INTERVENTION
        assert d.action is BreakerAction.REROUTE
        assert d.frozen_payload == {"x": "biased"}

    def test_mixing_ratio_monotonic(self) -> None:
        b = _breaker()
        assert b.mixing_ratio(0.0) < b.mixing_ratio(1.0) < b.mixing_ratio(2.0)
        assert b.mixing_ratio(1.0) == pytest.approx(0.5)  # 0.5 at tau_enter


class TestHysteresisAntiThrash:
    def test_does_not_flip_in_deadband(self) -> None:
        b = _breaker()  # enter 1.0, exit 0.5
        # Rise past enter, then oscillate inside the [0.5, 1.0] dead-band.
        assert b.check(0.2, 0.0).state is BreakerState.NORMAL
        assert b.check(1.1, 0.0).state is BreakerState.INTERVENTION  # trips once
        for fast in (0.9, 0.6, 0.8, 0.7, 0.95):  # all inside the dead-band
            assert b.check(fast, 0.0).state is BreakerState.INTERVENTION  # stays tripped
        assert b.check(0.4, 0.0).state is BreakerState.NORMAL  # only clears below tau_exit

    def test_clears_and_can_retrip(self) -> None:
        b = _breaker()
        assert b.check(1.2, 0.0).state is BreakerState.INTERVENTION  # trips
        assert b.check(0.3, 0.0).state is BreakerState.NORMAL  # clears below tau_exit
        assert b.check(1.2, 0.0).state is BreakerState.INTERVENTION  # re-trips


class TestFreeze:
    def test_freeze_snapshots_and_isolates(self) -> None:
        original = {"answer": ["a", "b"]}
        frozen = freeze(original)
        original["answer"].append("c")
        assert frozen == {"answer": ["a", "b"]}  # snapshot unaffected

    def test_freeze_none(self) -> None:
        assert freeze(None) is None


class TestControlMachineLifecycle:
    def test_normal_and_warning_proceed(self) -> None:
        m = ControlMachine(_breaker())
        assert m.step(0.1, 0.1).state is BreakerState.NORMAL
        assert m.step(0.1, 0.6).state is BreakerState.WARNING
        assert m.state is BreakerState.WARNING

    def test_spike_triggers_single_intervention(self) -> None:
        calls: list[BreakerDecision] = []
        m = ControlMachine(
            _breaker(), cooldown_steps=2, max_retries=3, intervention_hook=calls.append
        )
        d = m.step(1.2, 0.1, payload={"x": 1})
        assert d.action is BreakerAction.REROUTE
        assert m.state is BreakerState.RECOVERY
        assert m.attempts == 1
        assert len(calls) == 1  # hook fired once

    def test_cooldown_gates_recovery(self) -> None:
        m = ControlMachine(_breaker(), cooldown_steps=2)
        m.step(1.2, 0.1)  # -> intervention, then recovery, cooldown=2
        assert m.step(0.1, 0.1).state is BreakerState.RECOVERY  # cooldown 2->1, still halted
        assert m.step(0.1, 0.1).state is BreakerState.RECOVERY  # cooldown 1->0
        assert m.step(0.1, 0.1).state is BreakerState.NORMAL  # cleared + cooldown elapsed

    def test_recovers_only_after_bnet_clears(self) -> None:
        m = ControlMachine(_breaker(), cooldown_steps=0)
        m.step(1.2, 0.1)  # intervention -> recovery (cooldown 0)
        # Still above tau_exit at recheck -> re-intervenes rather than recovering.
        assert m.step(0.7, 0.1).state is BreakerState.INTERVENTION
        # Now drop below tau_exit -> recover to normal.
        assert m.step(0.2, 0.1).state is BreakerState.NORMAL

    def test_escalates_after_max_retries(self) -> None:
        m = ControlMachine(_breaker(), cooldown_steps=0, max_retries=2)
        # Persistent high bias: never clears -> retries -> escalation.
        states = [m.step(1.2, 0.1).state for _ in range(6)]
        assert BreakerState.ESCALATED in states
        # Terminal: stays escalated.
        assert m.step(0.0, 0.0).state is BreakerState.ESCALATED
        assert m.step(1.2, 0.1).action is BreakerAction.ESCALATE

    def test_machine_validation(self) -> None:
        with pytest.raises(ValueError, match="cooldown_steps"):
            ControlMachine(_breaker(), cooldown_steps=-1)
        with pytest.raises(ValueError, match="max_retries"):
            ControlMachine(_breaker(), max_retries=0)


class TestRetryCounterIsPerIncident:
    """Regression: ``_attempts`` was never reset on a successful recovery, so ``max_retries``
    silently became a lifetime cap -- a long run that self-healed cleanly ``max_retries`` times
    escalated to human review on the strength of its recoveries rather than its failures."""

    def test_clean_recoveries_do_not_accumulate_toward_escalation(self) -> None:
        m = ControlMachine(_breaker(), cooldown_steps=0, max_retries=3)
        for _ in range(5):
            assert m.step(1.2, 0.1).state is BreakerState.INTERVENTION
            assert m.step(0.1, 0.1).state is BreakerState.NORMAL
            assert m.attempts == 0
        assert m.state is BreakerState.NORMAL

    def test_repeated_failures_within_one_incident_still_escalate(self) -> None:
        m = ControlMachine(_breaker(), cooldown_steps=0, max_retries=3)
        m.step(1.2, 0.1)  # attempt 1
        m.step(1.2, 0.1)  # attempt 2
        m.step(1.2, 0.1)  # attempt 3
        assert m.step(1.2, 0.1).state is BreakerState.ESCALATED

    def test_escalated_decision_keeps_the_payload_that_caused_the_halt(self) -> None:
        m = ControlMachine(_breaker(), cooldown_steps=0, max_retries=1)
        m.step(1.2, 0.1, payload={"answer": "biased"})
        assert m.step(1.2, 0.1).state is BreakerState.ESCALATED
        # B_net now falls back, so the breaker un-trips and stops freezing. The terminal decision
        # handed to human review must still carry the payload that caused the halt, not None.
        decision = m.step(0.1, 0.1)
        assert decision.state is BreakerState.ESCALATED
        assert decision.frozen_payload == {"answer": "biased"}
