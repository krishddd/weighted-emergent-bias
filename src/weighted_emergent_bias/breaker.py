"""The control plane -- hysteresis breaker and recovery state machine (M3).

Pure Python, no framework and no numpy: given the `fast`/`slow` ``B_net`` scales from M2, decide
whether to proceed, halt, reroute, or escalate. Two pieces:

- :class:`CircuitBreaker` -- a two-threshold hysteresis detector. It trips at ``tau_enter`` and
  only clears once ``B_net`` falls back below the *lower* ``tau_exit``; the dead-band between them
  is what stops a signal sitting near the line from flipping the breaker every step. It also reports
  a smooth ``mixing_ratio`` (sigmoid severity) as advisory input for M4.
- :class:`ControlMachine` -- the operational lifecycle on top: Normal -> Warning -> Intervention ->
  Recovery, with a cool-down window, a ``max_retries`` cap that escalates to human review instead of
  looping forever, and an injectable intervention hook (M4 plugs in here; the default is a no-op).

The actual repair (skeptics / MADERA) is M4. M3 decides *when* to halt and *whether* a repaired run
may re-enter; it does not repair.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy

from .types import BreakerAction, BreakerDecision, BreakerState, Payload

# Called when the machine enters Intervention; M4 does the repair. Default is a no-op.
InterventionHook = Callable[[BreakerDecision], None]


def _sigmoid(x: float) -> float:
    if x < 0:  # avoid overflow in exp for large positive -x
        z = math.exp(x)
        return z / (1.0 + z)
    return 1.0 / (1.0 + math.exp(-x))


def freeze(payload: Payload | None) -> Payload | None:
    """Return an immutable-enough snapshot of a compromised payload, safe to store and audit."""
    return deepcopy(payload) if payload is not None else None


class CircuitBreaker:
    """Two-threshold hysteresis detector over the ``B_net`` scales.

    ``tau_enter`` trips Intervention (on ``fast``); the breaker stays tripped until ``fast`` falls
    below ``tau_exit`` (< ``tau_enter``). ``tau_warn`` raises a drift Warning on the ``slow`` scale.
    ``kappa`` sets the softness of the advisory ``mixing_ratio``.
    """

    def __init__(
        self,
        *,
        tau_enter: float,
        tau_exit: float,
        tau_warn: float | None = None,
        kappa: float | None = None,
    ) -> None:
        if tau_exit < 0.0:
            raise ValueError(f"tau_exit must be non-negative, got {tau_exit}")
        if tau_enter <= tau_exit:
            raise ValueError(f"tau_enter ({tau_enter}) must exceed tau_exit ({tau_exit})")
        self._tau_enter = tau_enter
        self._tau_exit = tau_exit
        self._tau_warn = tau_exit if tau_warn is None else tau_warn
        self._kappa = max((tau_enter - tau_exit) / 4.0, 1e-9) if kappa is None else kappa
        if self._kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {self._kappa}")
        self._tripped = False

    @property
    def tripped(self) -> bool:
        return self._tripped

    def reset(self) -> None:
        self._tripped = False

    def mixing_ratio(self, fast: float) -> float:
        """Smooth severity in [0, 1]: 0.5 at ``tau_enter``, rising with ``fast``."""
        return _sigmoid((fast - self._tau_enter) / self._kappa)

    def check(self, fast: float, slow: float, *, payload: Payload | None = None) -> BreakerDecision:
        """Update hysteresis and return a Normal / Warning / Intervention decision."""
        if self._tripped:
            if fast < self._tau_exit:
                self._tripped = False
        elif fast >= self._tau_enter:
            self._tripped = True

        mix = self.mixing_ratio(fast)
        if self._tripped:
            return BreakerDecision(
                state=BreakerState.INTERVENTION,
                action=BreakerAction.REROUTE,
                mixing_ratio=mix,
                b_net=fast,
                reason=f"fast B_net {fast:.3f} >= tau_enter {self._tau_enter:.3f}",
                frozen_payload=freeze(payload),
            )
        if slow >= self._tau_warn:
            return BreakerDecision(
                state=BreakerState.WARNING,
                action=BreakerAction.PROCEED,
                mixing_ratio=mix,
                b_net=fast,
                reason=f"slow B_net {slow:.3f} >= tau_warn {self._tau_warn:.3f} (drift)",
            )
        return BreakerDecision(
            state=BreakerState.NORMAL,
            action=BreakerAction.PROCEED,
            mixing_ratio=mix,
            b_net=fast,
            reason="below thresholds",
        )


class ControlMachine:
    """The Normal -> Warning -> Intervention -> Recovery lifecycle with cool-down and escalation.

    Wraps a :class:`CircuitBreaker` for threshold detection and layers recovery on top. Feed it the
    ``fast``/``slow`` scales once per superstep via :meth:`step`. On entering Intervention it calls
    ``intervention_hook`` (where M4 repairs); recovery to Normal requires the cool-down to elapse
    *and* ``B_net`` to have fallen below ``tau_exit`` (re-measured, not assumed). After
    ``max_retries`` failed attempts it escalates to a terminal ESCALATED state.
    """

    def __init__(
        self,
        breaker: CircuitBreaker,
        *,
        cooldown_steps: int = 2,
        max_retries: int = 3,
        intervention_hook: InterventionHook | None = None,
    ) -> None:
        if cooldown_steps < 0:
            raise ValueError(f"cooldown_steps must be non-negative, got {cooldown_steps}")
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self._breaker = breaker
        self._cooldown_steps = cooldown_steps
        self._max_retries = max_retries
        self._hook: InterventionHook = intervention_hook or (lambda _d: None)
        self._state = BreakerState.NORMAL
        self._cooldown = 0
        self._attempts = 0

    @property
    def state(self) -> BreakerState:
        return self._state

    @property
    def attempts(self) -> int:
        return self._attempts

    def _decision(
        self,
        state: BreakerState,
        action: BreakerAction,
        raw: BreakerDecision,
        reason: str,
    ) -> BreakerDecision:
        return BreakerDecision(
            state=state,
            action=action,
            mixing_ratio=raw.mixing_ratio,
            b_net=raw.b_net,
            reason=reason,
            frozen_payload=raw.frozen_payload,
        )

    def _enter_intervention(self, raw: BreakerDecision) -> BreakerDecision:
        self._state = BreakerState.INTERVENTION
        self._attempts += 1
        decision = self._decision(
            BreakerState.INTERVENTION,
            BreakerAction.REROUTE,
            raw,
            f"intervention attempt {self._attempts}",
        )
        self._hook(decision)  # M4 acts here (default no-op)
        self._state = BreakerState.RECOVERY
        self._cooldown = self._cooldown_steps
        return decision

    def step(self, fast: float, slow: float, *, payload: Payload | None = None) -> BreakerDecision:
        """Advance one superstep; return the deterministic control decision."""
        raw = self._breaker.check(fast, slow, payload=payload)

        if self._state is BreakerState.ESCALATED:
            return self._decision(
                BreakerState.ESCALATED, BreakerAction.ESCALATE, raw, "escalated to human review"
            )

        if self._state in (BreakerState.NORMAL, BreakerState.WARNING):
            if raw.state is BreakerState.INTERVENTION:
                return self._enter_intervention(raw)
            self._state = raw.state
            return self._decision(raw.state, BreakerAction.PROCEED, raw, raw.reason)

        # RECOVERY
        if self._cooldown > 0:
            self._cooldown -= 1
            return self._decision(
                BreakerState.RECOVERY, BreakerAction.HALT, raw, f"cool-down, {self._cooldown} left"
            )
        if raw.state is not BreakerState.INTERVENTION:
            self._state = BreakerState.NORMAL
            return self._decision(
                BreakerState.NORMAL, BreakerAction.PROCEED, raw, "recovered: B_net below tau_exit"
            )
        if self._attempts >= self._max_retries:
            self._state = BreakerState.ESCALATED
            return self._decision(
                BreakerState.ESCALATED, BreakerAction.ESCALATE, raw, "max retries exceeded"
            )
        return self._enter_intervention(raw)
