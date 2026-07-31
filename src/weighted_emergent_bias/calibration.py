"""Threshold calibration -- pick tau from recorded control runs, not a magic constant.

The Phase 2 study showed ``B_net``'s absolute scale is deployment-dependent (it moves with the Katz
attenuation and the graph's propagation dynamics). So a hardcoded ``tau`` would be the M3 analogue
of thresholding raw divergence. Instead, :func:`calibrate_thresholds` takes ``B_net`` trajectories
from *control* (unbiased) runs and sets ``tau_enter`` at the level only a ``target_false_halt_rate``
fraction of control steps exceed -- targeting a false-halt rate directly. The exit threshold is a
fixed fraction below (the hysteresis dead-band), and the slow-scale warn threshold is set the same
way on the slow trajectories.

As with the Phase 1/2 studies: the library ships the *method*, the deployment owns the *number*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .breaker import CircuitBreaker

_EPS = 1e-9


@dataclass(frozen=True)
class ThresholdConfig:
    """Calibrated breaker thresholds. :meth:`breaker` builds a ready :class:`CircuitBreaker`."""

    tau_enter: float
    tau_exit: float
    tau_warn: float
    target_false_halt_rate: float

    def breaker(self) -> CircuitBreaker:
        return CircuitBreaker(
            tau_enter=self.tau_enter, tau_exit=self.tau_exit, tau_warn=self.tau_warn
        )


def calibrate_thresholds(
    control_fast: Sequence[float],
    control_slow: Sequence[float] | None = None,
    *,
    target_false_halt_rate: float = 0.01,
    exit_ratio: float = 0.7,
) -> ThresholdConfig:
    """Choose thresholds targeting ``target_false_halt_rate`` on unbiased control ``B_net``.

    ``control_fast`` are fast-scale ``B_net`` values from control (unbiased) runs; ``control_slow``
    the slow-scale values (defaults to ``control_fast``). ``tau_enter`` is set at the
    ``1 - target`` quantile of control fast, ``tau_exit = exit_ratio * tau_enter`` (the dead-band),
    and ``tau_warn`` at the ``1 - target`` quantile of control slow.
    """
    if not 0.0 < target_false_halt_rate < 1.0:
        raise ValueError(f"target_false_halt_rate must be in (0, 1), got {target_false_halt_rate}")
    if not 0.0 < exit_ratio < 1.0:
        raise ValueError(f"exit_ratio must be in (0, 1), got {exit_ratio}")
    if len(control_fast) == 0:
        raise ValueError("control_fast must be non-empty")

    quantile = 100.0 * (1.0 - target_false_halt_rate)
    fast = np.asarray(control_fast, dtype=np.float64)
    slow = np.asarray(control_slow if control_slow is not None else control_fast, dtype=np.float64)

    tau_enter = max(float(np.percentile(fast, quantile)), _EPS)
    tau_exit = exit_ratio * tau_enter
    tau_warn = max(float(np.percentile(slow, quantile)), 0.0)

    return ThresholdConfig(
        tau_enter=tau_enter,
        tau_exit=tau_exit,
        tau_warn=tau_warn,
        target_false_halt_rate=target_false_halt_rate,
    )
