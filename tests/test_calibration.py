"""Tests for threshold calibration."""

from __future__ import annotations

import numpy as np
import pytest

from weighted_emergent_bias import CircuitBreaker, calibrate_thresholds


class TestCalibrateThresholds:
    def test_achieves_target_false_halt_rate(self, rng: np.random.Generator) -> None:
        # Control B_net ~ noise; calibrate at 5%, then check false-halt rate on held-out control.
        control = rng.uniform(0.0, 1.0, size=5000).tolist()
        cfg = calibrate_thresholds(control, target_false_halt_rate=0.05)
        holdout = rng.uniform(0.0, 1.0, size=5000)
        false_halt_rate = float((holdout > cfg.tau_enter).mean())
        assert false_halt_rate == pytest.approx(0.05, abs=0.02)

    def test_enter_exceeds_exit(self, rng: np.random.Generator) -> None:
        cfg = calibrate_thresholds(rng.uniform(0, 1, 1000).tolist(), exit_ratio=0.7)
        assert cfg.tau_enter > cfg.tau_exit
        assert cfg.tau_exit == pytest.approx(0.7 * cfg.tau_enter)

    def test_all_zero_control_still_valid(self) -> None:
        # Degenerate control (never any bias) must still yield a usable breaker (enter > exit).
        cfg = calibrate_thresholds([0.0] * 100)
        assert cfg.tau_enter > cfg.tau_exit
        cfg.breaker()  # must not raise

    def test_breaker_convenience(self, rng: np.random.Generator) -> None:
        cfg = calibrate_thresholds(rng.uniform(0, 1, 500).tolist())
        assert isinstance(cfg.breaker(), CircuitBreaker)

    def test_separate_slow_scale(self, rng: np.random.Generator) -> None:
        fast = rng.uniform(0.0, 1.0, 1000).tolist()
        slow = rng.uniform(0.0, 0.3, 1000).tolist()  # slow scale is smaller
        cfg = calibrate_thresholds(fast, slow, target_false_halt_rate=0.05)
        assert cfg.tau_warn < cfg.tau_enter  # warn calibrated on the smaller slow scale

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="target_false_halt_rate"):
            calibrate_thresholds([0.1, 0.2], target_false_halt_rate=1.5)
        with pytest.raises(ValueError, match="exit_ratio"):
            calibrate_thresholds([0.1, 0.2], exit_ratio=1.0)
        with pytest.raises(ValueError, match="non-empty"):
            calibrate_thresholds([])


class TestSampleSufficiency:
    """A quantile cannot resolve a rate finer than 1/n. Returning a threshold whose stated
    false-halt rate the control data cannot support is the M3 analogue of a magic number."""

    def test_rejects_too_few_control_samples_for_the_target_rate(self) -> None:
        with pytest.raises(ValueError, match="at least 100 control_fast samples"):
            calibrate_thresholds([0.1] * 10, target_false_halt_rate=0.01)

    def test_accepts_exactly_enough_samples(self) -> None:
        cfg = calibrate_thresholds(list(np.linspace(0.0, 1.0, 100)), target_false_halt_rate=0.01)
        assert cfg.tau_enter > 0.0

    def test_rejects_empty_control_slow(self) -> None:
        with pytest.raises(ValueError, match="control_slow must be non-empty"):
            calibrate_thresholds([0.1] * 20, [], target_false_halt_rate=0.05)
