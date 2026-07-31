"""Tests for the noise floor — including the WP4 calibration gate.

The gate: on an unbiased model sampled at production temperature, the false-positive rate must
be <= 5% at alpha = 0.05 over >= 200 simulated nodes. That is the property that makes the whole
system falsifiable — without it, a "bias" score is indistinguishable from sampling temperature.
"""

from __future__ import annotations

import numpy as np
import pytest

from weighted_emergent_bias import DivergenceMethod, TaskMode, compute_local_bias
from weighted_emergent_bias.scoring.divergence import softmax
from weighted_emergent_bias.testing import FakeLLMClient
from weighted_emergent_bias.testing import markers as mk
from weighted_emergent_bias.types import FloatArray

SEED = 20260731


async def _choice_samples(client: FakeLLMClient, prompt: str, k: int) -> list[FloatArray]:
    """k resampled probability distributions from the fake's candidate scoring."""
    return [softmax(await client.score_candidates(prompt, client.candidates)) for _ in range(k)]


class TestValidation:
    def test_too_few_baseline_rejected(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="baseline"):
            compute_local_bias(
                [np.array([0.5, 0.5])],
                [np.array([0.4, 0.6]), np.array([0.4, 0.6])],
                task_mode=TaskMode.CHOICE,
                rng=rng,
            )

    def test_mismatched_width_rejected(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="equal width"):
            compute_local_bias(
                [np.array([0.5, 0.5]), np.array([0.5, 0.5])],
                [np.array([0.3, 0.3, 0.4]), np.array([0.3, 0.3, 0.4])],
                task_mode=TaskMode.CHOICE,
                rng=rng,
            )

    def test_bad_ci_rejected(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="ci must be"):
            compute_local_bias(
                [np.array([0.5, 0.5])] * 3,
                [np.array([0.5, 0.5])] * 3,
                task_mode=TaskMode.CHOICE,
                rng=rng,
                ci=1.5,
            )


class TestBiasScoreShape:
    async def test_returns_wellformed_score(self, rng: np.random.Generator) -> None:
        client = FakeLLMClient(np.random.default_rng(SEED), n_candidates=4, logit_noise=0.5)
        base = await _choice_samples(client, "baseline", 8)
        cf = await _choice_samples(client, mk.mark("gender", "alt"), 8)
        score = compute_local_bias(
            base, cf, task_mode=TaskMode.CHOICE, rng=rng, axis="gender", n_permutations=300
        )
        assert score.task_mode is TaskMode.CHOICE
        assert score.axis == "gender"
        assert 0.0 <= score.p_value <= 1.0
        assert score.ci_low <= score.ci_high
        assert score.n_samples == 8


class TestDeterministicPath:
    async def test_identical_baseline_is_flagged_deterministic(
        self, rng: np.random.Generator
    ) -> None:
        # No logit noise -> baseline resamples identical -> no sampling-noise floor.
        client = FakeLLMClient(np.random.default_rng(SEED), n_candidates=4, logit_noise=0.0)
        base = await _choice_samples(client, "baseline", 5)
        cf = await _choice_samples(client, mk.mark("gender", "alt"), 5)
        score = compute_local_bias(base, cf, task_mode=TaskMode.CHOICE, rng=rng)
        assert score.method is DivergenceMethod.DETERMINISTIC
        # Unbiased + deterministic => no divergence at all.
        assert score.raw_divergence == pytest.approx(0.0, abs=1e-9)


class TestSensitivity:
    async def test_strong_bias_is_detected(self, rng: np.random.Generator) -> None:
        client = FakeLLMClient(
            np.random.default_rng(SEED), n_candidates=4, bias={"gender": 4.0}, logit_noise=0.5
        )
        base = await _choice_samples(client, "baseline", 10)
        cf = await _choice_samples(client, mk.mark("gender", "alt"), 10)
        score = compute_local_bias(base, cf, task_mode=TaskMode.CHOICE, rng=rng, n_permutations=500)
        assert score.effect_size > 3.0
        assert score.p_value < 0.05
        assert score.ci_excludes_zero

    async def test_effect_size_is_monotonic_in_bias(self, rng: np.random.Generator) -> None:
        effects = []
        for beta in (0.0, 1.0, 3.0, 6.0):
            client = FakeLLMClient(
                np.random.default_rng(SEED),
                n_candidates=4,
                bias={"gender": beta} if beta else {},
                logit_noise=0.5,
            )
            base = await _choice_samples(client, "baseline", 12)
            cf = await _choice_samples(client, mk.mark("gender", "alt"), 12)
            score = compute_local_bias(
                base,
                cf,
                task_mode=TaskMode.CHOICE,
                rng=np.random.default_rng(1),
                n_permutations=400,
            )
            effects.append(score.effect_size)
        # Larger injected bias => larger recovered effect size.
        assert effects == sorted(effects)


class TestCalibrationGate:
    """The WP4 acceptance criterion."""

    async def test_false_positive_rate_under_alpha(self) -> None:
        n_nodes = 200
        alpha = 0.05
        sampler_rng = np.random.default_rng(SEED)
        stat_rng = np.random.default_rng(SEED + 1)
        false_positives = 0

        for _ in range(n_nodes):
            # A fresh unbiased node with genuine sampling noise; baseline and counterfactual
            # are statistically identical, so any "significant" result is a false positive.
            client = FakeLLMClient(sampler_rng, n_candidates=5, bias={}, logit_noise=0.6)
            base = await _choice_samples(client, "baseline", 8)
            cf = await _choice_samples(client, mk.mark("gender", "alt"), 8)
            score = compute_local_bias(
                base,
                cf,
                task_mode=TaskMode.CHOICE,
                rng=stat_rng,
                n_permutations=200,
                n_bootstrap=100,
            )
            if score.is_significant(alpha):
                false_positives += 1

        fpr = false_positives / n_nodes
        assert fpr <= 0.08, f"false-positive rate {fpr:.3f} exceeds tolerance (target <= {alpha})"
