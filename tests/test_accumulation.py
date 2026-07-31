"""Tests for the per-node magnitude that feeds network accumulation.

The n-invariance test is the reason this function exists: the WP7 study showed effect size inflates
with sample count, so M2 must weight on something stable. node_magnitude (excess divergence) is
stable across n where effect_size is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from weighted_emergent_bias import (
    BiasScore,
    DivergenceMethod,
    TaskMode,
    compute_local_bias,
    node_magnitude,
)
from weighted_emergent_bias.scoring.divergence import softmax
from weighted_emergent_bias.testing import FakeLLMClient
from weighted_emergent_bias.testing import markers as mk
from weighted_emergent_bias.types import FloatArray

SEED = 20260731


def _score(**overrides: object) -> BiasScore:
    kwargs: dict[str, object] = {
        "effect_size": 5.0,
        "ci_low": 2.0,
        "ci_high": 8.0,
        "p_value": 0.01,
        "method": DivergenceMethod.JENSEN_SHANNON,
        "task_mode": TaskMode.CHOICE,
        "n_samples": 8,
        "raw_divergence": 0.40,
        "null_mean": 0.10,
        "null_std": 0.05,
        "axis": "gender",
    }
    kwargs.update(overrides)
    return BiasScore(**kwargs)  # type: ignore[arg-type]


class TestGating:
    def test_significant_returns_excess_divergence(self) -> None:
        assert node_magnitude(_score(p_value=0.01)) == pytest.approx(0.30)  # 0.40 - 0.10

    def test_insignificant_is_zero(self) -> None:
        assert node_magnitude(_score(p_value=0.20)) == 0.0

    def test_gate_boundary(self) -> None:
        assert node_magnitude(_score(p_value=0.05), alpha=0.05) == pytest.approx(0.30)
        assert node_magnitude(_score(p_value=0.051), alpha=0.05) == 0.0

    def test_excess_clamped_at_zero(self) -> None:
        # raw below the noise floor mean should never yield a negative magnitude.
        assert node_magnitude(_score(raw_divergence=0.05, null_mean=0.10)) == 0.0

    def test_custom_alpha(self) -> None:
        assert node_magnitude(_score(p_value=0.08), alpha=0.10) == pytest.approx(0.30)


async def _choice_group(client: FakeLLMClient, prompt: str, n: int) -> list[FloatArray]:
    return [softmax(await client.score_candidates(prompt, client.candidates)) for _ in range(n)]


class TestNInvariance:
    async def test_magnitude_stable_across_n_while_effect_size_drifts(self) -> None:
        """The load-bearing property: b_i is ~n-invariant; effect_size is not."""
        magnitudes: list[float] = []
        effects: list[float] = []
        for n in (5, 12, 30):
            client = FakeLLMClient(
                np.random.default_rng(SEED), n_candidates=5, bias={"gender": 4.0}, logit_noise=0.5
            )
            base = await _choice_group(client, "the applicant", n)
            cf = await _choice_group(client, mk.mark("gender", "alt"), n)
            score = compute_local_bias(
                base,
                cf,
                task_mode=TaskMode.CHOICE,
                rng=np.random.default_rng(1),
                n_permutations=400,
            )
            magnitudes.append(node_magnitude(score))
            effects.append(score.effect_size)

        # Effect size inflates substantially with n (the WP7 finding).
        assert effects[-1] > effects[0] * 2
        # Magnitude stays within a tight band across the same n range.
        spread = max(magnitudes) - min(magnitudes)
        assert spread < 0.15, f"node_magnitude not n-stable: {magnitudes}"
        assert all(m > 0 for m in magnitudes)
