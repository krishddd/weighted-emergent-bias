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
    NetworkAccumulator,
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


class TestNetworkAccumulator:
    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="fast_alpha"):
            NetworkAccumulator(fast_alpha=0.0)
        with pytest.raises(ValueError, match="slow_alpha"):
            NetworkAccumulator(slow_alpha=1.5)
        with pytest.raises(ValueError, match="reduction"):
            NetworkAccumulator(reduction="median")

    def test_initial_state_is_zero(self) -> None:
        acc = NetworkAccumulator()
        assert acc.state.fast == 0.0
        assert acc.state.slow == 0.0
        assert acc.state.step == 0

    def test_warmup_correction_exact_on_constant_input(self) -> None:
        # Bias correction makes a constant input read at its true level from step 1 -- both scales,
        # every step, exactly (no ramp-up from zero).
        acc = NetworkAccumulator(fast_alpha=0.7, slow_alpha=0.1)
        mags = {"a": 0.4}
        weights = {"a": 1.0}
        for _ in range(5):
            state = acc.update(mags, weights)
            assert state.fast == pytest.approx(0.4)
            assert state.slow == pytest.approx(0.4)

    def test_fast_reacts_more_than_slow_to_a_spike(self) -> None:
        acc = NetworkAccumulator(fast_alpha=0.7, slow_alpha=0.1)
        weights = {"a": 1.0}
        for _ in range(5):
            acc.update({"a": 0.0}, weights)  # quiet steps
        state = acc.update({"a": 1.0}, weights)  # a spike
        assert state.fast > state.slow

    def test_superstep_reduction_is_order_independent(self) -> None:
        # Shuffling the firing set's dict order must not change B_net.
        weights = {"a": 0.5, "b": 0.3, "c": 0.2}
        mags_fwd = {"a": 0.1, "b": 0.9, "c": 0.5}
        mags_rev = {"c": 0.5, "b": 0.9, "a": 0.1}
        s1 = NetworkAccumulator().update(mags_fwd, weights)
        s2 = NetworkAccumulator().update(mags_rev, weights)
        assert s1.fast == pytest.approx(s2.fast)
        assert s1.slow == pytest.approx(s2.slow)

    def test_weighted_mean_favors_high_weight_node(self) -> None:
        # weighted_mean: a biased central (high-w) node dominates over a clean leaf.
        acc = NetworkAccumulator(reduction="weighted_mean")
        state = acc.update({"central": 1.0, "leaf": 0.0}, {"central": 0.9, "leaf": 0.1})
        assert state.fast == pytest.approx(0.9)  # 0.9*1 + 0.1*0 / 1.0

    def test_max_reduction(self) -> None:
        acc = NetworkAccumulator(reduction="max")
        state = acc.update({"a": 0.2, "b": 0.7}, {"a": 0.9, "b": 0.1})
        assert state.fast == pytest.approx(0.7)  # worst raw magnitude, weight-agnostic

    def test_empty_superstep_decays(self) -> None:
        acc = NetworkAccumulator(fast_alpha=0.7, slow_alpha=0.1)
        acc.update({"a": 1.0}, {"a": 1.0})
        after_spike = acc.state.fast
        state = acc.update({}, {})  # quiet step
        assert state.fast < after_spike

    def test_constant_input_converges_both_scales(self) -> None:
        acc = NetworkAccumulator()
        weights = {"a": 1.0}
        for _ in range(50):
            state = acc.update({"a": 0.3}, weights)
        assert state.fast == pytest.approx(0.3)
        assert state.slow == pytest.approx(0.3)

    def test_missing_weight_raises(self) -> None:
        acc = NetworkAccumulator()
        with pytest.raises(ValueError, match="no dependency weight"):
            acc.update({"a": 0.5}, {"b": 1.0})
