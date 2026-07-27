"""Tests for the divergence estimators.

The disjoint-support test is the important one: it pins JSD to exactly 1.0 where a KL-based
implementation would return inf. That is the whole reason true Jensen-Shannon was chosen, and
the reason the review's symmetrized-KL "fix" is not adopted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from weighted_emergent_bias import BiasScore, DivergenceMethod, TaskMode
from weighted_emergent_bias.scoring import (
    assert_same_estimator,
    cosine_distance,
    jensen_shannon,
    js_divergence_from_logits,
    raw_divergence,
    softmax,
)
from weighted_emergent_bias.scoring.divergence import _kl_bits


def _score(**overrides: object) -> BiasScore:
    """A valid BiasScore with fields overridable per test (local to avoid cross-module import)."""
    kwargs: dict[str, object] = {
        "effect_size": 2.0,
        "ci_low": 1.0,
        "ci_high": 3.0,
        "p_value": 0.01,
        "method": DivergenceMethod.JENSEN_SHANNON,
        "task_mode": TaskMode.CHOICE,
        "n_samples": 20,
        "raw_divergence": 0.4,
        "null_mean": 0.1,
        "null_std": 0.15,
        "axis": "gender",
    }
    kwargs.update(overrides)
    return BiasScore(**kwargs)  # type: ignore[arg-type]


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        assert softmax([1.0, 2.0, 3.0]).sum() == pytest.approx(1.0)

    def test_uniform_logits_give_uniform(self) -> None:
        np.testing.assert_allclose(softmax([5.0, 5.0, 5.0]), [1 / 3, 1 / 3, 1 / 3])

    def test_numerically_stable_on_large_logits(self) -> None:
        out = softmax([1000.0, 1001.0])
        assert math.isfinite(float(out.sum()))
        assert out.sum() == pytest.approx(1.0)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            softmax([])


class TestJensenShannon:
    def test_identical_is_zero(self) -> None:
        assert jensen_shannon([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        p, q = [0.7, 0.2, 0.1], [0.1, 0.3, 0.6]
        assert jensen_shannon(p, q) == pytest.approx(jensen_shannon(q, p))

    def test_disjoint_supports_is_one(self) -> None:
        # KL(p||q) would be +inf here; true JSD is exactly 1.0 (log base 2).
        assert jensen_shannon([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_zero_entries_handled(self) -> None:
        # 0*log0 = 0 convention; no nan.
        val = jensen_shannon([1.0, 0.0], [0.5, 0.5])
        assert math.isfinite(val) and 0.0 <= val <= 1.0

    def test_known_value_half_overlap(self) -> None:
        # p=[1,0], q=[0.5,0.5]: M=[0.75,0.25].
        # KL(p||M) = 1*log2(1/0.75) = log2(4/3); KL(q||M)=0.5*log2(0.5/0.75)+0.5*log2(0.5/0.25)
        expected = 0.5 * math.log2(4 / 3) + 0.5 * (
            0.5 * math.log2(0.5 / 0.75) + 0.5 * math.log2(0.5 / 0.25)
        )
        assert jensen_shannon([1.0, 0.0], [0.5, 0.5]) == pytest.approx(expected)

    def test_rejects_mismatched_support(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            jensen_shannon([0.5, 0.5], [1 / 3, 1 / 3, 1 / 3])

    def test_rejects_non_normalized(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            jensen_shannon([0.5, 0.4], [0.5, 0.5])

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            jensen_shannon([1.5, -0.5], [0.5, 0.5])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            jensen_shannon([], [])


class TestKLBitsHelper:
    def test_kl_of_equal_is_zero(self) -> None:
        p = np.array([0.5, 0.5])
        assert _kl_bits(p, p) == pytest.approx(0.0)


class TestJSFromLogits:
    def test_matches_manual_softmax(self) -> None:
        a, b = [2.0, 1.0, 0.0], [0.0, 1.0, 2.0]
        assert js_divergence_from_logits(a, b) == pytest.approx(
            jensen_shannon(softmax(a), softmax(b))
        )

    def test_equal_logits_zero(self) -> None:
        assert js_divergence_from_logits([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)


class TestCosineDistance:
    def test_identical_is_zero(self) -> None:
        assert cosine_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)

    def test_scaled_same_direction_is_zero(self) -> None:
        assert cosine_distance([1.0, 0.0], [5.0, 0.0]) == pytest.approx(0.0)

    def test_orthogonal_is_one(self) -> None:
        assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_opposite_is_two(self) -> None:
        assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)

    def test_symmetric(self) -> None:
        u, v = [1.0, 2.0, 3.0], [3.0, 1.0, 2.0]
        assert cosine_distance(u, v) == pytest.approx(cosine_distance(v, u))

    def test_zero_vector_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero-norm"):
            cosine_distance([0.0, 0.0], [1.0, 1.0])

    def test_mismatched_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal shape"):
            cosine_distance([1.0, 0.0, 0.0], [1.0, 0.0])


class TestRawDivergenceDispatch:
    def test_choice_uses_js(self) -> None:
        val, method = raw_divergence([2.0, 0.0], [0.0, 2.0], task_mode=TaskMode.CHOICE)
        assert method is DivergenceMethod.JENSEN_SHANNON
        assert 0.0 <= val <= 1.0

    def test_generative_uses_embedding(self) -> None:
        val, method = raw_divergence([1.0, 0.0], [0.0, 1.0], task_mode=TaskMode.GENERATIVE)
        assert method is DivergenceMethod.EMBEDDING
        assert val == pytest.approx(1.0)


class TestAssertSameEstimator:
    def test_empty_ok(self) -> None:
        assert_same_estimator([])

    def test_same_estimator_ok(self) -> None:
        assert_same_estimator([_score(), _score(axis="race")])

    def test_mixed_method_rejected(self) -> None:
        with pytest.raises(ValueError, match="different estimators"):
            assert_same_estimator([_score(), _score(method=DivergenceMethod.EMBEDDING)])

    def test_mixed_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="different estimators"):
            assert_same_estimator(
                [
                    _score(method=DivergenceMethod.EMBEDDING, task_mode=TaskMode.CHOICE),
                    _score(method=DivergenceMethod.EMBEDDING, task_mode=TaskMode.GENERATIVE),
                ]
            )


# --- Property tests --------------------------------------------------------------------------

_probs = st.integers(min_value=2, max_value=8).flatmap(
    lambda n: st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=n, max_size=n).filter(
        lambda xs: sum(xs) > 0
    )
)


def _normalize(xs: list[float]) -> list[float]:
    total = sum(xs)
    return [x / total for x in xs]


@settings(max_examples=300, deadline=None)
@given(_probs, _probs)
def test_js_is_bounded_and_symmetric(p_raw: list[float], q_raw: list[float]) -> None:
    if len(p_raw) != len(q_raw):
        return  # shared support only
    p, q = _normalize(p_raw), _normalize(q_raw)
    d_pq = jensen_shannon(p, q)
    d_qp = jensen_shannon(q, p)
    assert 0.0 <= d_pq <= 1.0 + 1e-12
    assert d_pq == pytest.approx(d_qp)


@settings(max_examples=200, deadline=None)
@given(_probs)
def test_js_of_self_is_zero(p_raw: list[float]) -> None:
    p = _normalize(p_raw)
    assert jensen_shannon(p, p) == pytest.approx(0.0, abs=1e-12)
