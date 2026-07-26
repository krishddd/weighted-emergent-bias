"""Unit tests for the core value types and their invariants."""

from __future__ import annotations

import pytest

from weighted_emergent_bias import (
    BiasScore,
    DivergenceMethod,
    Perturbation,
    PerturbationKind,
    TaskMode,
)


def _score(**overrides: object) -> BiasScore:
    """A valid BiasScore with fields overridable per test."""
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


class TestBiasScore:
    def test_valid_construction(self) -> None:
        s = _score()
        assert s.effect_size == 2.0
        assert s.ci == (1.0, 3.0)
        assert s.axis == "gender"

    def test_frozen(self) -> None:
        s = _score()
        with pytest.raises((AttributeError, TypeError)):
            s.effect_size = 5.0  # type: ignore[misc]

    @pytest.mark.parametrize("bad_p", [-0.01, 1.01, 2.0, -1.0])
    def test_rejects_out_of_range_p_value(self, bad_p: float) -> None:
        with pytest.raises(ValueError, match="p_value"):
            _score(p_value=bad_p)

    def test_rejects_inverted_ci(self) -> None:
        with pytest.raises(ValueError, match="ci_low"):
            _score(ci_low=3.0, ci_high=1.0)

    def test_rejects_negative_samples(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            _score(n_samples=-1)

    def test_rejects_negative_null_std(self) -> None:
        with pytest.raises(ValueError, match="null_std"):
            _score(null_std=-0.1)

    def test_degenerate_ci_allowed(self) -> None:
        # A point interval (low == high) is valid — e.g. a deterministic node.
        assert _score(ci_low=2.0, ci_high=2.0).ci == (2.0, 2.0)

    @pytest.mark.parametrize(
        ("p", "alpha", "expected"),
        [(0.01, 0.05, True), (0.05, 0.05, True), (0.06, 0.05, False), (0.5, 0.1, False)],
    )
    def test_is_significant(self, p: float, alpha: float, expected: bool) -> None:
        assert _score(p_value=p).is_significant(alpha) is expected

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
    def test_is_significant_rejects_bad_alpha(self, bad_alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            _score().is_significant(bad_alpha)

    @pytest.mark.parametrize(("low", "expected"), [(0.5, True), (0.0, False), (-0.5, False)])
    def test_ci_excludes_zero(self, low: float, expected: bool) -> None:
        assert _score(ci_low=low, ci_high=low + 1.0).ci_excludes_zero is expected

    def test_axis_defaults_to_none(self) -> None:
        assert _score(axis=None).axis is None


class TestPerturbation:
    def test_valid(self) -> None:
        p = Perturbation(axis="race", original="a", perturbed="b")
        assert p.kind is PerturbationKind.EXPLICIT

    def test_proxy_kind(self) -> None:
        p = Perturbation(axis="race", original="a", perturbed="b", kind=PerturbationKind.PROXY)
        assert p.kind is PerturbationKind.PROXY

    def test_empty_axis_rejected(self) -> None:
        with pytest.raises(ValueError, match="axis"):
            Perturbation(axis="", original="a", perturbed="b")


class TestEnums:
    def test_task_mode_is_str(self) -> None:
        assert TaskMode.CHOICE.value == "choice"
        assert isinstance(TaskMode.CHOICE, str)

    def test_divergence_method_is_str(self) -> None:
        assert DivergenceMethod.EMBEDDING.value == "embedding"
        assert isinstance(DivergenceMethod.EMBEDDING, str)

    def test_distinct_modes(self) -> None:
        assert len(set(TaskMode)) == 2
