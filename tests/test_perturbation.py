"""Tests for the counterfactual perturbation engine.

The contract is narrow and strict: perturbation changes *only* the targeted axis's tokens in
string leaves, and nothing else — not structure, not keys, not non-string values, not
untargeted words. The property tests enforce that on arbitrary nested payloads.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from weighted_emergent_bias import (
    AxisSpec,
    PerturbationKind,
    Substitution,
    perturb,
)

GENDER = AxisSpec(
    name="gender",
    substitutions=(
        Substitution("he", "she"),
        Substitution("his", "her"),
        Substitution("man", "woman"),
    ),
)


class TestConfigValidation:
    def test_empty_axes_rejected(self) -> None:
        with pytest.raises(ValueError, match="no default axis set"):
            perturb("he ran", [])

    def test_axis_needs_a_substitution(self) -> None:
        with pytest.raises(ValueError, match="at least one substitution"):
            AxisSpec(name="gender", substitutions=())

    def test_empty_axis_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name must be non-empty"):
            AxisSpec(name="", substitutions=(Substitution("a", "b"),))

    def test_empty_pattern_rejected(self) -> None:
        with pytest.raises(ValueError, match="pattern must be non-empty"):
            Substitution("", "b")


class TestStringPayload:
    def test_basic_substitution(self) -> None:
        (p,) = perturb("he took his coat", [GENDER])
        assert p.perturbed == "she took her coat"
        assert p.original == "he took his coat"
        assert p.axis == "gender"
        assert p.kind is PerturbationKind.EXPLICIT

    def test_case_preservation(self) -> None:
        (p,) = perturb("He and HIS dog", [GENDER])
        assert p.perturbed == "She and HER dog"

    def test_word_boundary_respected(self) -> None:
        # "the" contains "he" but must not be touched; "heman" is not the word "he".
        (p,) = perturb("the theme here", [GENDER])
        assert p.perturbed == "the theme here"

    def test_absent_axis_is_identity(self) -> None:
        (p,) = perturb("the weather is fine", [GENDER])
        assert p.perturbed == p.original


class TestStructuredPayload:
    def test_only_string_leaves_change(self) -> None:
        payload = {
            "prompt": "he is the manager",
            "score": 42,
            "tags": ["his record", 7, "unrelated"],
            "meta": {"note": "man of the year"},
        }
        (p,) = perturb(payload, [GENDER])
        assert p.perturbed == {
            "prompt": "she is the manager",  # "manager" untouched (word boundary)
            "score": 42,
            "tags": ["her record", 7, "unrelated"],
            "meta": {"note": "woman of the year"},
        }

    def test_keys_are_not_perturbed(self) -> None:
        # A dict key that matches a pattern must stay put; only values are text to perturb.
        (p,) = perturb({"he": "he wins"}, [GENDER])
        assert p.perturbed == {"he": "she wins"}

    def test_tuple_stays_tuple(self) -> None:
        (p,) = perturb(("he", "his"), [GENDER])
        assert p.perturbed == ("she", "her")
        assert isinstance(p.perturbed, tuple)


class TestAxisGrouping:
    def test_one_perturbation_per_axis(self) -> None:
        age = AxisSpec(name="age", substitutions=(Substitution("young", "old"),))
        out = perturb("he is young", [GENDER, age])
        assert [p.axis for p in out] == ["gender", "age"]

    def test_explicit_and_proxy_are_separate(self) -> None:
        axis = AxisSpec(
            name="race",
            substitutions=(
                Substitution("him", "her", kind=PerturbationKind.EXPLICIT),
                Substitution("90210", "10453", kind=PerturbationKind.PROXY),
            ),
        )
        out = perturb("call him at 90210", [axis])
        kinds = {p.kind: p.perturbed for p in out}
        assert kinds[PerturbationKind.EXPLICIT] == "call her at 90210"  # proxy untouched
        assert kinds[PerturbationKind.PROXY] == "call him at 10453"  # explicit untouched
        # Stable order: explicit group emitted before proxy group.
        assert [p.kind for p in out] == [PerturbationKind.EXPLICIT, PerturbationKind.PROXY]


# --- Property tests on arbitrary nested payloads -------------------------------------------

_SAFE_TEXT = st.text(alphabet="abcdefg ", max_size=20)  # never contains axis patterns


def _payloads(leaves: st.SearchStrategy[Any]) -> st.SearchStrategy[Any]:
    return st.recursive(
        leaves | st.integers() | st.booleans(),
        lambda children: (
            st.lists(children, max_size=4)
            | st.dictionaries(st.text(alphabet="xyz", max_size=5), children, max_size=4)
        ),
        max_leaves=15,
    )


def _same_shape(a: Any, b: Any) -> bool:
    """True if a and b have identical structure and identical non-string leaves."""
    if isinstance(a, str):
        return isinstance(b, str)
    if isinstance(a, dict):
        return (
            isinstance(b, dict) and a.keys() == b.keys() and all(_same_shape(a[k], b[k]) for k in a)
        )
    if isinstance(a, list):
        return (
            isinstance(b, list)
            and len(a) == len(b)
            and all(_same_shape(x, y) for x, y in zip(a, b, strict=True))
        )
    return bool(a == b)  # non-string scalar must be unchanged


@settings(max_examples=200, deadline=None)
@given(_payloads(_SAFE_TEXT))
def test_absent_axis_is_identity_on_any_structure(payload: Any) -> None:
    (p,) = perturb(payload, [GENDER])
    assert p.perturbed == payload


@settings(max_examples=200, deadline=None)
@given(_payloads(st.text(max_size=20)))
def test_structure_is_always_preserved(payload: Any) -> None:
    (p,) = perturb(payload, [GENDER])
    assert _same_shape(payload, p.perturbed)
