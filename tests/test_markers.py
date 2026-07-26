"""Tests for the fake-only prompt-marker convention."""

from __future__ import annotations

import pytest

from weighted_emergent_bias.testing import markers


def test_mark_format() -> None:
    assert markers.mark("gender", "female") == "[[gender=female]]"


def test_decode_single() -> None:
    assert markers.decode("hello [[gender=female]] world") == {"gender": "female"}


def test_decode_multiple_axes() -> None:
    prompt = f"{markers.mark('gender', 'female')} and {markers.mark('race', 'black')}"
    assert markers.decode(prompt) == {"gender": "female", "race": "black"}


def test_decode_empty() -> None:
    assert markers.decode("no markers here") == {}


def test_decode_last_occurrence_wins() -> None:
    prompt = f"{markers.mark('age', 'young')} then {markers.mark('age', 'old')}"
    assert markers.decode(prompt) == {"age": "old"}


def test_roundtrip() -> None:
    m = markers.mark("religion", "atheist")
    assert markers.decode(m) == {"religion": "atheist"}


@pytest.mark.parametrize("bad", ["has space", "has=eq", "", "a-b"])
def test_mark_rejects_non_word(bad: str) -> None:
    with pytest.raises(ValueError, match="word-characters"):
        markers.mark(bad, "state")
