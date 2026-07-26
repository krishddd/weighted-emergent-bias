"""Wiring smoke test — proves the package imports and fixtures load.

Real tests land in Phase 1 WP1 onward.
"""

from __future__ import annotations

import numpy as np

import weighted_emergent_bias as web


def test_package_version_is_string() -> None:
    assert isinstance(web.__version__, str)
    assert web.__version__ != ""


def test_rng_is_reproducible(rng: np.random.Generator, seed: int) -> None:
    """The rng fixture must produce identical draws across runs."""
    draws = rng.standard_normal(5)
    expected = np.random.default_rng(seed).standard_normal(5)
    np.testing.assert_array_equal(draws, expected)
