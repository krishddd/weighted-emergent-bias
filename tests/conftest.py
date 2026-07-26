"""Shared test fixtures.

Every stochastic test in this project must be reproducible. The ``rng`` fixture
is the single sanctioned source of randomness — tests that reach for
``numpy.random.default_rng`` directly or read the clock defeat the point.
"""

from __future__ import annotations

import numpy as np
import pytest

DEFAULT_SEED = 20260726


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh seeded numpy Generator, identical run-to-run."""
    return np.random.default_rng(DEFAULT_SEED)


@pytest.fixture
def seed() -> int:
    """The seed value itself, for reproducing failures by hand."""
    return DEFAULT_SEED
