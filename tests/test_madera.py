"""Tests for the MADERA reasoning-repair pipeline."""

from __future__ import annotations

import pytest

from weighted_emergent_bias import MaderaEditor
from weighted_emergent_bias.testing import (
    DecayEditor,
    FakeDiagnoser,
    FakeRetriever,
    bias_field_scorer,
)


def _madera(decay: float, *, threshold: float = 0.1, max_rewrites: int = 3) -> MaderaEditor:
    return MaderaEditor(
        FakeDiagnoser(),
        FakeRetriever(),
        DecayEditor(decay),
        bias_field_scorer,
        threshold=threshold,
        max_rewrites=max_rewrites,
    )


class TestConvergence:
    async def test_iterative_rewrite_clears_bias(self) -> None:
        # bias 0.8 * 0.4 per rewrite: 0.32, 0.128, 0.0512 -> clears 0.1 by the 3rd rewrite.
        result = await _madera(decay=0.4).repair({"bias": 0.8, "text": "biased"})
        assert result.converged
        assert result.strategy == "madera"
        assert isinstance(result.repaired_payload, dict)
        assert result.repaired_payload["bias"] < 0.1

    async def test_already_clean_needs_no_rewrite(self) -> None:
        result = await _madera(decay=0.4).repair({"bias": 0.02})
        assert result.converged
        # No rewrite happened -> only the initial-score trail entry plus the converged marker.
        assert not any("rewrite" in line for line in result.trail)

    async def test_non_convergence_returns_best_effort_flagged(self) -> None:
        # decay 1.0 never reduces bias -> hits the cap, returns not converged.
        result = await _madera(decay=1.0, max_rewrites=2).repair({"bias": 0.8})
        assert not result.converged
        assert result.repaired_payload is not None
        assert sum("rewrite" in line for line in result.trail) == 2  # capped at max_rewrites

    async def test_trail_records_each_pass(self) -> None:
        result = await _madera(decay=0.4).repair({"bias": 0.8})
        assert result.trail[0].startswith("initial bias")
        assert result.trail[-1] == "converged"


class TestValidation:
    def test_max_rewrites_positive(self) -> None:
        with pytest.raises(ValueError, match="max_rewrites"):
            _madera(decay=0.4, max_rewrites=0)

    def test_threshold_non_negative(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            _madera(decay=0.4, threshold=-0.1)
