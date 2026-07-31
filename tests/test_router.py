"""Tests for the intervention routing seam."""

from __future__ import annotations

import pytest

from weighted_emergent_bias import InterventionContext, route_intervention


class TestInterventionContext:
    def test_breadth(self) -> None:
        assert InterventionContext(2, 4, 0.5).breadth == pytest.approx(0.5)

    def test_empty_graph_breadth_zero(self) -> None:
        assert InterventionContext(0, 0, 0.0).breadth == 0.0

    def test_rejects_biased_exceeding_total(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            InterventionContext(5, 3, 0.5)


class TestRouteIntervention:
    def test_not_halted_is_none(self) -> None:
        ctx = InterventionContext(3, 4, 0.9, halted=False)
        assert route_intervention(ctx) == "none"

    def test_no_bias_is_none(self) -> None:
        ctx = InterventionContext(0, 4, 0.0)
        assert route_intervention(ctx) == "none"

    def test_broad_contamination_routes_to_skeptics(self) -> None:
        # 3 of 4 nodes biased -> conformity spiral -> skeptics.
        ctx = InterventionContext(3, 4, 0.6)
        assert route_intervention(ctx) == "skeptics"

    def test_concentrated_bias_routes_to_madera(self) -> None:
        # 1 of 5 nodes biased -> entrenched parametric -> madera.
        ctx = InterventionContext(1, 5, 0.9)
        assert route_intervention(ctx) == "madera"

    def test_threshold_is_tunable(self) -> None:
        ctx = InterventionContext(1, 4, 0.5)  # breadth 0.25
        assert route_intervention(ctx, breadth_threshold=0.5) == "madera"
        assert route_intervention(ctx, breadth_threshold=0.2) == "skeptics"

    def test_bad_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="breadth_threshold"):
            route_intervention(InterventionContext(1, 4, 0.5), breadth_threshold=0.0)
