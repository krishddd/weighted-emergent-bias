"""Tests for audit reporting: JSON, self-contained HTML, and breach trace-back."""

from __future__ import annotations

from weighted_emergent_bias import (
    AuditKind,
    AuditTrail,
    BiasScore,
    BreakerAction,
    BreakerDecision,
    BreakerState,
    DivergenceMethod,
    TaskMode,
    to_html,
    to_json,
    trace_breach,
)


def _score(axis: str, effect: float, *, significant: bool) -> BiasScore:
    return BiasScore(
        effect_size=effect,
        ci_low=effect - 1,
        ci_high=effect + 1,
        p_value=0.001 if significant else 0.5,
        method=DivergenceMethod.JENSEN_SHANNON,
        task_mode=TaskMode.CHOICE,
        n_samples=8,
        raw_divergence=0.5,
        null_mean=0.1,
        null_std=0.1,
        axis=axis,
    )


def _breached_trail() -> AuditTrail:
    trail = AuditTrail()
    trail.record_score(_score("age", 1.5, significant=True), node="worker")
    trail.record_score(_score("gender", 6.0, significant=True), node="router")  # top contributor
    # A few supersteps of B_net rising toward the threshold, then the halt.
    trail.record_decision(
        BreakerDecision(BreakerState.NORMAL, BreakerAction.PROCEED, 0.0, 0.2, "ok")
    )
    trail.record_decision(
        BreakerDecision(BreakerState.WARNING, BreakerAction.PROCEED, 0.0, 0.4, "drift")
    )
    trail.record_decision(
        BreakerDecision(BreakerState.INTERVENTION, BreakerAction.REROUTE, 0.7, 0.6, "fast >= tau"),
        node="router",
    )
    return trail


class TestTraceBreach:
    def test_points_at_top_significant_score(self) -> None:
        breach = trace_breach(_breached_trail())
        assert breach is not None
        assert breach["origin_node"] == "router"
        assert breach["origin_axis"] == "gender"  # highest effect size before the halt
        assert breach["origin_effect_size"] == 6.0

    def test_none_when_no_halt(self) -> None:
        trail = AuditTrail()
        trail.record_score(_score("gender", 6.0, significant=True), node="a")
        trail.record_decision(
            BreakerDecision(BreakerState.NORMAL, BreakerAction.PROCEED, 0.0, 0.1, "ok")
        )
        assert trace_breach(trail) is None


class TestToJson:
    def test_summary_and_breach(self) -> None:
        doc = to_json(_breached_trail())
        assert doc["n_events"] == 5
        assert doc["by_kind"]["score"] == 2
        assert doc["by_kind"]["breaker"] == 3
        assert doc["breach"]["origin_axis"] == "gender"
        assert len(doc["b_net_trajectory"]) == 3


class TestToHtml:
    def test_is_self_contained(self) -> None:
        # No external assets -> works offline and under a strict CSP.
        report = to_html(_breached_trail())
        assert "http://" not in report
        assert "https://" not in report
        assert "src=" not in report

    def test_contains_breach_and_nodes(self) -> None:
        report = to_html(_breached_trail())
        assert "Breach traced to" in report
        assert "router" in report
        assert "<svg" in report  # trajectory sparkline rendered

    def test_escapes_content(self) -> None:
        trail = AuditTrail()
        trail.record(AuditKind.NOTE, node="<script>", detail={"x": "<b>"})
        report = to_html(trail)
        assert "<script>" not in report  # escaped
        assert "&lt;script&gt;" in report
