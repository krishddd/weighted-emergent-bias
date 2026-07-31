"""Tests for the LangGraph Send-based skeptic fan-out (behind importorskip)."""

from __future__ import annotations

import pytest

from weighted_emergent_bias import PanelDecision, SkepticStance, TrustGraph
from weighted_emergent_bias.testing import FakeSkeptic


class TestSkepticFanout:
    async def test_fanout_runs_all_skeptics_and_aggregates(self) -> None:
        pytest.importorskip("langgraph")
        from weighted_emergent_bias.integrations.langgraph import build_skeptic_graph

        agents = [
            FakeSkeptic("auditor", SkepticStance.REVISE, revision={"bias": 0.0}),
            FakeSkeptic("advocate", SkepticStance.REVISE, revision={"bias": 0.0}),
            FakeSkeptic("diversity", SkepticStance.STANDS),
        ]
        compiled = build_skeptic_graph(agents, TrustGraph(weighting="trust"))
        result = await compiled.ainvoke({"disputed": {"bias": 0.8}})

        # All three verdicts were collected in parallel and aggregated.
        assert len(result["verdicts"]) == 3
        panel = result["result"]
        assert panel.decision is PanelDecision.REVISED
        assert panel.corrected_payload == {"bias": 0.0}
