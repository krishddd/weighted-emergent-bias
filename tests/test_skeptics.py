"""Tests for the skeptic panel."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np
import pytest

from weighted_emergent_bias import SkepticPanel, SkepticStance, SkepticVerdict
from weighted_emergent_bias.clients import FloatArray
from weighted_emergent_bias.intervention.skeptics import (
    devils_advocate,
    empirical_auditor,
)
from weighted_emergent_bias.testing import FakeLLMClient, FakeSkeptic


class TestPanelConstruction:
    def test_requires_two_distinct_types(self) -> None:
        with pytest.raises(ValueError, match="at least 2 distinct"):
            SkepticPanel([FakeSkeptic("auditor")])

    def test_rejects_clones_of_one_type(self) -> None:
        with pytest.raises(ValueError, match="at least 2 distinct"):
            SkepticPanel([FakeSkeptic("auditor"), FakeSkeptic("auditor")])

    def test_two_distinct_ok(self) -> None:
        panel = SkepticPanel([FakeSkeptic("auditor"), FakeSkeptic("advocate")])
        assert len(panel.agents) == 2


class TestPanelReview:
    async def test_collects_all_verdicts(self) -> None:
        panel = SkepticPanel(
            [
                FakeSkeptic("auditor", SkepticStance.REVISE),
                FakeSkeptic("advocate", SkepticStance.STANDS),
                FakeSkeptic("diversity", SkepticStance.REJECT),
            ]
        )
        verdicts = await panel.review({"answer": "biased"})
        assert {v.agent_type for v in verdicts} == {"auditor", "advocate", "diversity"}
        assert {v.stance for v in verdicts} == {
            SkepticStance.REVISE,
            SkepticStance.STANDS,
            SkepticStance.REJECT,
        }

    async def test_runs_concurrently(self) -> None:
        # Deterministic concurrency check: all three reviews are in flight at once.
        peak = 0
        in_flight = 0

        class _Tracker:
            def __init__(self, agent_type: str) -> None:
                self.agent_type = agent_type

            async def review(self, disputed: object, context: str = "") -> SkepticVerdict:
                nonlocal peak, in_flight
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1
                return SkepticVerdict(self.agent_type, SkepticStance.STANDS, "", 0.5)

        panel = SkepticPanel([_Tracker("a"), _Tracker("b"), _Tracker("c")])
        await panel.review({"x": 1})
        assert peak == 3


class TestLLMSkeptic:
    async def test_parses_revise_with_revision(self) -> None:
        class _Client:
            async def generate(self, prompt: str) -> str:
                return (
                    "REVISE\nThis stereotypes the applicant; here is evidence.\n---\nNeutral text."
                )

            async def score_candidates(self, prompt: str, candidates: Sequence[str]) -> FloatArray:
                raise NotImplementedError

        skeptic = empirical_auditor(_Client())
        verdict = await skeptic.review({"answer": "biased"})
        assert verdict.stance is SkepticStance.REVISE
        assert verdict.proposed_payload == "Neutral text."
        assert verdict.has_evidence  # "evidence" appeared in the reply

    async def test_defaults_to_stands_on_unparseable(self) -> None:
        class _Client:
            async def generate(self, prompt: str) -> str:
                return "hmm, seems fine to me"

            async def score_candidates(self, prompt: str, candidates: Sequence[str]) -> FloatArray:
                raise NotImplementedError

        verdict = await devils_advocate(_Client()).review({"answer": "ok"})
        assert verdict.stance is SkepticStance.STANDS

    async def test_reference_agents_have_distinct_types(self) -> None:
        client = FakeLLMClient(np.random.default_rng(1))
        assert empirical_auditor(client).agent_type != devils_advocate(client).agent_type
