"""Deterministic test doubles for the intervention layer (M4).

Ground-truth fakes so the panel / trust graph / MADERA can be exercised without a real model:
a skeptic that returns a canned verdict, and (for MADERA) fakes that diagnose, retrieve, and edit
toward a known unbiased target.
"""

from __future__ import annotations

import asyncio

from ..intervention.types import SkepticStance, SkepticVerdict
from ..types import Payload


class FakeSkeptic:
    """A ``SkepticAgent`` giving a fixed verdict, optionally delayed (for concurrency tests)."""

    def __init__(
        self,
        agent_type: str,
        stance: SkepticStance = SkepticStance.REVISE,
        *,
        confidence: float = 0.8,
        has_evidence: bool = True,
        revision: Payload | None = None,
        critique: str = "canned critique",
        delay: float = 0.0,
    ) -> None:
        self.agent_type = agent_type
        self._stance = stance
        self._confidence = confidence
        self._has_evidence = has_evidence
        self._revision = revision
        self._critique = critique
        self._delay = delay

    async def review(self, disputed: Payload, context: str = "") -> SkepticVerdict:
        if self._delay > 0.0:
            await asyncio.sleep(self._delay)
        return SkepticVerdict(
            agent_type=self.agent_type,
            stance=self._stance,
            critique=self._critique,
            confidence=self._confidence,
            has_evidence=self._has_evidence,
            proposed_payload=self._revision,
        )
