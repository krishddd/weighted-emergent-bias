"""The skeptic panel -- parallel adversarial reviewers (M4).

A conformity spiral is broken by fanning out several *distinct* skeptics that each attack the
disputed output from a different angle (DESIGN §6): an **Empirical Auditor** (demands evidence), a
**Devil's Advocate** (argues the opposite), and a **Diversity Champion** (surfaces under-represented
views). Requiring at least two distinct agents is a deliberate anti-SPOF rule (a panel of one, or
three clones of one, is not a panel).

Each skeptic is a thin wrapper over a user-injected :class:`~..clients.LLMClient` with an
overridable role prompt -- no vendor SDK. The panel only *collects* verdicts concurrently;
aggregating them into a decision (with trust weighting and pruning) is the trust graph's job.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from typing import Protocol

from ..clients import LLMClient
from ..types import Payload
from .types import SkepticStance, SkepticVerdict

_ROLE_PROMPTS: dict[str, str] = {
    "empirical_auditor": (
        "You are an empirical auditor. Demand concrete evidence for every claim and flag any "
        "unsupported or stereotyped assertion."
    ),
    "devils_advocate": (
        "You are a devil's advocate. Argue the strongest possible case against the output and "
        "expose assumptions it takes for granted."
    ),
    "diversity_champion": (
        "You are a diversity champion. Surface perspectives, groups, or interpretations the output "
        "under-represents or erases."
    ),
}


class SkepticAgent(Protocol):
    """A single adversarial reviewer. Implement to plug a custom skeptic into the panel."""

    agent_type: str

    async def review(self, disputed: Payload, context: str = "") -> SkepticVerdict: ...


_EVIDENCE_MARKER = "EVIDENCE:"
_URL_RE = re.compile(r"https?://\S+")


def _detect_evidence(text: str) -> bool:
    """Whether a reply carries citable evidence, from an *explicit* marker or a URL.

    Deliberately not a substring search for ``"evidence"``. That heuristic inverts on the single
    most common phrasing a skeptic uses -- "no evidence provided" sets the flag *true* -- and the
    ``empirical_auditor`` role prompt contains the word itself, which models routinely echo. Since
    ``has_evidence`` drives credibility, the overconfidence prune, and the trust weight, it needs a
    signal that cannot be produced by discussing evidence in the abstract. A skeptic must either
    emit an ``EVIDENCE:`` line (the prompt asks for one) or cite a URL.
    """
    return _EVIDENCE_MARKER in text.upper() or bool(_URL_RE.search(text))


def _parse_verdict(agent_type: str, text: str) -> SkepticVerdict:
    """Parse a reference skeptic's free-form reply into a structured verdict.

    Convention: first token is ``STANDS`` / ``REVISE`` / ``REJECT``; the rest is the critique, with
    an optional proposed revision after a ``---`` separator. Lenient (defaults to STANDS).
    """
    stripped = text.strip()
    head, _, rest = stripped.partition("\n")
    token = head.strip().split(" ", 1)[0].upper() if head.strip() else ""
    stance = {
        "STANDS": SkepticStance.STANDS,
        "REVISE": SkepticStance.REVISE,
        "REJECT": SkepticStance.REJECT,
    }.get(token, SkepticStance.STANDS)

    critique, _, revision = rest.partition("---")
    proposed: Payload | None = None
    if stance is not SkepticStance.STANDS and revision.strip():
        proposed = revision.strip()
    has_evidence = _detect_evidence(stripped)
    return SkepticVerdict(
        agent_type=agent_type,
        stance=stance,
        critique=critique.strip(),
        confidence=0.7,
        has_evidence=has_evidence,
        proposed_payload=proposed,
    )


class LLMSkeptic:
    """Reference :class:`SkepticAgent` over an injected client and an overridable role prompt."""

    def __init__(self, agent_type: str, client: LLMClient, *, prompt: str | None = None) -> None:
        if not agent_type:
            raise ValueError("agent_type must be non-empty")
        self.agent_type = agent_type
        self._client = client
        self._prompt = prompt if prompt is not None else _ROLE_PROMPTS.get(agent_type, "")

    async def review(self, disputed: Payload, context: str = "") -> SkepticVerdict:
        prompt = (
            f"{self._prompt}\n\nOutput under review:\n{disputed}\n{context}\n\n"
            "Respond with STANDS, REVISE, or REJECT on the first line, then your critique. "
            "If you can cite concrete support, add a line beginning 'EVIDENCE:' (a URL counts); "
            "omit it when you cannot. "
            "Then optionally '---' followed by a corrected version."
        )
        text = await self._client.generate(prompt)
        return _parse_verdict(self.agent_type, text)


def empirical_auditor(client: LLMClient, *, prompt: str | None = None) -> LLMSkeptic:
    return LLMSkeptic("empirical_auditor", client, prompt=prompt)


def devils_advocate(client: LLMClient, *, prompt: str | None = None) -> LLMSkeptic:
    return LLMSkeptic("devils_advocate", client, prompt=prompt)


def diversity_champion(client: LLMClient, *, prompt: str | None = None) -> LLMSkeptic:
    return LLMSkeptic("diversity_champion", client, prompt=prompt)


class SkepticPanel:
    """Runs a set of distinct skeptics concurrently and returns their raw verdicts.

    Requires **>= 2 distinct agent types** (anti-SPOF). Aggregation into a decision is the trust
    graph's responsibility -- this panel only gathers.
    """

    def __init__(self, agents: Sequence[SkepticAgent]) -> None:
        if len({a.agent_type for a in agents}) < 2:
            raise ValueError("a panel needs at least 2 distinct skeptic types (anti-SPOF)")
        self._agents = tuple(agents)

    @property
    def agents(self) -> tuple[SkepticAgent, ...]:
        return self._agents

    async def review(self, disputed: Payload, context: str = "") -> tuple[SkepticVerdict, ...]:
        """Collect every skeptic's verdict concurrently; a failing skeptic is dropped, not fatal.

        One flaky client would otherwise take the whole panel down with it and discard the verdicts
        that did arrive. Failures are omitted from the result, so the trust graph weighs only real
        verdicts -- and an empty return says plainly that no reviewer survived.
        """
        settled = await asyncio.gather(
            *(a.review(disputed, context) for a in self._agents), return_exceptions=True
        )
        return tuple(v for v in settled if isinstance(v, SkepticVerdict))
