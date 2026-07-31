"""MADERA-style reasoning repair (M4).

For an entrenched *parametric* bias, debate does not help -- it is baked into the model. MADERA
instead does surgery on the reasoning chain (DESIGN §6): **diagnose** the biased logical leap,
**retrieve** external counter-evidence, and **rewrite** the reasoning, iterating until the re-scored
bias clears. Each phase is an injected callable behind a protocol -- the retriever especially is
user-supplied (no built-in web access) -- and iteration is bounded by ``max_rewrites`` so it cannot
loop forever, mirroring M3's ``max_retries``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from ..types import Payload
from .types import InterventionResult

# Re-scores a payload's bias magnitude (e.g. an M1 probe wrapped in node_magnitude).
BiasScorer = Callable[[Payload], Awaitable[float]]


class Diagnoser(Protocol):
    async def diagnose(self, payload: Payload, context: str = "") -> str: ...


class Retriever(Protocol):
    async def retrieve(self, query: str) -> Sequence[str]: ...


class Editor(Protocol):
    async def rewrite(
        self, payload: Payload, diagnosis: str, evidence: Sequence[str]
    ) -> Payload: ...


class MaderaEditor:
    """Runs diagnose -> retrieve -> rewrite, re-scoring each pass, until the bias clears or the cap.

    ``scorer`` returns a bias magnitude; the loop stops once it drops below ``threshold``. If the
    cap is reached first, the best attempt is returned flagged ``converged=False``.
    """

    def __init__(
        self,
        diagnoser: Diagnoser,
        retriever: Retriever,
        editor: Editor,
        scorer: BiasScorer,
        *,
        threshold: float = 0.1,
        max_rewrites: int = 3,
    ) -> None:
        if max_rewrites < 1:
            raise ValueError(f"max_rewrites must be >= 1, got {max_rewrites}")
        if threshold < 0.0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        self._diagnoser = diagnoser
        self._retriever = retriever
        self._editor = editor
        self._scorer = scorer
        self._threshold = threshold
        self._max_rewrites = max_rewrites

    async def repair(self, payload: Payload, context: str = "") -> InterventionResult:
        """Iteratively repair ``payload`` until its re-scored bias clears ``threshold`` or the cap.

        The corrected payload is returned with ``converged`` telling M3 whether the repair worked.
        """
        current = payload
        score = await self._scorer(current)
        trail = [f"initial bias {score:.3f}"]

        rewrites = 0
        while score >= self._threshold and rewrites < self._max_rewrites:
            diagnosis = await self._diagnoser.diagnose(current, context)
            evidence = await self._retriever.retrieve(diagnosis)
            current = await self._editor.rewrite(current, diagnosis, evidence)
            score = await self._scorer(current)
            rewrites += 1
            trail.append(
                f"rewrite {rewrites}: {diagnosis!r} + {len(evidence)} evidence -> {score:.3f}"
            )

        converged = score < self._threshold
        trail.append("converged" if converged else "reached cap without converging")
        return InterventionResult(
            strategy="madera",
            repaired_payload=current,
            converged=converged,
            trail=tuple(trail),
        )
