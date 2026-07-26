"""Model-client protocols the user supplies.

The library never imports a vendor SDK. A user wires their own model behind these two
protocols; the reference agents (M4) and the probe (WP5) depend only on this surface.
Both protocols are async because probing runs the standard, counterfactual, and null
calls concurrently — sequential probing would triple the latency on a node's critical path.

Two methods, matching the two task modes:

- :meth:`LLMClient.score_candidates` powers ``TaskMode.CHOICE`` — the well-defined path,
  scoring a *shared* candidate set under two prompts so divergence is exact.
- :meth:`LLMClient.generate` powers ``TaskMode.GENERATIVE`` — free-form output that is then
  embedded via :class:`EmbeddingClient` and compared as a semantic distance.

A given deployment may only support one mode (many hosted chat APIs expose no logprobs, so
``score_candidates`` is unavailable); that limitation is discovered and recorded, never
papered over.

Monitor independence (WP5): a client used to *score* a node need not be the same client that
*produced* the node's output. The probe (WP5) will accept an optional independent monitor
client — ideally a different model family on a read-only channel — so the monitor does not
inherit the exact blind spots of the agent it watches. These protocols are that seam; nothing
here assumes one shared client.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@runtime_checkable
class LLMClient(Protocol):
    """A user-supplied language model. Implementations must be safe to call concurrently."""

    async def generate(self, prompt: str) -> str:
        """Sample one free-form completion at the node's production sampling settings.

        Stochastic: repeated calls on an identical prompt yield different text, and that
        variation *is* the sampling-noise floor the estimator calibrates against.
        """
        ...

    async def score_candidates(self, prompt: str, candidates: Sequence[str]) -> FloatArray:
        """Return an unnormalized log-score for each candidate continuation under ``prompt``.

        Shape ``(len(candidates),)``, aligned to ``candidates`` order. The scoring layer
        normalizes over the candidate set; keeping raw log-scores here avoids baking a
        normalization choice into the protocol. Raise :class:`NotImplementedError` if the
        underlying model exposes no candidate scoring.
        """
        ...


@runtime_checkable
class EmbeddingClient(Protocol):
    """A user-supplied text embedder, used for ``TaskMode.GENERATIVE`` divergence."""

    async def embed(self, texts: Sequence[str]) -> FloatArray:
        """Embed each text. Shape ``(len(texts), dim)``, one row per input, order-preserving."""
        ...
