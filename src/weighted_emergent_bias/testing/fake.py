"""Controllable fake model clients with a known, dial-able bias.

The estimator can only be validated against ground truth, and ground truth does not exist
for a real model. :class:`FakeLLMClient` manufactures it: a latent categorical output
distribution over ``n_candidates`` symbols, shifted by a *known* magnitude along named axes
whenever a prompt carries that axis's marker (see :mod:`.markers`). A test dials in bias
``beta`` on an axis and asserts the recovered effect size tracks it.

The fake separates the two noise sources the design cares about:

- **Generative noise** — :meth:`generate` samples from the categorical, so repeated calls
  differ. This is the ``TaskMode.GENERATIVE`` noise floor.
- **Scoring noise** — :meth:`score_candidates` adds optional per-call logit jitter
  (``logit_noise``), modeling the context-sensitivity that gives ``TaskMode.CHOICE`` a
  non-degenerate null. With ``logit_noise=0`` scoring is deterministic (the R4 temp-0 case).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..clients import FloatArray
from . import markers


def _softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total: np.float64 = exp.sum()
    return exp / total


class FakeLLMClient:
    """A deterministic-given-seed stand-in for a user's :class:`~..clients.LLMClient`.

    Parameters
    ----------
    rng:
        Source of all randomness. Pass a seeded generator for reproducible tests.
    n_candidates:
        Size ``K`` of the latent output vocabulary ``c0..c{K-1}``.
    bias:
        Map of ``axis -> magnitude``. When a prompt marks ``axis`` in any non-reference
        state, the logits shift by ``magnitude`` along that axis's fixed random direction.
        An empty map is a perfectly unbiased model.
    base_logits:
        Optional length-``K`` baseline logits. Defaults to uniform.
    logit_noise:
        Standard deviation of per-call Gaussian logit jitter. ``0.0`` makes
        :meth:`score_candidates` deterministic.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        n_candidates: int = 4,
        bias: Mapping[str, float] | None = None,
        base_logits: Sequence[float] | None = None,
        logit_noise: float = 0.0,
    ) -> None:
        if n_candidates < 2:
            raise ValueError(f"n_candidates must be >= 2, got {n_candidates}")
        if logit_noise < 0.0:
            raise ValueError(f"logit_noise must be non-negative, got {logit_noise}")

        self._rng = rng
        self._k = n_candidates
        self._logit_noise = logit_noise
        self._bias: dict[str, float] = dict(bias or {})

        if base_logits is None:
            self._base = np.zeros(n_candidates, dtype=np.float64)
        else:
            base = np.asarray(base_logits, dtype=np.float64)
            if base.shape != (n_candidates,):
                raise ValueError(f"base_logits must have shape ({n_candidates},), got {base.shape}")
            self._base = base

        # A fixed unit direction in logit space per biased axis, drawn once so bias is
        # stable across calls and reproducible under the seed.
        self._directions: dict[str, FloatArray] = {}
        for axis in self._bias:
            vec = self._rng.standard_normal(n_candidates)
            norm = float(np.linalg.norm(vec))
            self._directions[axis] = vec / norm if norm > 0.0 else vec

    @property
    def candidates(self) -> list[str]:
        """The fake's canonical candidate set, ``["c0", ..., "c{K-1}"]``."""
        return [f"c{i}" for i in range(self._k)]

    def _index_of(self, candidate: str) -> int:
        if candidate.startswith("c"):
            suffix = candidate[1:]
            if suffix.isdigit():
                idx = int(suffix)
                if 0 <= idx < self._k:
                    return idx
        raise ValueError(f"unknown candidate {candidate!r}; expected one of {self.candidates}")

    def _logits(self, prompt: str, *, noisy: bool) -> FloatArray:
        logits = self._base.copy()
        state = markers.decode(prompt)
        for axis, value in state.items():
            if value == markers.REFERENCE_STATE:
                continue
            magnitude = self._bias.get(axis)
            if magnitude is not None:
                logits = logits + magnitude * self._directions[axis]
        if noisy and self._logit_noise > 0.0:
            logits = logits + self._rng.normal(0.0, self._logit_noise, size=self._k)
        return logits

    async def generate(self, prompt: str) -> str:
        probs = _softmax(self._logits(prompt, noisy=True))
        idx = int(self._rng.choice(self._k, p=probs))
        return f"c{idx}"

    async def score_candidates(self, prompt: str, candidates: Sequence[str]) -> FloatArray:
        logits = self._logits(prompt, noisy=True)
        return np.array([logits[self._index_of(c)] for c in candidates], dtype=np.float64)


class FakeEmbeddingClient:
    """A stand-in :class:`~..clients.EmbeddingClient`.

    Each distinct text is assigned a fixed random unit vector on first sight and memoized,
    so identical outputs embed identically and distinct outputs are near-orthogonal. That is
    all the ``GENERATIVE`` divergence path needs to tell "same answer" from "different answer."
    """

    def __init__(self, rng: np.random.Generator, *, dim: int = 16) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self._rng = rng
        self._dim = dim
        self._cache: dict[str, FloatArray] = {}

    def _vector_for(self, text: str) -> FloatArray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vec = self._rng.standard_normal(self._dim)
        norm = float(np.linalg.norm(vec))
        unit = vec / norm if norm > 0.0 else vec
        self._cache[text] = unit
        return unit

    async def embed(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float64)
        return np.vstack([self._vector_for(t) for t in texts])
