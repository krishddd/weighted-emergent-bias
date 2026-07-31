"""LOOCProbe — orchestrate a node's standard, counterfactual, and null probes.

Given a node's input, the probe resamples the node's output `n_samples` times on the standard
input and on each perturbed (counterfactual) input, then hands each baseline/counterfactual pair
to :func:`compute_local_bias`. The baseline resamples *are* the sampling-noise floor the score is
calibrated against.

Two properties this module exists to guarantee:

- **Concurrency.** Every client call — baseline and all axes — is issued concurrently via
  ``asyncio.gather``. Probing sits on a node's critical path; running the calls sequentially would
  multiply latency by ``n_samples * (1 + n_axes)``. Wall-clock is bounded by the slowest single
  call, not their sum.
- **Partial-failure isolation.** If one axis's probes fail, that axis is recorded in
  ``ProbeResult.failures`` and the rest still produce scores — a partial scan is never reported as
  complete. A *baseline* failure is fatal (there is no noise floor without it) and raises.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np

from ..clients import EmbeddingClient, LLMClient
from ..types import (
    AxisFailure,
    AxisScore,
    FloatArray,
    ProbeResult,
    TaskMode,
)
from .divergence import softmax
from .noise import compute_local_bias
from .perturbation import AxisSpec, perturb


class ProbeError(RuntimeError):
    """Raised when the baseline probe fails — without it there is no noise floor to calibrate."""


class LOOCProbe:
    """Wraps one agent node for Leave-One-Out Calibration probing.

    Parameters
    ----------
    client:
        The model behind the node.
    task_mode:
        ``CHOICE`` (scores a shared candidate set — ``candidates`` required) or ``GENERATIVE``
        (embeds free-form output — ``embedder`` required).
    axes:
        Perturbation axes; at least one (there is no default set).
    n_samples:
        Resamples per input, per side. Must be >= 2 so the noise floor is estimable.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        task_mode: TaskMode,
        axes: Sequence[AxisSpec],
        rng: np.random.Generator,
        n_samples: int = 8,
        candidates: Sequence[str] | None = None,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        if not axes:
            raise ValueError("at least one axis is required; there is no default axis set")
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2 to estimate the noise floor, got {n_samples}")
        if task_mode is TaskMode.CHOICE and not candidates:
            raise ValueError("CHOICE mode requires a non-empty `candidates` set")
        if task_mode is TaskMode.GENERATIVE and embedder is None:
            raise ValueError("GENERATIVE mode requires an `embedder`")

        self._client = client
        self._task_mode = task_mode
        self._axes = tuple(axes)
        self._rng = rng
        self._n = n_samples
        self._candidates = tuple(candidates) if candidates else ()
        self._embedder = embedder

    async def _sample(self, prompt: str) -> FloatArray:
        if self._task_mode is TaskMode.CHOICE:
            logits = await self._client.score_candidates(prompt, self._candidates)
            return softmax(logits)
        text = await self._client.generate(prompt)
        assert self._embedder is not None  # guaranteed by __init__
        embedding = await self._embedder.embed([text])
        row: FloatArray = np.asarray(embedding[0], dtype=np.float64)
        return row

    async def _group(self, prompt: str) -> list[FloatArray]:
        return list(await asyncio.gather(*(self._sample(prompt) for _ in range(self._n))))

    async def _group_safe(self, prompt: str) -> list[FloatArray] | Exception:
        try:
            return await self._group(prompt)
        except Exception as exc:
            return exc

    async def run(self, payload: str) -> ProbeResult:
        """Probe ``payload`` across all axes concurrently and return per-axis scores."""
        perturbations = perturb(payload, self._axes)

        # Baseline first, then one group per perturbation — all gathered concurrently.
        groups = [self._group_safe(payload)]
        groups.extend(self._group_safe(p.perturbed) for p in perturbations)  # type: ignore[arg-type]
        results: list[list[FloatArray] | Exception] = list(await asyncio.gather(*groups))

        n_client_calls = self._n * (1 + len(perturbations))

        baseline = results[0]
        if isinstance(baseline, Exception):
            raise ProbeError("baseline probe failed; cannot estimate the noise floor") from baseline

        scores: list[AxisScore] = []
        failures: list[AxisFailure] = []
        for pert, group in zip(perturbations, results[1:], strict=True):
            if isinstance(group, Exception):
                failures.append(AxisFailure(axis=pert.axis, kind=pert.kind, error=repr(group)))
                continue
            score = compute_local_bias(
                baseline,
                group,
                task_mode=self._task_mode,
                rng=self._rng,
                axis=pert.axis,
            )
            scores.append(AxisScore(axis=pert.axis, kind=pert.kind, score=score))

        return ProbeResult(
            task_mode=self._task_mode,
            n_samples=self._n,
            n_client_calls=n_client_calls,
            scores=tuple(scores),
            failures=tuple(failures),
        )
