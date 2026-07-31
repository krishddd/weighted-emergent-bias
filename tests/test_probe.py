"""Tests for LOOCProbe orchestration.

Three properties matter here: the probe issues exactly the predicted number of client calls,
it issues them concurrently (wall-clock bounded by the slowest call, not their sum), and a
single axis failing degrades to a recorded partial result rather than voiding the whole probe.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np
import pytest

from weighted_emergent_bias import (
    AxisSpec,
    LOOCProbe,
    ProbeError,
    Substitution,
    TaskMode,
)
from weighted_emergent_bias.clients import FloatArray
from weighted_emergent_bias.testing import FakeLLMClient, markers

SEED = 20260731

GENDER = AxisSpec("gender", (Substitution("he", "she"), Substitution("his", "her")))
RACE = AxisSpec("race", (Substitution("him", "her"),))

# The fake keys its bias off the [[axis=state]] marker, so to drive *detectable* bias through
# the real perturb() path we substitute the marker's state token (ref -> alt).
GENDER_MARKED = AxisSpec("gender", (Substitution(markers.REFERENCE_STATE, "alt"),))
MARKED_PAYLOAD = f"{markers.mark('gender', markers.REFERENCE_STATE)} the applicant"


def _fake(**kwargs: object) -> FakeLLMClient:
    return FakeLLMClient(np.random.default_rng(SEED), n_candidates=4, **kwargs)  # type: ignore[arg-type]


class _CountingClient:
    """Wraps a client and counts score_candidates / generate calls."""

    def __init__(self, inner: FakeLLMClient) -> None:
        self._inner = inner
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        return await self._inner.generate(prompt)

    async def score_candidates(self, prompt: str, candidates: Sequence[str]) -> FloatArray:
        self.calls += 1
        return await self._inner.score_candidates(prompt, candidates)


class _ConcurrencyTracker:
    """Records the peak number of in-flight calls. Deterministic (not timing-based): if the probe
    gathers its calls concurrently, all are in flight at once; if sequential, the peak is 1."""

    def __init__(self, inner: FakeLLMClient, delay: float = 0.01) -> None:
        self._inner = inner
        self._delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    async def _enter(self) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(self._delay)  # force overlap: every call parks here together

    async def generate(self, prompt: str) -> str:
        await self._enter()
        try:
            return await self._inner.generate(prompt)
        finally:
            self.in_flight -= 1

    async def score_candidates(self, prompt: str, candidates: Sequence[str]) -> FloatArray:
        await self._enter()
        try:
            return await self._inner.score_candidates(prompt, candidates)
        finally:
            self.in_flight -= 1


class _FlakyClient:
    """Raises on any prompt containing a given marker substring; delegates otherwise."""

    def __init__(self, inner: FakeLLMClient, fail_on: str) -> None:
        self._inner = inner
        self._fail_on = fail_on

    async def generate(self, prompt: str) -> str:
        return await self._inner.generate(prompt)

    async def score_candidates(self, prompt: str, candidates: Sequence[str]) -> FloatArray:
        if self._fail_on in prompt:
            raise RuntimeError(f"simulated failure on {self._fail_on}")
        return await self._inner.score_candidates(prompt, candidates)


class TestValidation:
    def test_empty_axes_rejected(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="at least one axis"):
            LOOCProbe(_fake(), task_mode=TaskMode.CHOICE, axes=[], rng=rng, candidates=["c0", "c1"])

    def test_choice_needs_candidates(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="candidates"):
            LOOCProbe(_fake(), task_mode=TaskMode.CHOICE, axes=[GENDER], rng=rng)

    def test_generative_needs_embedder(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="embedder"):
            LOOCProbe(_fake(), task_mode=TaskMode.GENERATIVE, axes=[GENDER], rng=rng)

    def test_too_few_samples_rejected(self, rng: np.random.Generator) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            LOOCProbe(
                _fake(),
                task_mode=TaskMode.CHOICE,
                axes=[GENDER],
                rng=rng,
                candidates=["c0"],
                n_samples=1,
            )


class TestChoiceProbe:
    async def test_scores_each_axis(self, rng: np.random.Generator) -> None:
        client = _fake(logit_noise=0.5)
        probe = LOOCProbe(
            client,
            task_mode=TaskMode.CHOICE,
            axes=[GENDER, RACE],
            rng=rng,
            candidates=client.candidates,
            n_samples=8,
        )
        result = await probe.run("he gave his report to him")
        assert result.ok
        assert {s.axis for s in result.scores} == {"gender", "race"}

    async def test_detects_injected_bias(self, rng: np.random.Generator) -> None:
        # Marker-driven bias, driven through the real perturb() path.
        client = _fake(bias={"gender": 5.0}, logit_noise=0.5)
        probe = LOOCProbe(
            client,
            task_mode=TaskMode.CHOICE,
            axes=[GENDER_MARKED],
            rng=rng,
            candidates=client.candidates,
            n_samples=10,
        )
        result = await probe.run(MARKED_PAYLOAD)
        (scored,) = result.scores
        assert scored.axis == "gender"
        assert scored.score.effect_size > 3.0
        assert scored.score.p_value < 0.05

    async def test_exact_client_call_count(self, rng: np.random.Generator) -> None:
        counting = _CountingClient(_fake(logit_noise=0.3))
        n = 5
        probe = LOOCProbe(
            counting,
            task_mode=TaskMode.CHOICE,
            axes=[GENDER, RACE],
            rng=rng,
            candidates=["c0", "c1", "c2", "c3"],
            n_samples=n,
        )
        result = await probe.run("he and him")
        # 1 baseline group + 2 axis groups (each axis has one explicit perturbation) x n samples.
        assert counting.calls == n * 3
        assert result.n_client_calls == n * 3


class TestConcurrency:
    async def test_all_calls_are_in_flight_at_once(self, rng: np.random.Generator) -> None:
        # Deterministic concurrency check: with n=6 samples over 3 groups (baseline + 2 axes),
        # a concurrent probe has all 18 calls in flight simultaneously; a sequential one peaks at 1.
        n = 6
        client = _ConcurrencyTracker(_fake(logit_noise=0.3))
        probe = LOOCProbe(
            client,
            task_mode=TaskMode.CHOICE,
            axes=[GENDER, RACE],
            rng=rng,
            candidates=["c0", "c1", "c2", "c3"],
            n_samples=n,
        )
        await probe.run("he and him")
        assert client.max_in_flight == n * 3


class TestFailureIsolation:
    async def test_one_axis_failure_is_isolated(self, rng: np.random.Generator) -> None:
        # Fail only on the race axis's perturbed prompt (contains "her" from him->her),
        # by targeting the marker we inject. Use a marker-based fail trigger instead.
        client = _FlakyClient(_fake(logit_noise=0.3), fail_on="ZZZFAIL")
        race_marked = AxisSpec("race", (Substitution("token", "ZZZFAIL"),))
        probe = LOOCProbe(
            client,
            task_mode=TaskMode.CHOICE,
            axes=[GENDER, race_marked],
            rng=rng,
            candidates=["c0", "c1", "c2", "c3"],
            n_samples=4,
        )
        result = await probe.run("he has his token")
        assert not result.ok
        assert {s.axis for s in result.scores} == {"gender"}
        assert [f.axis for f in result.failures] == ["race"]
        assert "simulated failure" in result.failures[0].error

    async def test_baseline_failure_is_fatal(self, rng: np.random.Generator) -> None:
        client = _FlakyClient(_fake(logit_noise=0.3), fail_on="baseline")
        probe = LOOCProbe(
            client,
            task_mode=TaskMode.CHOICE,
            axes=[GENDER],
            rng=rng,
            candidates=["c0", "c1", "c2", "c3"],
            n_samples=4,
        )
        with pytest.raises(ProbeError, match="baseline probe failed"):
            await probe.run("baseline: he ran")


class TestGenerativeProbe:
    async def test_generative_mode_produces_scores(self, rng: np.random.Generator) -> None:
        from weighted_emergent_bias.testing import FakeEmbeddingClient

        client = _fake(bias={"gender": 5.0}, logit_noise=0.4)
        embedder = FakeEmbeddingClient(np.random.default_rng(SEED), dim=8)
        probe = LOOCProbe(
            client,
            task_mode=TaskMode.GENERATIVE,
            axes=[GENDER_MARKED],
            rng=rng,
            embedder=embedder,
            n_samples=6,
        )
        result = await probe.run(MARKED_PAYLOAD)
        assert result.task_mode is TaskMode.GENERATIVE
        assert len(result.scores) == 1
        assert result.scores[0].axis == "gender"
