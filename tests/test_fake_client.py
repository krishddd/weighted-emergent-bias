"""Tests for the ground-truth fake clients.

These pin the fake's contract: an unbiased fake is genuinely indistinguishable from noise,
and a biased fake shifts its output by an amount that tracks the dialed-in magnitude. Every
later calibration test leans on both properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from weighted_emergent_bias.clients import EmbeddingClient, LLMClient
from weighted_emergent_bias.testing import FakeEmbeddingClient, FakeLLMClient
from weighted_emergent_bias.testing import markers as mk

SEED = 20260726


def _client(**kwargs: object) -> FakeLLMClient:
    return FakeLLMClient(np.random.default_rng(SEED), **kwargs)  # type: ignore[arg-type]


async def _full_logits(client: FakeLLMClient, prompt: str) -> np.ndarray:
    return await client.score_candidates(prompt, client.candidates)


class TestProtocolConformance:
    def test_llm_client_conforms(self) -> None:
        assert isinstance(_client(), LLMClient)

    def test_embedding_client_conforms(self) -> None:
        assert isinstance(FakeEmbeddingClient(np.random.default_rng(SEED)), EmbeddingClient)


class TestConstruction:
    def test_candidates_list(self) -> None:
        assert _client(n_candidates=3).candidates == ["c0", "c1", "c2"]

    @pytest.mark.parametrize("bad_k", [0, 1, -2])
    def test_rejects_too_few_candidates(self, bad_k: int) -> None:
        with pytest.raises(ValueError, match="n_candidates"):
            _client(n_candidates=bad_k)

    def test_rejects_negative_noise(self) -> None:
        with pytest.raises(ValueError, match="logit_noise"):
            _client(logit_noise=-1.0)

    def test_rejects_mismatched_base_logits(self) -> None:
        with pytest.raises(ValueError, match="base_logits"):
            _client(n_candidates=4, base_logits=[0.0, 1.0])


class TestBiasKnob:
    async def test_unbiased_noiseless_is_flat(self) -> None:
        """An unbiased fake gives identical scores for baseline and counterfactual."""
        client = _client(bias={})
        base = await _full_logits(client, mk.mark("gender", mk.REFERENCE_STATE))
        cf = await _full_logits(client, mk.mark("gender", "alt"))
        np.testing.assert_allclose(base, cf)

    async def test_reference_state_induces_no_shift(self) -> None:
        client = _client(bias={"gender": 2.0})
        base = await _full_logits(client, "no marker")
        ref = await _full_logits(client, mk.mark("gender", mk.REFERENCE_STATE))
        np.testing.assert_allclose(base, ref)

    async def test_bias_shifts_counterfactual(self) -> None:
        client = _client(bias={"gender": 1.5})
        base = await _full_logits(client, "no marker")
        cf = await _full_logits(client, mk.mark("gender", "alt"))
        assert not np.allclose(base, cf)

    async def test_shift_magnitude_tracks_beta(self) -> None:
        """The counterfactual logit shift equals beta (direction is a unit vector),
        so the recovered magnitude is monotonic in the injected bias."""
        distances: list[float] = []
        for beta in (0.5, 1.0, 2.0, 4.0):
            client = _client(bias={"gender": beta})
            base = await _full_logits(client, "no marker")
            cf = await _full_logits(client, mk.mark("gender", "alt"))
            distances.append(float(np.linalg.norm(cf - base)))
            assert distances[-1] == pytest.approx(beta, abs=1e-9)
        assert distances == sorted(distances)

    async def test_untargeted_axis_ignored(self) -> None:
        client = _client(bias={"gender": 2.0})
        base = await _full_logits(client, "no marker")
        other = await _full_logits(client, mk.mark("race", "alt"))
        np.testing.assert_allclose(base, other)


class TestNoiseFloor:
    async def test_noiseless_scoring_is_deterministic(self) -> None:
        client = _client(bias={}, logit_noise=0.0)
        a = await _full_logits(client, "p")
        b = await _full_logits(client, "p")
        np.testing.assert_allclose(a, b)

    async def test_noisy_scoring_varies(self) -> None:
        """logit_noise > 0 gives a non-degenerate null: same prompt, different scores."""
        client = _client(bias={}, logit_noise=0.5)
        a = await _full_logits(client, "p")
        b = await _full_logits(client, "p")
        assert not np.allclose(a, b)


class TestGenerate:
    async def test_reproducible_under_seed(self) -> None:
        c1, c2 = _client(bias={"gender": 1.0}), _client(bias={"gender": 1.0})
        out1 = [await c1.generate("no marker") for _ in range(10)]
        out2 = [await c2.generate("no marker") for _ in range(10)]
        assert out1 == out2

    async def test_returns_valid_candidate(self) -> None:
        client = _client(n_candidates=4)
        assert await client.generate("p") in client.candidates

    async def test_bias_shifts_sampling_distribution(self) -> None:
        """Strong bias measurably moves the empirical output distribution."""
        n, beta = 3000, 5.0
        client = _client(n_candidates=4, bias={"gender": beta})
        base = [await client.generate("no marker") for _ in range(n)]
        cf = [await client.generate(mk.mark("gender", "alt")) for _ in range(n)]

        cands = client.candidates
        p = np.array([base.count(c) for c in cands], dtype=float) / n
        q = np.array([cf.count(c) for c in cands], dtype=float) / n
        total_variation = 0.5 * float(np.abs(p - q).sum())
        assert total_variation > 0.1


class TestScoreCandidatesValidation:
    async def test_unknown_candidate_rejected(self) -> None:
        client = _client(n_candidates=3)
        with pytest.raises(ValueError, match="unknown candidate"):
            await client.score_candidates("p", ["c9"])

    async def test_respects_candidate_order(self) -> None:
        client = _client(n_candidates=3, bias={"gender": 2.0})
        forward = await client.score_candidates("no marker", ["c0", "c1", "c2"])
        reverse = await client.score_candidates("no marker", ["c2", "c1", "c0"])
        np.testing.assert_allclose(forward, reverse[::-1])


class TestFakeEmbedding:
    async def test_same_text_same_vector(self) -> None:
        client = FakeEmbeddingClient(np.random.default_rng(SEED), dim=8)
        out = await client.embed(["hello", "hello"])
        np.testing.assert_allclose(out[0], out[1])

    async def test_distinct_texts_differ(self) -> None:
        client = FakeEmbeddingClient(np.random.default_rng(SEED), dim=8)
        out = await client.embed(["hello", "world"])
        assert not np.allclose(out[0], out[1])

    async def test_unit_norm(self) -> None:
        client = FakeEmbeddingClient(np.random.default_rng(SEED), dim=8)
        out = await client.embed(["x"])
        assert float(np.linalg.norm(out[0])) == pytest.approx(1.0)

    async def test_empty_returns_shaped_array(self) -> None:
        client = FakeEmbeddingClient(np.random.default_rng(SEED), dim=8)
        out = await client.embed([])
        assert out.shape == (0, 8)

    def test_rejects_bad_dim(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            FakeEmbeddingClient(np.random.default_rng(SEED), dim=0)
