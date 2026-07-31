"""A real-model :class:`~...clients.LLMClient` backed by the Anthropic API (optional).

This is what converts "demonstrated on the ground-truth fake" into "runnable against a real model".
It is an *optional* adapter -- the core library still has no vendor dependency, and nothing imports
this module unless you ask for it (``pip install weighted-emergent-bias[anthropic]``).

**Resolving Phase 1 spike S1.** PHASE-1 R1 left one question open: is ``TaskMode.CHOICE`` reachable
on a real hosted API? For the Anthropic Messages API the answer is **no via logprobs** -- it does
not expose token log-probabilities, so exact JSD over a shared support is not directly available.

The recoverable alternative is Monte Carlo: ask the model to pick from the candidate set, sample it
`n` times, and let the *empirical* distribution stand in for the analytic one. This composes exactly
with the existing estimator, because :func:`~...scoring.noise.compute_local_bias` already averages
each group -- so a one-hot draw per sample means the group mean **is** the estimated distribution,
and the permutation test runs unchanged. The cost is statistical: an empirical distribution from `n`
draws is noisier than a logprob vector, so CHOICE on this client needs a larger `n` than the fake
suggests. That tradeoff is real and is reported, not hidden.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from ..types import FloatArray

DEFAULT_MODEL = "claude-opus-5"
_ONE_HOT_LOGIT = 0.0
_SUPPRESSED_LOGIT = -20.0  # softmax(0 vs -20) is one-hot to ~1e-9


class AnthropicClient:
    """Adapts the Anthropic Messages API to the library's :class:`LLMClient` protocol.

    ``generate`` maps to a single message (the ``GENERATIVE`` path). ``score_candidates`` performs a
    forced choice over the candidate set and returns a near-one-hot log-score vector -- one Monte
    Carlo draw of the node's output distribution (the ``CHOICE`` path; see the module docstring).

    The client is constructed lazily so importing this module never requires the SDK or a key.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 512,
        system: str | None = None,
        client: object | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._client = client

    def _ensure_client(self) -> object:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError(
                    "the Anthropic adapter needs the SDK: "
                    'pip install "weighted-emergent-bias[anthropic]"'
                ) from exc
            self._client = AsyncAnthropic()
        return self._client

    async def _message(self, prompt: str, *, max_tokens: int | None = None) -> str:
        client = self._ensure_client()
        kwargs: dict[str, object] = {
            "model": self._model,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._system is not None:
            kwargs["system"] = self._system
        response = await client.messages.create(**kwargs)  # type: ignore[attr-defined]

        # A safety decline returns HTTP 200 with stop_reason "refusal" and possibly empty content;
        # surface it rather than indexing into content and raising an opaque IndexError.
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError("the model declined this request (stop_reason='refusal')")
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return str(block.text)
        return ""

    async def generate(self, prompt: str) -> str:
        """Sample one free-form completion (the ``GENERATIVE`` path)."""
        return await self._message(prompt)

    async def score_candidates(self, prompt: str, candidates: Sequence[str]) -> FloatArray:
        """One Monte Carlo draw over ``candidates``, as a near-one-hot log-score vector.

        The API exposes no logprobs, so this forces a single choice and encodes it as a one-hot.
        Averaged over the resamples the probe already performs, the group mean is the empirical
        output distribution -- which is what the estimator consumes.
        """
        if not candidates:
            raise ValueError("candidates must be non-empty")
        options = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
        choice_prompt = (
            f"{prompt}\n\nChoose exactly one option and reply with its number only.\n{options}"
        )
        reply = await self._message(choice_prompt, max_tokens=8)

        logits = np.full(len(candidates), _SUPPRESSED_LOGIT, dtype=np.float64)
        match = re.search(r"\d+", reply)
        index = int(match.group()) - 1 if match else -1
        if 0 <= index < len(candidates):
            logits[index] = _ONE_HOT_LOGIT
        else:
            # Unparseable reply: return a uniform vector rather than silently biasing toward
            # candidate 0. It contributes noise, not a fabricated preference.
            logits[:] = _ONE_HOT_LOGIT
        return logits
