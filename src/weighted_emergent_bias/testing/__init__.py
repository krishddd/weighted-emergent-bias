"""Test doubles and ground-truth fixtures.

Importable by users writing their own tests, not part of the runtime path. The fakes here
are the only place in the project where the "true" bias of a node is known, which makes them
the reference against which the estimator is calibrated (Phase 1 WP7).
"""

from __future__ import annotations

from .fake import FakeEmbeddingClient, FakeLLMClient

__all__ = ["FakeEmbeddingClient", "FakeLLMClient"]
