"""A tiny prompt-marker convention, used only by the fakes.

Real perturbation (WP2) edits genuine demographic content in real text. The *fake* client
cannot read English, so for a self-consistent ground-truth stack we tag a prompt's latent
axis state with explicit markers of the form ``[[axis=state]]`` that the fake parses. This
lets a test dial in a known bias and check the estimator recovers it — the only place in
the project where ground truth exists.

This convention is deliberately confined to the ``testing`` subpackage; nothing in the
core library depends on it.
"""

from __future__ import annotations

import re

_MARKER_RE = re.compile(r"\[\[(\w+)=(\w+)\]\]")

REFERENCE_STATE = "ref"
"""The unperturbed baseline state for an axis. A marker in this state induces no bias shift."""


def mark(axis: str, state: str) -> str:
    """Render a single ``[[axis=state]]`` marker."""
    if not re.fullmatch(r"\w+", axis) or not re.fullmatch(r"\w+", state):
        raise ValueError(f"axis and state must be word-characters only: {axis!r}={state!r}")
    return f"[[{axis}={state}]]"


def decode(prompt: str) -> dict[str, str]:
    """Extract every ``axis -> state`` marker from a prompt.

    If an axis appears more than once, the last occurrence wins — matching how a real
    prompt's most-recent context would dominate.
    """
    return {axis: state for axis, state in _MARKER_RE.findall(prompt)}
