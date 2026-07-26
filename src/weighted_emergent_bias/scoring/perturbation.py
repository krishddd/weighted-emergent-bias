"""Counterfactual perturbation: build the demographically-altered twin of an input.

``perturb`` takes a payload (a string, or a nested structure of them) and a set of axis
specifications, and returns one :class:`~..types.Perturbation` per (axis, kind) group: the
original payload paired with a twin in which only that axis's substitutions have been applied
to string leaves. Everything else — structure, keys, non-string values, untargeted words — is
held fixed, so any divergence the estimator later measures is attributable to the axis alone.

Two deliberate design choices:

- **No default axes ship.** A hardcoded list of demographic categories is itself an editorial
  position about which identities count, which is the wrong thing for a fairness library to
  bake in. Axes are a required, explicit argument; example sets live in ``docs/``. Calling
  ``perturb`` with no axes is an error, not a no-op.
- **Explicit and proxy substitutions produce separate perturbations.** An axis may carry both
  (e.g. ``he→she`` and a zip-code swap). They are never applied together, because "the model
  reacts to a stated attribute" and "the model reacts to a proxy for it" are different findings
  and the audit trail must be able to tell them apart.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types import Perturbation, PerturbationKind

# Emit groups in a stable order regardless of how substitutions were listed on the axis.
_KIND_ORDER = (PerturbationKind.EXPLICIT, PerturbationKind.PROXY)


@dataclass(frozen=True, slots=True)
class Substitution:
    """A single whole-word replacement, applied case-insensitively with case preservation.

    ``pattern`` is matched on word boundaries (so ``he`` does not fire inside ``the``), and the
    replacement inherits the matched token's capitalization (``He`` → ``She``, ``HE`` → ``SHE``).
    """

    pattern: str
    replacement: str
    kind: PerturbationKind = PerturbationKind.EXPLICIT

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("Substitution.pattern must be non-empty")


@dataclass(frozen=True, slots=True)
class AxisSpec:
    """A named perturbation axis and its substitutions (explicit and/or proxy)."""

    name: str
    substitutions: tuple[Substitution, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AxisSpec.name must be non-empty")
        if not self.substitutions:
            raise ValueError(f"AxisSpec {self.name!r} must define at least one substitution")


def _sub_one(text: str, sub: Substitution) -> str:
    def repl(match: re.Match[str]) -> str:
        matched = match.group(0)
        rep = sub.replacement
        if len(matched) > 1 and matched.isupper():
            return rep.upper()
        if matched[:1].isupper():
            return rep[:1].upper() + rep[1:]
        return rep

    return re.sub(rf"\b{re.escape(sub.pattern)}\b", repl, text, flags=re.IGNORECASE)


def _apply(text: str, subs: Sequence[Substitution]) -> str:
    for sub in subs:
        text = _sub_one(text, sub)
    return text


def _perturb_node(node: Any, subs: Sequence[Substitution]) -> Any:
    """Recurse a payload, substituting in string leaves only; all else passes through."""
    if isinstance(node, str):
        return _apply(node, subs)
    if isinstance(node, Mapping):
        return {key: _perturb_node(value, subs) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        rebuilt = [_perturb_node(item, subs) for item in node]
        return tuple(rebuilt) if isinstance(node, tuple) else rebuilt
    return node


def perturb(payload: Any, axes: Sequence[AxisSpec]) -> list[Perturbation]:
    """Build counterfactual twins of ``payload``, one per (axis, kind) group.

    Raises ``ValueError`` if ``axes`` is empty — there is intentionally no default axis set.
    """
    if not axes:
        raise ValueError(
            "perturb requires at least one axis; there is no default axis set (see module docs)"
        )

    results: list[Perturbation] = []
    for axis in axes:
        by_kind: dict[PerturbationKind, list[Substitution]] = {}
        for sub in axis.substitutions:
            by_kind.setdefault(sub.kind, []).append(sub)
        for kind in _KIND_ORDER:
            subs = by_kind.get(kind)
            if not subs:
                continue
            results.append(
                Perturbation(
                    axis=axis.name,
                    original=payload,
                    perturbed=_perturb_node(payload, subs),
                    kind=kind,
                )
            )
    return results
