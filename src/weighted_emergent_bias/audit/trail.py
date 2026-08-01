"""The append-only causal audit trail (M5).

Every decision the system makes -- a per-node score, a breaker action, an intervention outcome --
is recorded here as an ordered, immutable event. This is the single source of truth the SARIF export
and the run report read from; it observes what M1-M4 produce and never recomputes it.

Two discipline choices: events carry a monotonic **sequence index** (not wall-clock) so a trail is
reproducible and diffable, with wall-clock time *optional and injected* (a clock callable), same as
the seeded-RNG rule; and the trail is **append-only** -- no public way to mutate or delete a past
event, so the record cannot be quietly rewritten.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from ..intervention.types import PanelResult
from ..types import BiasScore, BreakerDecision, NodeId


class AuditKind(str, Enum):
    """The kind of event recorded. Extendable; these are the ones M1-M4 emit today."""

    SCORE = "score"
    BREAKER = "breaker"
    INTERVENTION = "intervention"
    ROUTING = "routing"
    NOTE = "note"


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """One immutable entry in the trail."""

    seq: int
    kind: AuditKind
    node: NodeId | None
    detail: Mapping[str, Any]
    time: float | None = None


class AuditTrail:
    """An append-only, ordered log of audit events.

    ``clock`` optionally stamps wall-clock time on each event; omit it (the default) for fully
    reproducible trails keyed only on the sequence index.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._events: list[AuditEvent] = []
        self._clock = clock

    def record(
        self,
        kind: AuditKind,
        *,
        node: NodeId | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        """Append an event and return it. The only way to grow the trail.

        ``detail`` is deep-copied behind a read-only view. A shallow ``dict()`` copy left the record
        rewritable two ways -- ``trail.events[i].detail[k] = ...`` straight through the frozen
        dataclass, and through any nested object the caller still held a reference to -- which is
        not the append-only guarantee this module exists to provide.
        """
        event = AuditEvent(
            seq=len(self._events),
            kind=kind,
            node=node,
            detail=MappingProxyType(deepcopy(dict(detail or {}))),
            time=self._clock() if self._clock is not None else None,
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        """An immutable snapshot of the trail in order."""
        return tuple(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def of_kind(self, kind: AuditKind) -> tuple[AuditEvent, ...]:
        return tuple(e for e in self._events if e.kind is kind)

    # --- convenience recorders for the M1-M4 value types -------------------------------------

    def record_score(self, score: BiasScore, *, node: NodeId) -> AuditEvent:
        return self.record(
            AuditKind.SCORE,
            node=node,
            detail={
                "axis": score.axis,
                "effect_size": score.effect_size,
                "p_value": score.p_value,
                "method": score.method.value,
                "raw_divergence": score.raw_divergence,
                "null_mean": score.null_mean,
                "significant": score.is_significant(),
            },
        )

    def record_decision(
        self, decision: BreakerDecision, *, node: NodeId | None = None
    ) -> AuditEvent:
        return self.record(
            AuditKind.BREAKER,
            node=node,
            detail={
                "state": decision.state.value,
                "action": decision.action.value,
                "b_net": decision.b_net,
                "mixing_ratio": decision.mixing_ratio,
                "reason": decision.reason,
            },
        )

    def record_panel(self, result: PanelResult, *, node: NodeId | None = None) -> AuditEvent:
        return self.record(
            AuditKind.INTERVENTION,
            node=node,
            detail={
                "decision": result.decision.value,
                "confidence": result.confidence,
                "pruned": [{"agent": p.agent_type, "reason": p.reason} for p in result.pruned],
            },
        )
