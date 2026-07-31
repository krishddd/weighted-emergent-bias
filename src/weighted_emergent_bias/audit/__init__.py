"""Evidence layer (M5): append-only causal audit trail, SARIF export, reporting."""

from __future__ import annotations

from .sarif import to_sarif
from .trail import AuditEvent, AuditKind, AuditTrail

__all__ = ["AuditEvent", "AuditKind", "AuditTrail", "to_sarif"]
