"""Evidence layer (M5): append-only causal audit trail, SARIF export, reporting."""

from __future__ import annotations

from .report import to_html, to_json, trace_breach
from .sarif import to_sarif
from .trail import AuditEvent, AuditKind, AuditTrail

__all__ = [
    "AuditEvent",
    "AuditKind",
    "AuditTrail",
    "to_html",
    "to_json",
    "to_sarif",
    "trace_breach",
]
