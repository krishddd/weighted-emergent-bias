"""SARIF 2.1.0 export of the audit trail (M5).

Maps the causal trail to a SARIF 2.1.0 log so findings land in standard code-scanning / compliance
tooling. The output validates against the published SARIF 2.1.0 schema in CI (see
``tests/test_sarif.py``); per DESIGN §2c we only make that claim because the validation runs.

Each significant score, halting breaker decision, and intervention becomes a SARIF ``result`` with a
``ruleId``, a severity ``level``, a ``message``, a logical ``location`` naming the node, and the raw
event ``detail`` in its property bag.
"""

from __future__ import annotations

from typing import Any

from .trail import AuditKind, AuditTrail

_SCHEMA_URI = "https://json.schemastore.org/sarif-2.1.0.json"
_INFO_URI = "https://github.com/krishddd/weighted-emergent-bias"


def _tool_version() -> str:
    try:
        from importlib.metadata import version

        return version("weighted-emergent-bias")
    except Exception:
        return "0"


def _result_for(event: Any) -> dict[str, Any] | None:
    """Turn one audit event into a SARIF result, or None if it is not a reportable finding."""
    detail = dict(event.detail)
    if event.kind is AuditKind.SCORE:
        significant = bool(detail.get("significant"))
        level = "warning" if significant else "note"
        axis = detail.get("axis") or "unknown"
        rule_id = f"bias/{axis}"
        text = (
            f"node {event.node!r}: counterfactual bias on axis '{axis}' "
            f"(effect {float(detail.get('effect_size', 0.0)):.2f}, "
            f"p={float(detail.get('p_value', 1.0)):.3f})"
        )
    elif event.kind is AuditKind.BREAKER:
        if detail.get("action") == "proceed":
            return None
        level = "error"
        rule_id = f"breaker/{detail.get('action', 'halt')}"
        text = f"breaker {detail.get('state')}: {detail.get('reason')}"
    elif event.kind is AuditKind.INTERVENTION:
        level = "note"
        rule_id = "intervention"
        text = f"intervention decision: {detail.get('decision')}"
    else:
        return None

    result: dict[str, Any] = {
        "ruleId": rule_id,
        "level": level,
        "message": {"text": text},
        "properties": detail,
    }
    if event.node is not None:
        result["locations"] = [{"logicalLocations": [{"name": event.node}]}]
    return result


def to_sarif(trail: AuditTrail, *, tool_version: str | None = None) -> dict[str, Any]:
    """Export ``trail`` as a SARIF 2.1.0 log document (a JSON-serializable dict)."""
    results = [r for r in (_result_for(e) for e in trail.events) if r is not None]
    rule_ids = sorted({r["ruleId"] for r in results})
    rules = [{"id": rid, "shortDescription": {"text": rid}} for rid in rule_ids]

    return {
        "version": "2.1.0",
        "$schema": _SCHEMA_URI,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "weighted-emergent-bias",
                        "version": tool_version if tool_version is not None else _tool_version(),
                        "informationUri": _INFO_URI,
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }
