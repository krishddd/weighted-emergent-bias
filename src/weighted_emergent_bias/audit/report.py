"""Run reports over the audit trail (M5): JSON, self-contained HTML, and breach trace-back.

``trace_breach`` answers "why did this halt?" by pointing at the highest-effect significant score
that preceded the breaker's halt. ``to_json`` is the machine-readable summary; ``to_html`` renders a
single self-contained page (no external assets, so it works offline and under a strict CSP) with the
``B_net`` trajectory, the event timeline, and the breach origin.
"""

from __future__ import annotations

import html
from typing import Any

from .trail import AuditKind, AuditTrail

_HALT_ACTIONS = {"halt", "reroute", "escalate"}


def trace_breach(trail: AuditTrail) -> dict[str, Any] | None:
    """Provenance of the first halt: the breaker reason plus the top significant score before it.

    Returns ``None`` if the run never halted.
    """
    halt = next(
        (
            e
            for e in trail.events
            if e.kind is AuditKind.BREAKER and e.detail.get("action") in _HALT_ACTIONS
        ),
        None,
    )
    if halt is None:
        return None
    prior_scores = [
        e
        for e in trail.events
        if e.kind is AuditKind.SCORE and e.seq < halt.seq and e.detail.get("significant")
    ]
    top = max(prior_scores, key=lambda e: float(e.detail.get("effect_size", 0.0)), default=None)
    return {
        "halt_seq": halt.seq,
        "reason": halt.detail.get("reason"),
        "b_net": halt.detail.get("b_net"),
        "origin_node": top.node if top is not None else halt.node,
        "origin_axis": top.detail.get("axis") if top is not None else None,
        "origin_effect_size": top.detail.get("effect_size") if top is not None else None,
    }


def _bnet_trajectory(trail: AuditTrail) -> list[dict[str, Any]]:
    return [
        {"seq": e.seq, "b_net": e.detail.get("b_net")}
        for e in trail.events
        if e.kind is AuditKind.BREAKER
    ]


def to_json(trail: AuditTrail) -> dict[str, Any]:
    """A machine-readable summary of the trail: counts, trajectory, breach, and full events."""
    by_kind: dict[str, int] = {}
    for e in trail.events:
        by_kind[e.kind.value] = by_kind.get(e.kind.value, 0) + 1
    return {
        "n_events": len(trail),
        "by_kind": by_kind,
        "b_net_trajectory": _bnet_trajectory(trail),
        "breach": trace_breach(trail),
        "events": [
            {"seq": e.seq, "kind": e.kind.value, "node": e.node, "detail": dict(e.detail)}
            for e in trail.events
        ],
    }


def _sparkline_svg(values: list[float]) -> str:
    """A tiny inline SVG polyline of the B_net trajectory. Empty string if nothing to plot."""
    points = [v for v in values if v is not None]
    if len(points) < 2:
        return ""
    width, height = 320, 60
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0
    step = width / (len(points) - 1)
    coords = " ".join(
        f"{i * step:.1f},{height - (v - lo) / span * (height - 8) - 4:.1f}"
        for i, v in enumerate(points)
    )
    return (
        f'<svg width="{width}" height="{height}" role="img" aria-label="B_net trajectory">'
        f'<polyline fill="none" stroke="#b8860b" stroke-width="2" points="{coords}"/></svg>'
    )


def to_html(trail: AuditTrail) -> str:
    """Render the trail as a single self-contained HTML page (no external assets)."""
    breach = trace_breach(trail)
    trajectory = [d["b_net"] for d in _bnet_trajectory(trail) if d["b_net"] is not None]

    rows = "\n".join(
        "<tr><td>{seq}</td><td>{kind}</td><td>{node}</td><td>{detail}</td></tr>".format(
            seq=e.seq,
            kind=html.escape(e.kind.value),
            node=html.escape(str(e.node)),
            detail=html.escape(", ".join(f"{k}={v}" for k, v in e.detail.items())),
        )
        for e in trail.events
    )

    if breach is None:
        breach_html = '<p class="ok">No breach: the run stayed below threshold.</p>'
    else:
        breach_html = (
            '<div class="breach"><strong>Breach traced to</strong> '
            f"node <code>{html.escape(str(breach['origin_node']))}</code> "
            f"on axis <code>{html.escape(str(breach['origin_axis']))}</code> "
            f"(reason: {html.escape(str(breach['reason']))}).</div>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Bias audit report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.3rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f0f0f0; }}
  .breach {{ background: #fdecea; border: 1px solid #f5c6cb; padding: 8px 12px; }}
  .ok {{ color: #1f7a1f; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
</style></head>
<body>
<h1>weighted-emergent-bias — run audit</h1>
<p>{len(trail)} events recorded.</p>
{breach_html}
<h2>B_net trajectory</h2>
{_sparkline_svg(trajectory) or "<p>(no breaker events)</p>"}
<h2>Event timeline</h2>
<table><thead><tr><th>#</th><th>kind</th><th>node</th><th>detail</th></tr></thead>
<tbody>
{rows}
</tbody></table>
</body></html>"""
