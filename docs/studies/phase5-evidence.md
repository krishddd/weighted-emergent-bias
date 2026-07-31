# Phase 5 evidence study

**Reproduce:** `python studies/phase5_audit.py`. Numbers come from that script; raw output in
`phase5-results.json`, the exported SARIF in `phase5-sarif.json`, and a rendered report in
`phase5-sample-report.html`.

**Scope.** A small but *real* M1→M5 run on the ground-truth fake — probe → accumulate → breaker →
intervene — with every decision recorded, then the evidence layer exercised on that trail. It shows
the audit/SARIF/report mechanics work end to end; it is not a claim about real models.

---

## The run

Two nodes are probed: a biased `router` (gender bias dialed in) and a clean `worker`. The router's
magnitude is accumulated and the breaker trips; MADERA repairs the concentrated bias. Every step is
recorded into an append-only `AuditTrail`.

| Evidence output | Result |
| --- | --- |
| Events recorded | 4 (2 score, 1 breaker, 1 intervention) |
| **SARIF 2.1.0 valid** (against the bundled schema) | ✅ **true** |
| SARIF results emitted | 4 |
| HTML report self-contained (no external assets) | ✅ true |

## Breach trace-back

`trace_breach` answers "why did this halt?" by pointing at the top significant score before the halt:

| Field | Value |
| --- | --- |
| origin node | `router` |
| origin axis | `gender` |
| origin effect size | 12.04 |
| `B_net` at halt | 0.245 |

The trail traces the breach back to the exact node and perturbation axis that caused it — the M5
success criterion.

## What ships

- **`AuditTrail`** — append-only, reproducible (monotonic sequence index; wall-clock optional and
  injected), recording `BiasScore` / `BreakerDecision` / `PanelResult`.
- **`to_sarif`** — a SARIF 2.1.0 log validated against the published schema in CI, so the "SARIF
  2.1.0" claim is earned (DESIGN §2c).
- **`to_json` / `to_html`** — a machine summary and a self-contained report (inline CSS + inline SVG
  trajectory, HTML-escaped), plus `trace_breach` for provenance.

Sample artifacts: [`phase5-sample-report.html`](phase5-sample-report.html),
[`phase5-sarif.json`](phase5-sarif.json).
