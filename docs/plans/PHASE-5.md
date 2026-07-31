# PHASE 5 — Evidence (M5, v0.5)

**Goal.** Make every decision the system made auditable. Record an append-only causal trail of
probes, scores, weights, breaker decisions, and interventions; export it as **SARIF 2.1.0** for
compliance tooling; and render a human-readable run report that traces a breach back to its
originating node and perturbation.

**Explicit non-goals.** No new detection/propagation/control/intervention logic — M5 *observes* what
M1–M4 already produce. No benchmark reproduction (still out of scope, per DESIGN §2c). No claim that
the SARIF output is legally sufficient for any compliance regime; it is a machine-readable trail.

**The one-sentence success test.** After a run that halts on a seeded biased node, the audit trail
(a) is append-only, (b) exports to SARIF that validates against the SARIF 2.1.0 schema, and (c) lets a
report point at the exact node and perturbation axis that caused the breach.

**Inheritance.** The value types already carry the evidence: `BiasScore` (effect size, CI, p-value,
axis, estimator), `WeightResult`, `BreakerDecision` (reason, frozen payload), `PanelResult` (pruned
agents + reasons), `InterventionResult` (trail). M5 records and serializes these; it does not recompute
them. The LangGraph `BiasState` already has an `audit` channel M5 formalizes.

---

## 1. Research questions

### R1 — What is an audit event, and how is order/time handled?

**Recommendation:** an `AuditEvent` with a monotonic **sequence index** (not wall-clock, so trails are
reproducible and diffable), an event `kind`, an optional `node`, and a structured `detail` mapping.
Wall-clock time is *optional* and injected (a clock callable), never read implicitly — same discipline
as the seeded-RNG rule. The `AuditTrail` is append-only: it exposes `record(...)` and read views, with
no mutation or deletion of past events.

### R2 — SARIF 2.1.0: real validation or "inspired"?

DESIGN §2c is explicit: only claim "SARIF 2.1.0" if the output **validates against the published
schema in CI**; otherwise say "SARIF-inspired". 

**Recommendation:** bundle the SARIF 2.1.0 JSON schema as a committed fixture and validate exported
documents against it with `jsonschema` in the test suite (a `sarif` extra). Map the trail to a single
`run` with a `tool.driver` (name = the library, rules = the event kinds / axes) and one `result` per
significant finding (`ruleId`, `level`, `message`, `locations` naming the node, plus properties for
effect size / p-value / weight). If bundling+validating proves infeasible in CI, downgrade the claim
to "SARIF-inspired JSON" in the docs rather than overstate it. The honesty rule wins over the label.

### R3 — Report content and format

**Recommendation:** two renderers off the same trail — a `dict`/JSON summary (machine-consumable) and
a self-contained HTML report (no external assets) showing the `B_net` trajectory over supersteps, the
per-node scores, the breaker timeline, and the breach trace-back. HTML is a string the caller can
write anywhere; no server, no template engine dependency.

### R4 — Breach trace-back

**Recommendation:** the trail records enough to answer "why did this halt?" — the sequence of scores
that drove `B_net` over `tau`, the top-contributing node (highest `weight × magnitude`), and its axis.
A `trace_breach(trail)` helper returns that provenance; the report surfaces it prominently.

### R5 — Extended observability (banked, DESIGN §8)

**Recommendation:** the trail also accepts the M3/M4 observability signals — trigger reason, the
routing/intervention path taken, and (when a router exists) routing entropy over time. These are
optional event kinds, recorded when supplied; v0.5 wires the ones already produced (trigger reason,
intervention path) and leaves routing-entropy as a documented event kind for when a real router lands.

---

## 2. Work packages

### WP1 — Audit trail (`audit/trail.py`)
`AuditEvent` (seq, kind, node, detail, optional time) and `AuditTrail` (append-only `record`, read
views, helpers to record a `BiasScore` / `BreakerDecision` / `PanelResult`). Frozen events.
*Accept when:* events are ordered by a monotonic seq; the trail is append-only (no public mutation of
past events); recording the M1–M4 value types produces well-formed events.

### WP2 — SARIF export (`audit/sarif.py`)
`to_sarif(trail) -> dict` producing a SARIF 2.1.0 document; schema fixture + `jsonschema` validation.
*Accept when:* the export validates against the bundled SARIF 2.1.0 schema in CI (or, if that is not
achievable, the docs say "SARIF-inspired" and the structural invariants are tested).

### WP3 — Reporting (`audit/report.py`)
`to_json(trail) -> dict` and `to_html(trail) -> str` (self-contained), with the `B_net` trajectory,
per-node scores, breaker timeline, and breach trace-back.
*Accept when:* JSON round-trips the trail's key facts; HTML is self-contained (no external refs) and
contains the trajectory and the breach origin.

### WP4 — Trace-back + audit study + cut v0.5
`trace_breach(trail)` provenance helper; an end-to-end demo (seeded DoT run → trail → SARIF + report →
trace-back to origin); a short study/report. Then README/ROADMAP/CHANGELOG, bump 0.5.0, tag.
*Accept when:* a seeded breach traces back to the correct node + axis; SARIF validates; report renders;
v0.5.0 tagged on green CI.

## 3. Test strategy

Four layers as before. M5-specific: SARIF schema validation (WP2); append-only property (WP1);
HTML self-containment check (no `http`/external `src`); trace-back correctness on a seeded breach. All
deterministic (seq indices, injected clock); no network; SARIF schema is a committed fixture.

## 4. Risks

| Risk | Signal | Response |
| --- | --- | --- |
| Overclaiming SARIF conformance | "SARIF 2.1.0" without validation | Validate against the bundled schema in CI, or downgrade the claim (R2). |
| Non-reproducible trails | wall-clock timestamps | Monotonic seq index; time is optional + injected (R1). |
| HTML pulls external assets | CSP / offline breakage | Self-contained HTML; test asserts no external refs (R3). |
| Trail mutation | audit integrity lost | Append-only API; frozen events; property test (WP1). |
| Scope creep into evaluation | benchmark work sneaks in | M5 reports what M1–M4 produce; no benchmark reproduction (non-goal). |

## 5. Definition of done

- [ ] Append-only `AuditTrail` with monotonic-seq `AuditEvent`s; records M1–M4 value types
- [ ] `to_sarif` validates against the bundled SARIF 2.1.0 schema in CI (or claim downgraded honestly)
- [ ] `to_json` + self-contained `to_html` with trajectory, timeline, and breach trace-back
- [ ] `trace_breach` returns the correct originating node + axis on a seeded breach
- [ ] Audit study/report published; end-to-end seeded-run demo reproducible
- [ ] `mypy --strict` clean on 3.10–3.12; no new detection/control logic; v0.5.0 tagged on green CI
