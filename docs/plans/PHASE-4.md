# PHASE 4 — Intervention (M4, v0.4)

**Goal.** Repair a halted run instead of just stopping it. When M3 trips, route to a mitigation
strategy and produce a corrected payload whose re-scored bias clears the breaker: a **conformity
spiral** goes to a parallel **skeptic panel** governed by a trust graph; an entrenched **parametric**
bias goes to a **MADERA**-style diagnose → retrieve → rewrite pipeline. Everything is protocol +
reference implementation with a **user-injected LLM callable** — no bundled vendor SDK.

**Explicit non-goals.** No audit trail / SARIF (M5). No new detection (M1), propagation (M2), or
breaker/state-machine (M3) — M4 *plugs into* the M3 intervention hook and the `route_intervention`
seam that already exist. No real web retrieval: MADERA's evidence retriever is an injected callable.

**The one-sentence success test.** A run halted by M3 routes to the panel/MADERA, the returned
corrected payload re-scores below `tau_exit` (so M3's recovery re-check passes and the machine
returns to Normal), and a **correct minority dissenter is never pruned** from the aggregation.

**Inheritance (settled inputs).**

- **The seam already exists** (M3): `route_intervention(context) → "skeptics" | "madera" | "none"`
  and `ControlMachine`'s injectable `intervention_hook`. M4 supplies the implementations behind them.
- **The mixing ratio is advisory** (M3 R2): `BreakerDecision.mixing_ratio` is the sigmoid severity
  M4 uses to blend original vs. corrected output — M3 computes it, M4 applies it.
- **BYO model** (project principle): protocols + reference impls + injected callable; testable against
  the fake client. No vendor SDK, no hard provider.
- **Banked review guardrails** (DESIGN §8): skeptic SPOF, trust-formula grounding, and the
  minority-suppression loop are the *reasons* several design choices below are non-negotiable.

---

## 1. Research questions

### R1 — Skeptic panel: structure and SPOF

The panel fans out parallel adversarial reviewers (DESIGN §6): an **Empirical Auditor** (demands
evidence), a **Devil's Advocate** (argues the opposite), a **Diversity Champion** (surfaces
under-represented views). The review flagged a single-point-of-failure: `Send`-fanning *one* base
skeptic means a compromised skeptic contaminates every branch.

**Recommendation:** a `SkepticAgent` protocol + three reference implementations, each a callable over
a user-injected LLM client with an overridable prompt. Require **≥ 2 skeptics of distinct type** to
run (a panel of one is rejected). Run them **concurrently** (`asyncio.gather`; `Send` in the
LangGraph adapter). Each returns a structured `SkepticVerdict` (does the output stand? a critique, a
proposed revision, a confidence).

### R2 — Trust-graph aggregation, and the minority-suppression trap ⚠️

CortexDebate weights verdicts by a trust score `T = (C + R + I) / S` (Credibility, Reliability,
Intimacy over Self-orientation) and prunes low-trust agents. Two hazards, both banked:

- **`S → 0` sends `T → ∞`.** Clamp the denominator.
- **Pruning on agreement is a suppression loop.** If trust tracks "agreed with the majority," the
  panel silences a correct dissenter — the exact failure gap #4 warned about.

**Recommendation:**
- **Anonymize** verdicts before aggregation (strip agent identity) so trust cannot simply reward
  conformity.
- **Never prune for dissent alone.** Pruning is allowed only on *overconfidence* (a low-quality,
  evidence-free, high-certainty verdict), never on disagreeing with the emerging consensus. This is a
  hard rule with a dedicated test.
- Make **`S` operational** via normalized entropy `H(P)/log(N)` (the review's suggestion) as an
  overconfidence proxy, and **clamp** it. `C`/`R`/`I` are operationalized (evidence presence /
  historical accuracy / domain match) but treated as heuristic.
- Ship **uniform aggregation as the default and baseline**; trust-weighting is opt-in and must beat
  uniform in the WP7 ablation before anyone should believe it. The library ships the mechanism and
  the honest comparison, not a claim.

### R3 — What the panel returns

The panel must yield a *corrected payload*, not just critiques.

**Recommendation:** aggregate verdicts into a `PanelResult`: a decision (original stands / revise /
reject), the chosen corrected payload (a skeptic's proposed revision or a synthesized one), the
aggregate confidence, and which agents were pruned + why (for M5's audit later). The corrected
payload is what M3 re-scores.

### R4 — MADERA pipeline

Three phases (DESIGN §6): **diagnose** the biased logical jump, **retrieve** external counter-
evidence, **rewrite** the reasoning chain — iterating until bias clears.

**Recommendation:** each phase is an injected callable behind a protocol (`Diagnoser`, `Retriever`,
`Editor`); the retriever is user-supplied (no built-in web access). Iterate up to `max_rewrites`,
**re-scoring with M1 after each rewrite** and stopping when the score clears (or the cap is hit,
returning the best attempt with a "not converged" flag). Bounded iteration mirrors M3's
`max_retries` — no infinite rewrite loop.

### R5 — Wiring into M3

M4 provides the `intervention_hook`. On halt it: builds an `InterventionContext` from the frozen
payload + per-node bias, calls `route_intervention`, runs the chosen strategy, and applies the
`mixing_ratio` to blend original vs. corrected output. Because M3's recovery already **re-measures**
`B_net`, a genuine repair drives the machine back to Normal with no extra coupling.

**Recommendation:** an `InterventionRunner` that takes a scorer (M1), a panel, and a MADERA pipeline,
and exposes a `hook` compatible with `ControlMachine`. The runner is pure orchestration; the LLM work
is all injected.

### R6 — Testing without real models

**Recommendation:** a fake skeptic and fake MADERA backed by the existing fake client / a dial —
e.g. a fake skeptic that reliably flags the injected bias and proposes an unbiased revision, and a
fake retriever returning canned counter-evidence. This lets WP7 measure: does post-intervention
`B_net` drop below `tau_exit`? does trust-weighting beat uniform? is the dissenter preserved? All
deterministic under seed, no network.

---

## 2. Work packages

### WP1 — Intervention types
`SkepticVerdict` (stands/revise/reject, critique, proposed_payload, confidence, evidence flag),
`PanelResult` (decision, corrected_payload, confidence, pruned), `TrustScore` (C/R/I/S + clamped
value), `InterventionResult` (repaired_payload, converged, strategy, trail), and an
`InterventionContext` extension carrying the frozen payload + per-node magnitudes. Frozen,
self-validating.
*Accept when:* invariants enforced (confidence in [0,1], clamped `S`); types round-trip.

### WP2 — Skeptic panel (`intervention/skeptics.py`)
`SkepticAgent` protocol + `EmpiricalAuditor` / `DevilsAdvocate` / `DiversityChampion` reference impls
over an injected client; a `SkepticPanel` running ≥2 concurrently.
*Accept when:* a panel of <2 is rejected; agents run concurrently (max-in-flight test as in M3 WP5);
against the fake, the panel flags injected bias and proposes a revision.

### WP3 — Trust graph (`intervention/trust.py`)
Clamped `T = (C+R+I)/S` with entropy-based `S`; anonymized aggregation; pruning that fires only on
overconfidence, **never on dissent**; pluggable uniform vs. trust-weighted aggregator.
*Accept when:* `S→0` does not blow up (clamp test); a **correct dissenter is never pruned** (the
minority-suppression test); uniform and trust-weighted are both selectable.

### WP4 — MADERA pipeline (`intervention/madera.py`)
`Diagnoser`/`Retriever`/`Editor` protocols + a `MaderaEditor` running diagnose→retrieve→rewrite,
re-scoring each iteration, bounded by `max_rewrites`.
*Accept when:* on a biased fake, iterative rewriting drives the re-scored bias down and stops on
convergence; hitting the cap returns the best attempt flagged not-converged.

### WP5 — Intervention runner (`intervention/runner.py`)
`InterventionRunner` wiring `route_intervention` + panel + MADERA into a `hook` for `ControlMachine`,
applying the `mixing_ratio`.
*Accept when:* a run halted by M3 completes the mitigation and, on re-measure, `B_net` clears so the
machine returns to Normal — an end-to-end M1→M4 test.

### WP6 — LangGraph `Send` fan-out (`integrations/langgraph/`)
Prebuilt skeptic-fanout via `Send`, aggregation node. Optional; behind `importorskip`.
*Accept when (langgraph installed):* a graph fans out skeptics in parallel and aggregates; default
suite unaffected.

### WP7 — Intervention study + cut v0.4
Report: post-intervention `B_net` drop; the **trust-vs-uniform ablation** (does trust earn its
place?); the minority-suppression check; MADERA convergence. Reproducible. Then README/ROADMAP/
CHANGELOG, bump 0.4.0, tag.
*Accept when:* the report states, with numbers, whether trust-weighting beats uniform (honestly,
either way), and shows the dissenter preserved and `B_net` dropping post-repair.

---

## 3. Test strategy

Four layers as before. M4-specific must-haves:
- **The minority-suppression test** (WP3): a correct, evidence-backed dissenter against a biased
  majority is never pruned. This is the load-bearing safety property.
- **Concurrency** (WP2): skeptics run in parallel (deterministic max-in-flight tracker, as in M3).
- **Convergence** (WP4/WP5): post-intervention re-score clears `tau_exit`; bounded iteration.
- All against the fake; default suite needs no network; LangGraph tests behind `importorskip`.

## 4. Risks

| Risk | Signal | Response |
| --- | --- | --- |
| Minority suppression | correct dissenter pruned | No-prune-on-dissent rule + anonymization + dedicated test (R2/WP3). |
| Trust formula unearned | trust-weighting shipped as fact | Uniform is default/baseline; trust must win the WP7 ablation; report honestly (R2). |
| `S → 0` blow-up | `T` explodes | Clamp the denominator (R2/WP3). |
| Skeptic SPOF | one bad skeptic taints all branches | Require ≥2 diverse skeptics; BYO callable, no single bundled agent (R1). |
| Infinite rewrite loop | MADERA never converges | `max_rewrites` cap; return best-effort flagged not-converged (R4). |
| Vendor lock-in | hard SDK dependency | Protocols + injected callable; fake-backed tests (project principle). |
| Over-coupling to M3 | brittle hook | Recovery re-measures `B_net`; the hook only needs to return a payload (R5). |

## 5. Definition of done

- [ ] `SkepticPanel`: ≥2 diverse skeptics, concurrent; flags injected bias on the fake
- [ ] Trust graph: clamped `S`; **minority-suppression test passes** (dissenter never pruned); uniform + trust-weighted both selectable
- [ ] MADERA: diagnose→retrieve→rewrite, re-scored, bounded; converges on the fake
- [ ] `InterventionRunner` as a `ControlMachine` hook; end-to-end halt → repair → recovery test
- [ ] LangGraph `Send` fan-out behind `importorskip`; default suite framework-free
- [ ] Intervention study: trust-vs-uniform ablation (honest result), dissenter preserved, `B_net` drop
- [ ] `mypy --strict` clean on 3.10–3.12; no vendor SDK; no M5 work
