# PHASE 1 — Detection core (M1, v0.1)

**Goal.** Given one agent node, one input payload, and an LLM client, produce a calibrated
bias score `B_i` with an honest uncertainty estimate — with zero graph, framework, or vendor
dependency.

**Explicit non-goals.** No DAG. No centrality. No EWMA. No LangGraph. No interventions. No
threshold. Phase 1 does not decide whether anything is "too biased" — it only measures, and
reports how confident the measurement is. Thresholding is M3's job and adding it early would
let a bad measurement hide behind a tuned cutoff.

**The one-sentence success test.** An unbiased model sampled at production temperature must
score at the noise floor, and a model with a known injected bias must score above it,
monotonically in the injected magnitude. If we can only get one of those two, we have not
finished Phase 1.

**What "bias" means here (scope).** `B_i` measures **counterfactual demographic invariance** —
how much a node's output shifts when only a protected attribute (or its proxy) changes,
anchored to the node's *own* unperturbed output. It is not a factual-correctness signal and
not a consensus-deviation signal: Phase 1 never compares a node to other nodes. This boundary
(reaffirmed by the 2026-07 external review) is why the noise floor, not an external oracle, is
the anchor — see [DESIGN.md §0](../DESIGN.md). Factual verification, if wanted, is an optional
injectable hook decided at the architecture level, not part of this estimator.

---

## 1. Research questions to resolve before writing the estimator

These are genuinely open. Each gets a spike (a timeboxed experiment, code thrown away
afterwards) before the real implementation. Recommendations below are the starting
hypothesis, not the conclusion.

### R1 — What exactly are we computing divergence *between*? ⚠️ highest risk

This is the question the source documents skip, and getting it wrong makes everything
downstream meaningless.

"Statistical divergence between the two output distributions" is underspecified for
free-form generation. Two generations from two different prompts diverge after the first
differing token, so their per-position token distributions are **not defined over a common
support** and are not comparable. You cannot JS-divergence them in any principled way. A
naive implementation that zips the two token sequences and averages per-position JS produces
a number, and that number is close to meaningless.

Options:

| Option | Definition | Verdict |
| --- | --- | --- |
| **(a) Fixed candidate set** | Score the *same* set of candidate continuations under both the standard and counterfactual prompt. Both are distributions over an identical support, so JS is exact and bounded. | **Correct and well-founded.** This is what the calibration literature actually does. Requires the task to have a candidate set (classification, MCQ, ranking, routing, scoring). |
| **(b) First-token distribution** | Compare only the top-k at position 0. | Cheap and well-defined, but only meaningful when the first token determines the answer. A narrow special case of (a). |
| **(c) Forced-scoring of a shared target** | Compute the sequence logprob of one fixed output string under both prompts. | Well-defined; gives a likelihood ratio, not a divergence. Useful as a secondary signal. |
| **(d) Naive per-position zip** | Align the two generations by index and average per-position divergence. | **Do not do this.** Unaligned supports, length mismatch, silent nonsense. |
| **(e) Semantic distance** | Embed both output texts, take a distance in embedding space. | Not a divergence and not comparable across embedding models — but it is the *only* option that works for free-form generation, which is most agent output. |

**Recommendation:** make the comparison mode explicit rather than pretending one metric
covers everything. `BiasScore` carries a `task_mode`:

- `CHOICE` — a candidate set exists → **option (a)**, exact JS over a shared support. This is
  the high-confidence path, and it covers a lot of real agent nodes (routers, classifiers,
  scorers, rankers, judges — note that MALIBU-style judge agents are *exactly* this shape).
- `GENERATIVE` — free-form text → **option (e)**, embedding distance.

Scores from the two modes are never mixed or averaged. The saving grace is that the noise
floor (R2) normalizes both into effect-size units, which makes them loosely comparable even
though their raw scales are not — but the mode still travels with the score so nobody has to
guess.

**Spike S1:** implement (a) and (e) against a real chat API. Confirm (a) is reachable — some
hosted APIs expose no logprobs at all, in which case `CHOICE` mode degrades to (e) and we
need to know that before designing around it.

### R2 — How do we define the score relative to the noise floor?

An LLM sampled twice on identical input diverges from itself. Let `D_cf` be the
standard-vs-counterfactual divergence and `D_null` the distribution of standard-vs-resampled-
standard divergences.

| Option | Definition | Verdict |
| --- | --- | --- |
| Raw difference | `D_cf − mean(D_null)` | Scale-dependent, no uncertainty, not comparable across nodes or estimators. |
| Standardized effect | `(D_cf − mean(D_null)) / std(D_null)` | Scale-free, comparable across nodes and across `CHOICE`/`GENERATIVE`. Assumes a roughly stable null spread. |
| Permutation test | p-value of `D_cf` against the empirical null | Gives significance, but a p-value is not a magnitude — and M2 needs a magnitude to weight. |

**Recommendation: report all three.** `BiasScore` carries a standardized effect size (the
magnitude M2 will consume), a bootstrap confidence interval, and a permutation p-value. The
effect size is the number; the p-value is what stops a wide-CI score from being treated as
real. A score whose CI straddles zero must be visibly distinguishable from a confident one,
because M3 will eventually make a halt decision on it.

**Spike S2:** how many samples does a usable CI actually need? Sweep n ∈ {3, 5, 10, 20, 50}
and find where CI width plateaus. This directly sets the cost of the whole system — if a
usable score needs n=50, the 3–6× overhead estimate in DESIGN.md is badly wrong and the
architecture needs rethinking now, not in M2.

### R3 — Multiple perturbation axes collapse to one score how?

Perturbing gender, race, age, and sociolect separately gives `k` divergences.

**Recommendation:** never collapse them at the estimator. Return a per-axis vector, and let
the *consumer* aggregate — with `max` as the documented default, because worst-case across
protected attributes is the safety-relevant statistic and averaging lets a severe
single-axis bias hide behind three clean axes. Keep the full vector for M5's audit trail:
"which axis tripped this" is exactly what an auditor asks first.

### R4 — What temperature do we probe at?

Tempting to probe at temperature 0 to kill the noise floor. But at temp 0 the null collapses
to ~0, `std(D_null) → 0`, and the standardized effect size in R2 divides by zero.

More importantly it measures the wrong system: the deployed agent runs at its production
temperature, and that is the regime whose emergent behavior we are guarding.

**Recommendation:** probe at the node's production sampling settings, with a floor on
`std(D_null)` to keep the standardization numerically stable, and an explicit
deterministic-node code path (temp 0 → any nonzero divergence is real by construction, no
statistics needed).

### R5 — Does SDC belong in Phase 1?

SDC has the model synthesize in-domain priors when no calibration data exists. It is
genuinely useful, but it is a *calibration input* to the estimator, not the estimator.

**Recommendation:** build it last in Phase 1, behind an interface, and ship v0.1 without it
if the validation study (WP7) is not clean. The core estimator must stand on its own first —
otherwise a bad estimator and a bad prior become impossible to tell apart.

---

## 2. Work packages

Ordered. Each has an acceptance criterion; none is "done" without it.

### WP0 — Sustainable foundations *(do this first, resist the urge to skip)*

Package skeleton under `src/weighted_emergent_bias/`, `pytest` + `pytest-asyncio`, `ruff`,
`mypy --strict` on the whole package, GitHub Actions CI across Python 3.10/3.11/3.12, a
seeded-RNG fixture so every stochastic test is reproducible, and `py.typed`.

*Accept when:* CI is green on an empty package and a deliberately-broken type annotation
fails the build. Strict typing from commit one — retrofitting it onto numeric code later is
miserable, and the whole library is numeric code.

### WP1 — Types and client protocols

`types.py`: `BiasScore` (effect size, CI, p-value, per-axis vector, `task_mode`, estimator,
sample count), `Perturbation`, `ProbeResult`, `DivergenceMethod`, `TaskMode`.
`clients.py`: `LLMClient` / `EmbeddingClient` protocols — async, minimal surface, no vendor
types anywhere.

Plus `testing/fake.py`: a **fake client with a bias knob** — a controllable generative
process with tunable sampling noise and a tunable, *known* bias magnitude along named axes.

*Accept when:* the fake client can produce an unbiased-noisy stream and a
bias=`x` stream on demand, reproducibly under a seed. This fake is the single most important
test asset in the project; everything in WP7 is measured against it, because it is the only
place where ground truth exists.

### WP2 — Perturbation engine

`scoring/perturbation.py`. Axis definitions, substitution into structured payloads, and
proxy-variable handling (zip code, institution, sociolect — the correlates, not just the
explicit attributes).

**No default axis list ships.** A hardcoded set of demographic categories is itself an
editorial position about which identities count, and baking one in would be a bad look for a
fairness library. Axes are a required explicit config; documented example sets live in
`docs/`, not in code.

*Accept when:* perturbing a payload changes only the targeted attribute and its proxies,
verified by property test on nested structures; round-tripping an unperturbed axis is
identity.

### WP3 — Divergence estimators

`scoring/divergence.py`. Exact JS over a shared candidate support (log base 2, bounded
[0,1]); embedding-distance fallback; explicit refusal to compare two `BiasScore`s from
different estimators.

*Accept when:* JS matches hand-computed values on known distributions; `JS(p,p) == 0`;
`JS(p,q) == JS(q,p)`; bounded in [0,1] under adversarial inputs including disjoint supports
and zero-probability entries (this is where a KL implementation would return `inf` — the
regression test that documents why we chose JS).

### WP4 — Noise floor and significance ⚠️ the critical package

`scoring/noise.py`. Empirical null via same-input resampling, standardized effect size,
bootstrap CI, permutation p-value, `std` flooring, deterministic-node path.

*Accept when:* on the unbiased fake client, the **false-positive rate is ≤ 5% at α = 0.05**,
measured over ≥ 200 simulated nodes. This is a calibration check, not a smoke test, and it is
the single acceptance criterion in Phase 1 that must not be waived.

### WP5 — Probe orchestration

`scoring/probe.py`. `LOOCProbe` runs standard, counterfactual (per axis), and null-resample
probes **concurrently** — `asyncio.gather`, never sequentially, because this sits on the
critical path of a production agent run. Sample-budget config, per-axis result vector,
partial-failure handling (one axis erroring must not void the whole probe).

*Accept when:* a probe with `k` axes and `n` samples issues exactly the predicted number of
client calls, wall-clock is bounded by the slowest single call rather than their sum, and a
deliberately failing axis degrades to a partial result with the gap recorded — never a
silently complete-looking one.

### WP6 — Synthetic Data Calibration

`scoring/synthetic.py`. Runtime in-domain prior generation, behind an interface, optional.

*Accept when:* the estimator produces identical results with SDC disabled, and enabling it
measurably tightens CIs on the fake client. If it does not tighten them, it does not ship —
and that finding gets written down, because it would be a real result about the technique.

### WP7 — Validation study *(the actual Phase 1 deliverable)*

Not a test suite — a written report, `docs/studies/phase1-calibration.md`, with plots.

1. **Calibration:** false-positive rate vs. nominal α on unbiased clients across the noise range.
2. **Sensitivity:** recovered effect size vs. injected bias magnitude. Is it monotonic? Linear? Where is the detection floor?
3. **Sample budget:** CI width vs. n (resolves spike S2 and prices the whole system).
4. **Estimator agreement:** on tasks where both `CHOICE` and `GENERATIVE` modes apply, do they rank nodes consistently?
5. **Cost:** measured calls and latency per probe, versus the 3–6× estimate in DESIGN.md.

*Accept when:* the report exists, the plots are reproducible from committed code, and the
detection floor and cost multiplier are stated as numbers. **These are the first numbers this
project may honestly call its own** — everything else in DESIGN.md §2c belongs to someone
else's paper.

---

## 3. Test strategy

Four layers, because a numeric library that only has example-based tests is a library whose
bugs are all in the cases nobody thought of:

1. **Unit** — closed-form checks against hand-computed values.
2. **Property** (Hypothesis) — symmetry, boundedness, identity, monotonicity. Especially: for any two distributions, `0 ≤ JS ≤ 1`.
3. **Statistical** — calibration and power against the fake client. Seeded, so they are reproducible rather than flaky, and asserted on rates over many trials rather than on any single draw.
4. **Contract** — the `LLMClient` protocol verified against both the fake and one real provider, the latter marked `slow` and excluded from default CI.

No network in the default test run. Ever. A test suite that needs an API key is a test suite
contributors will not run.

## 4. Risks

| Risk | Signal | Response |
| --- | --- | --- |
| Sample budget makes it unaffordable (R2/S2) | Usable CI needs n ≳ 20 | Surfaces in WP7 as a cost number. Amortize null estimation across probes — the noise floor is a property of the node, not of one input, so it can be cached and refreshed periodically rather than recomputed per probe. |
| Target APIs expose no logprobs (R1/S1) | Spike S1 fails on the intended provider | `CHOICE` mode degrades to embedding distance. Documented honestly rather than papered over; the mode field means nothing silently changes meaning. |
| Fake client is unrepresentative | Estimator passes on the fake, behaves oddly on a real model | The fake proves *correctness of the estimator*, not realism. WP7 item 4 runs against a real model to check the two agree on ranking. |
| Embedding distance is not scale-comparable | `GENERATIVE` scores cluster oddly | Already mitigated by design: standardizing against the per-node null makes it scale-free. Verify explicitly in WP7 rather than assuming. |
| Scope creep into M2 | "just a small `w_i` helper" | Phase 1 ships a score, not a decision. No threshold, no graph, no weighting. |

## 5. Definition of done

- [ ] `compute_local_bias`, `LOOCProbe`, `perturb` importable, typed, documented
- [ ] `mypy --strict` clean; CI green on 3.10–3.12
- [ ] WP4 calibration criterion met (FPR ≤ 5% at α = 0.05 over ≥ 200 simulated nodes)
- [ ] Sensitivity monotonic in injected bias magnitude
- [ ] `docs/studies/phase1-calibration.md` published with reproducible plots
- [ ] Detection floor and cost multiplier stated as measured numbers
- [ ] Zero graph/framework/vendor imports in the package
- [ ] Default test run needs no network and no API key
