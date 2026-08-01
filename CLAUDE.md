# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

## What this is

`weighted-emergent-bias` — a runtime circuit-breaker for Degeneration-of-Thought (DoT) bias in
multi-agent LLM systems. It detects demographic bias per node (counterfactual invariance, anchored
to the node's own sampling-noise floor — **not** peer consensus), weights it by graph centrality,
accumulates it, halts the run when it crosses a threshold, repairs it, and audits everything.

Repo: https://github.com/krishddd/weighted-emergent-bias · **all five milestones shipped (v0.5.0)**.

Read [docs/DESIGN.md](docs/DESIGN.md) (esp. §0 scope and §8 review revisions) and
[docs/ROADMAP.md](docs/ROADMAP.md) before making design decisions.

## Where the repo lives ⚠️

`C:\Users\hp\Music\weighted-emergent-bias`. There is a **stale partial copy** at
`C:\Users\hp\Downloads\weighted-emergent-bias` that is not its own git repo (git there walks up to
a `.git` in the home dir). It once hijacked the venv's editable install. If imports resolve to
Downloads, re-run `pip install -e ".[dev]"` from Music and verify with
`python -c "import weighted_emergent_bias as w; print(w.__file__)"`.

## Commands

```bash
pip install -e ".[dev]"     # setup
ruff check . && ruff format --check . && python -m mypy && pytest -m "not slow"   # the four CI checks
```

Use `python -m mypy`, not the bare `mypy` launcher — the launcher exits 1 with no output here.

## Architecture — five layered modules, all shipped

Each is a number the next transforms, so a wrong M1 cannot be recovered downstream.

- **M1 Detection** (`scoring/`) — perturbation → true JSD / embedding divergence → permutation
  noise floor → `BiasScore`. v0.1.0.
- **M2 Propagation** (`topology/`, `accumulation.py`) — transposed Katz blast radius + multi-scale
  (fast/slow) bias-corrected EWMA → `B_net`. v0.2.0.
- **M3 Control** (`breaker.py`, `calibration.py`, `integrations/langgraph/`) — two-threshold
  hysteresis breaker + Normal→Warning→Intervention→Recovery→Escalated machine. v0.3.0.
- **M4 Intervention** (`intervention/`) — skeptic panel + trust graph + MADERA repair + runner.
  v0.4.0.
- **M5 Evidence** (`audit/`) — append-only trail + schema-validated SARIF 2.1.0 + HTML/JSON
  reports + breach trace-back. v0.5.0.

Plans: `docs/plans/PHASE-1..5.md`. Per-phase studies with real numbers: `docs/studies/`.
Review triage: `docs/reviews/2026-07-external-review-response.md`.

## Invariants that must not be broken

Each was learned the hard way; changing one silently breaks something upstream.

- **Never accumulate `effect_size`.** It inflates with sample count (1.9σ at n=3 → 22.9σ at n=30,
  measured in the Phase 1 study). M2 accumulates `node_magnitude` = significance-gated *excess
  divergence*, which is n-stable.
- **Katz runs on the transposed adjacency.** Blast radius is what a node *reaches*. Backwards ranks
  terminal leaves as most critical — plausible-looking and exactly inverted.
- **True JSD, not symmetrized KL.** The mixture form is bounded [0,1] and finite on disjoint
  supports; the Jeffreys form an external reviewer proposed is unbounded and blows up there.
- **The noise floor is non-negotiable.** Thresholding raw divergence is thresholding temperature.
- **No prune-on-dissent.** Trust-graph pruning fires on overconfidence only; a correct
  evidence-backed dissenter is never pruned (dedicated test).
- **No self-updating error history.** Priors are injected-only, or it becomes a suppression loop.
- **No magic thresholds.** `tau` comes from `calibrate_thresholds` on control runs.
- **Determinism.** All randomness via the seeded `rng` fixture; the audit trail uses a monotonic
  seq index, not wall-clock; node ordering is insertion order (Python randomizes set iteration).

## Conventions

- **`mypy --strict` is non-negotiable**, and the **3.10/3.11/3.12 matrix is load-bearing**: numpy's
  stubs differ across versions, so local 3.11 passing does *not* mean CI passes. Reproduce with
  `pip install "numpy==2.2.6" && python -m mypy`. This has broken CI twice.
- Annotate numpy locals explicitly as `FloatArray` so shape inference can't narrow.
- `from __future__ import annotations` everywhere; frozen, self-validating value types.
- Ruff also lints Python inside README code fences, and flags ambiguous Unicode (α, ρ, −) in
  docstrings — write those ASCII.
- Optional integrations (`langgraph`, `anthropic`) are extras, never imported by the core; their
  tests sit behind `importorskip`. The default suite needs no network and no API key.
- **The fake client (`testing/fake.py`) is the only place ground truth exists** — dial-able known
  bias, and the whole calibration story is measured against it.

## Honesty rules (the project's defining discipline)

- **No validated-performance claims on real models.** Everything is demonstrated on the fake.
- Prior-work numbers (MALIBU, BBQ-Hard, the 57.5% SDC figure) are cited **as prior work only**. The
  57.5% figure in particular measures single-model label bias — wrong metric, wrong unit.
- Scope is narrow on purpose: WEB detects **demographic counterfactual bias**, not factual error
  and not consensus deviation (DESIGN §0).
- A claim is made only when a test or study backs it — e.g. "SARIF 2.1.0" is claimed because the
  export validates against the bundled schema in CI.
- Ship what the current milestone defines; bank "while I'm here" work in the design docs.

## Current state / what's left

302 tests, mypy-strict clean on 3.10–3.12, CI green, tags v0.1.0–v0.5.0.

Unreleased on `main`: the optional pre-trigger `Verifier` hook, plus a real-model adapter and
validation harness (`integrations/anthropic_client.py`, `studies/validate_real_model.py`, extra
`[anthropic]`). The harness makes real-model validation one command but **has not been run** — it
needs an API key and makes billable calls.

Open, and **not** solvable by writing code:
- **Real-model validation** — run `python studies/validate_real_model.py` with a key. A single run
  is evidence, not validation.
- **Benchmark reproduction** (MALIBU / BBQ-Hard) — deliberately unscheduled; needs datasets, money,
  and its own workstream.

Known finding: the Anthropic Messages API exposes **no logprobs**, so `TaskMode.CHOICE` is
unreachable analytically there; the adapter recovers it by Monte Carlo (one-hot draws whose group
mean is the empirical distribution), needing a larger `n`. This resolved Phase 1 spike S1.

## Housekeeping

Local source material (`*.docx`, `*.pdf`, `NotebookLM*.png`) is gitignored and reference-only.
Commits use the `krishddd` GitHub noreply email so they count toward contributions.
