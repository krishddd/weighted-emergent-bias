# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

## What this is

`weighted-emergent-bias` — a runtime circuit-breaker for Degeneration-of-Thought (DoT) bias in
multi-agent LLM systems. It detects demographic bias per node (counterfactual invariance,
anchored to the node's own sampling-noise floor — **not** peer consensus), weights it by graph
centrality, accumulates it, and halts the run when it crosses a threshold.

Read [docs/DESIGN.md](docs/DESIGN.md) (esp. §0 scope and §8 review revisions),
[docs/ROADMAP.md](docs/ROADMAP.md), and [docs/plans/PHASE-1.md](docs/plans/PHASE-1.md) before
making design decisions.

## Commands

```bash
pip install -e ".[dev]"     # setup
ruff check . && ruff format --check . && mypy && pytest -m "not slow"   # the four CI checks
```

## Architecture in one screen

Five layered modules, built in order — each is a number the next transforms, so a wrong M1
cannot be recovered downstream:

- **M1 Detection** (`scoring/`) — perturbation → divergence → noise floor → `BiasScore`. In
  progress. Pure math + client protocols; no graph, no framework, no vendor SDK.
- **M2 Propagation** (`topology/`, `accumulation.py`) — transposed Katz weight + multi-scale EWMA.
- **M3 Control** (`breaker.py`, `integrations/langgraph/`) — hysteresis breaker + state machine.
- **M4 Intervention** (`intervention/`) — skeptic panel + trust graph + MADERA.
- **M5 Evidence** (`audit/`) — causal trail + SARIF + reports.

## Conventions that matter here

- **`mypy --strict` is non-negotiable**, and the **Python 3.10/3.11/3.12 matrix is load-bearing**:
  numpy's type stubs differ across versions, so mypy passing on one interpreter does *not* mean
  it passes on all. Reproduce a version-specific failure by pinning that numpy
  (`pip install "numpy==2.2.6" && mypy`). This has broken CI twice — do not trust local 3.11 alone.
- Annotate numpy locals explicitly as `FloatArray` so shape inference can't narrow across versions.
- `from __future__ import annotations` in every module.
- Frozen, self-validating value types (invariants in `__post_init__`).
- Tests: all randomness via the seeded `rng` fixture; Hypothesis property tests for algebraic
  contracts; the default suite needs no network (`slow` marker for anything that does).
- **The fake client (`testing/fake.py`) is the only place ground truth exists** — it has a
  dial-able known bias, and the whole calibration story (WP4/WP7) is measured against it.

## Scope discipline

Ship what the current milestone defines and no more. "While I'm here" work that belongs to a
later module gets banked in the design docs (see DESIGN §8), not built early. WEB makes **no
validated-performance claims**; prior-work benchmark numbers are cited as prior work only.

## Housekeeping

The local source material (`*.docx`, `*.pdf`, `NotebookLM*.png`) is gitignored and
reference-only — never commit it.
