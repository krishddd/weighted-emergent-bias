# Contributing

Pre-alpha, moving fast, but the quality bar is fixed: `mypy --strict` clean, tests for every
behavior, and no validated-performance claims that aren't backed by a committed study.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; use .venv/bin/activate on POSIX
pip install -e ".[dev]"
```

## The four checks (CI runs exactly these)

```bash
ruff check .
ruff format --check .
mypy
pytest -m "not slow"
```

`ruff format .` and `ruff check --fix .` apply the fixes. `slow`-marked tests hit a real
network/model provider and are excluded from the default run — the default suite needs no API
key and no network, and it must stay that way.

## Gotcha: the Python matrix is load-bearing

CI runs all four checks under Python **3.10, 3.11, and 3.12**, and this is not ceremony. numpy
ships different type stubs across versions, and code that passes `mypy --strict` under one can
fail under another (this has already broken CI twice — once on a 3.12-only stub syntax, once on
3.10-only ndarray shape typing). **Passing mypy locally on one interpreter is not sufficient.**

To reproduce a version-specific mypy failure locally, pin the numpy the failing runner used and
re-run mypy, e.g.:

```bash
pip install "numpy==2.2.6" && mypy && pip install "numpy>=2.4"
```

## Conventions

- **Types first.** Everything is annotated; `from __future__ import annotations` at the top of
  every module. Explicit `FloatArray` annotations on numpy locals to stop shape-inference from
  narrowing across numpy versions.
- **Frozen, self-validating value types.** Invariants are enforced in `__post_init__`, not
  trusted at call sites.
- **Determinism in tests.** All randomness flows through the seeded `rng` fixture (or an
  explicit seed). No wall-clock, no unseeded RNG.
- **Property tests** (Hypothesis) for anything with an algebraic contract — divergences,
  perturbation invariants.
- **Scope discipline.** A module ships what its milestone defines and no more; see
  [docs/ROADMAP.md](docs/ROADMAP.md). "While I'm here" additions that belong to a later module
  get banked in the design docs, not built early.

## Commits & PRs

- Branch off `main`; don't commit directly to it.
- Conventional-ish messages: a `WP<n>:` or module prefix, a why-not-just-what body.
- Keep the local source material (`*.docx`, `*.pdf`, mind map) out of commits — it is gitignored
  and is reference-only, not distributed.
