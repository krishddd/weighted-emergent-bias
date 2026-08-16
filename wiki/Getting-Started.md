# Getting Started

## Install

```bash
pip install weighted-emergent-bias
```

Python 3.10+. Runtime dependencies are just `numpy` and `networkx` — no LLM SDK is bundled, and
nothing makes a network call unless you make one.

| Extra | Command | Adds |
| --- | --- | --- |
| `langgraph` | `pip install "weighted-emergent-bias[langgraph]"` | Reference LangGraph adapter |
| `anthropic` | `pip install "weighted-emergent-bias[anthropic]"` | Real-model adapter (**billable calls**) |
| `study` | `pip install "weighted-emergent-bias[study]"` | `matplotlib` for study plots |
| `dev` | `pip install -e ".[dev]"` | Test, lint, type-check toolchain |

## 1. Perturb an input

Perturbation walks nested payloads and edits only string leaves — structure, keys, and non-string
values are held fixed.

```python
from weighted_emergent_bias import AxisSpec, Substitution, perturb

gender = AxisSpec(
    name="gender",
    substitutions=(Substitution("he", "she"), Substitution("his", "her")),
)

perts = perturb("He submitted his application", [gender])
print(perts[0].perturbed)  # -> "She submitted her application"
```

**No axis list ships as a default.** Shipping a fixed set of demographic axes would encode its own
bias, so axes are a required explicit argument. Illustrative sets are documented, not baked in.

## 2. Score a node against its own noise floor

This is the load-bearing step. You supply repeated samples of the node's output under the standard
and perturbed inputs; the library measures the counterfactual divergence *in excess of* the node's
own sampling noise, via a permutation test.

```python
import numpy as np
from weighted_emergent_bias import TaskMode, compute_local_bias

score = compute_local_bias(
    baseline,          # >= 2 resampled representations under the standard input
    counterfactual,    # >= 2 under the perturbed input
    task_mode=TaskMode.CHOICE,      # distributions; GENERATIVE takes embeddings
    rng=np.random.default_rng(42),  # required — results must be reproducible
)
print(score.effect_size, score.p_value, score.is_significant())
```

`rng` is mandatory by design. A defaulted RNG would make results irreproducible, and the whole
apparatus exists to produce a number someone can check.

## 3. Weight by blast radius, then accumulate

```python
from weighted_emergent_bias import AgentDAG, NetworkAccumulator, dependency_weights

dag = AgentDAG([("router", "worker"), ("router", "judge"), ("worker", "judge")])
weights = dependency_weights(dag).weights  # normalized, sums to 1

acc = NetworkAccumulator()                     # fast + slow bias-corrected EWMA
state = acc.update({"router": 0.4}, weights)   # a biased *central* node fires
print(round(state.fast, 3))
```

A biased router moves `B_net` far more than a biased leaf — that is the entire point of weighting.

## 4. Break the circuit

```python
from weighted_emergent_bias import CircuitBreaker, ControlMachine

machine = ControlMachine(CircuitBreaker(tau_enter=0.3, tau_exit=0.15))
decision = machine.step(fast=0.5, slow=0.1)
print(decision.action, decision.state)
# BreakerAction.REROUTE BreakerState.INTERVENTION
```

Thresholds should come from `calibrate_thresholds` on control runs, not from constants you picked.
See **[Invariants](Invariants)** for why.

## 5. Wire it into a graph

The optional LangGraph adapter stages each node's output and only promotes it once the breaker
clears — so a biased payload never reaches the next node. That edge-level interception is the
difference between catching bias and catching it too late.

## Running the test suite

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && python -m mypy && pytest -m "not slow"
```

The default suite needs no network and no API key. The CI matrix across 3.10/3.11/3.12 is
load-bearing — numpy's type stubs differ across versions.
