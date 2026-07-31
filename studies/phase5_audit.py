"""Phase 5 evidence study -- the WP7/WP4 deliverable.

Runs a small but real M1->M5 pipeline (probe -> accumulate -> breaker -> intervene), records every
decision into an audit trail, and then exercises the evidence layer: SARIF export (validated against
the bundled 2.1.0 schema), a breach trace-back, and a self-contained HTML report. Numbers feed
``docs/studies/phase5-evidence.md``; a sample report is written next to it.

    python studies/phase5_audit.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from weighted_emergent_bias import (
    AgentDAG,
    AuditKind,
    AuditTrail,
    CircuitBreaker,
    ControlMachine,
    InterventionContext,
    InterventionRunner,
    MaderaEditor,
    NetworkAccumulator,
    TaskMode,
    compute_local_bias,
    dependency_weights,
    node_magnitude,
    to_html,
    to_json,
    to_sarif,
    trace_breach,
)
from weighted_emergent_bias.scoring.divergence import softmax
from weighted_emergent_bias.testing import (
    DecayEditor,
    FakeDiagnoser,
    FakeLLMClient,
    FakeRetriever,
    bias_field_scorer,
)
from weighted_emergent_bias.testing import markers as mk

SEED = 20260801


async def _probe(bias: float, rng_seed: int, axis: str = "gender") -> Any:
    client = FakeLLMClient(
        np.random.default_rng(rng_seed),
        n_candidates=5,
        bias={axis: bias} if bias else {},
        logit_noise=0.5,
    )
    cf_prompt = f"{mk.mark(axis, 'alt')} the applicant"

    async def group(prompt: str) -> list[Any]:
        return [softmax(await client.score_candidates(prompt, client.candidates)) for _ in range(8)]

    base = await group("the applicant")
    cf = await group(cf_prompt)
    return compute_local_bias(
        base,
        cf,
        task_mode=TaskMode.CHOICE,
        rng=np.random.default_rng(rng_seed + 1),
        axis=axis,
        n_permutations=300,
    )


async def build_trail() -> AuditTrail:
    trail = AuditTrail()

    # M1: probe a biased router node and a clean worker node.
    router_score = await _probe(bias=4.0, rng_seed=SEED)
    worker_score = await _probe(bias=0.0, rng_seed=SEED + 10)
    trail.record_score(router_score, node="router")
    trail.record_score(worker_score, node="worker")

    # M2 + M3: accumulate the router's magnitude and step the breaker until it halts.
    weights = dependency_weights(AgentDAG([("router", "worker")])).weights
    magnitude = node_magnitude(router_score)
    accumulator = NetworkAccumulator()
    machine = ControlMachine(CircuitBreaker(tau_enter=0.15, tau_exit=0.07))
    for _ in range(4):
        state = accumulator.update({"router": magnitude}, weights)
        decision = machine.step(state.fast, state.slow, payload={"node": "router"})
        trail.record_decision(decision, node="router")
        if decision.halted:
            break

    # M4: repair via MADERA and record the outcome.
    runner = InterventionRunner(
        madera=MaderaEditor(
            FakeDiagnoser(), FakeRetriever(), DecayEditor(0.3), bias_field_scorer, threshold=0.1
        )
    )
    ctx = InterventionContext(biased_node_count=1, total_node_count=5, max_node_magnitude=magnitude)
    result = await runner.run(ctx, {"bias": magnitude})
    trail.record(
        AuditKind.INTERVENTION,
        node="router",
        detail={"strategy": result.strategy, "converged": result.converged},
    )
    return trail


def _validate_sarif(doc: dict[str, Any]) -> bool:
    try:
        import jsonschema
    except ImportError:
        return False
    schema_path = Path("tests/fixtures/sarif-2.1.0.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(doc, schema)
    return True


async def run_all(out: Path) -> dict[str, Any]:
    trail = await build_trail()
    sarif = to_sarif(trail)
    summary = to_json(trail)
    html = to_html(trail)

    (out / "phase5-sample-report.html").write_text(html, encoding="utf-8")
    (out / "phase5-sarif.json").write_text(json.dumps(sarif, indent=2), encoding="utf-8")

    return {
        "n_events": len(trail),
        "by_kind": summary["by_kind"],
        "sarif_valid_2_1_0": _validate_sarif(sarif),
        "sarif_result_count": len(sarif["runs"][0]["results"]),
        "breach": trace_breach(trail),
        "html_self_contained": ("http://" not in html and "https://" not in html),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 evidence study")
    parser.add_argument("--quick", action="store_true")  # accepted for parity
    parser.add_argument("--out", type=Path, default=Path("docs/studies"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results = asyncio.run(run_all(args.out))
    (args.out / "phase5-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
