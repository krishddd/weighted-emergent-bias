"""Phase 4 intervention study -- the WP7 deliverable.

Answers the honest questions M4 has to answer: does trust-weighting actually beat the uniform
baseline, is the correct minority preserved, does MADERA converge, and does a full halt->repair
loop drop B_net? Emits JSON + a figure; numbers feed ``docs/studies/phase4-intervention.md``.

    python studies/phase4_intervention.py            # full run
    python studies/phase4_intervention.py --quick     # fast run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from weighted_emergent_bias import (
    CircuitBreaker,
    ControlMachine,
    InterventionContext,
    InterventionRunner,
    MaderaEditor,
    PanelDecision,
    SkepticStance,
    SkepticVerdict,
    TrustGraph,
)
from weighted_emergent_bias.testing import (
    DecayEditor,
    FakeDiagnoser,
    FakeRetriever,
    bias_field_scorer,
)


def _conformity_scenario() -> list[SkepticVerdict]:
    """An overconfident biased majority vs. one correct, evidence-backed dissenter."""
    return [
        SkepticVerdict("majority_1", SkepticStance.STANDS, "", 0.99, has_evidence=False),
        SkepticVerdict("majority_2", SkepticStance.STANDS, "", 0.99, has_evidence=False),
        SkepticVerdict("majority_3", SkepticStance.STANDS, "", 0.99, has_evidence=False),
        SkepticVerdict(
            "dissenter",
            SkepticStance.REVISE,
            "",
            0.9,
            has_evidence=True,
            proposed_payload={"bias": 0.0},
        ),
    ]


def run_ablation() -> dict[str, Any]:
    verdicts = _conformity_scenario()
    uniform = TrustGraph(weighting="uniform").aggregate(verdicts)
    trust = TrustGraph(weighting="trust").aggregate(verdicts)
    return {
        "scenario": "overconfident biased majority (3) vs. evidence-backed dissenter (1)",
        "uniform_decision": uniform.decision.value,
        "uniform_recovers_correct_output": uniform.decision is PanelDecision.REVISED,
        "trust_decision": trust.decision.value,
        "trust_recovers_correct_output": trust.decision is PanelDecision.REVISED,
        "trust_pruned": [p.agent_type for p in trust.pruned],
        "dissenter_pruned_uniform": any(p.agent_type == "dissenter" for p in uniform.pruned),
        "dissenter_pruned_trust": any(p.agent_type == "dissenter" for p in trust.pruned),
    }


async def run_madera_convergence() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for beta in (0.3, 0.5, 0.8):
        madera = MaderaEditor(
            FakeDiagnoser(),
            FakeRetriever(),
            DecayEditor(0.4),
            bias_field_scorer,
            threshold=0.1,
            max_rewrites=5,
        )
        result = await madera.repair({"bias": beta})
        rewrites = sum(1 for line in result.trail if line.startswith("rewrite"))
        rows.append({"beta": beta, "converged": result.converged, "rewrites": rewrites})
    return {"rows": rows}


async def run_end_to_end() -> dict[str, Any]:
    runner = InterventionRunner(
        madera=MaderaEditor(
            FakeDiagnoser(), FakeRetriever(), DecayEditor(0.3), bias_field_scorer, threshold=0.1
        )
    )
    machine = ControlMachine(CircuitBreaker(tau_enter=0.3, tau_exit=0.15), cooldown_steps=0)
    payload = {"bias": 0.8}

    before = await bias_field_scorer(payload)
    halted = machine.step(before, 0.0, payload=payload)
    ctx = InterventionContext(biased_node_count=1, total_node_count=5, max_node_magnitude=before)
    result = await runner.run(ctx, payload)
    assert result.repaired_payload is not None
    after = await bias_field_scorer(result.repaired_payload)
    recovered = machine.step(after, 0.0)
    return {
        "b_net_before": before,
        "halt_state": halted.state.value,
        "strategy": result.strategy,
        "b_net_after": after,
        "recovered_state": recovered.state.value,
    }


def make_plot(results: dict[str, Any], outdir: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    outdir.mkdir(parents=True, exist_ok=True)
    ab = results["ablation"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = ["uniform", "trust"]
    recovered = [
        1 if ab["uniform_recovers_correct_output"] else 0,
        1 if ab["trust_recovers_correct_output"] else 0,
    ]
    ax.bar(labels, recovered, color=["tab:gray", "tab:green"])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("recovers the correct (dissenting) output")
    ax.set_title("Trust-weighting vs. uniform on a conformity spiral")
    fig.tight_layout()
    fig.savefig(outdir / "trust-vs-uniform.png", dpi=140)
    plt.close(fig)
    return True


async def run_all() -> dict[str, Any]:
    return {
        "ablation": run_ablation(),
        "madera_convergence": await run_madera_convergence(),
        "end_to_end": await run_end_to_end(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 intervention study")
    parser.add_argument("--quick", action="store_true")  # accepted for parity; run is already fast
    parser.add_argument("--out", type=Path, default=Path("docs/studies"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    results = asyncio.run(run_all())
    results["runtime_seconds"] = time.perf_counter() - start

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "phase4-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plotted = False if args.no_plots else make_plot(results, args.out / "figures")

    print(json.dumps(results, indent=2))
    print(f"\nruntime: {results['runtime_seconds']:.2f}s  plot: {'yes' if plotted else 'skipped'}")


if __name__ == "__main__":
    main()
