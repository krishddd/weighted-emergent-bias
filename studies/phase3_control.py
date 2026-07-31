"""Phase 3 control study -- the WP7 deliverable.

Shows the control plane behaving as designed: hysteresis prevents the thrashing a binary threshold
would produce, the state machine recovers or escalates, and calibration hits a target false-halt
rate. Emits JSON + figures; every number in ``docs/studies/phase3-control.md`` comes from here.

    python studies/phase3_control.py            # full run
    python studies/phase3_control.py --quick     # fast run

Experiments
-----------
1. **Hysteresis vs. binary** -- a B_net trajectory oscillating near the threshold: count how many
   times each controller enters Intervention.
2. **Lifecycle** -- a spike that clears (recovers) and a persistent spike (escalates).
3. **Calibration** -- calibrate on control noise, measure the achieved false-halt rate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weighted_emergent_bias import (
    BreakerState,
    CircuitBreaker,
    ControlMachine,
    calibrate_thresholds,
)

MASTER_SEED = 20260731
TAU_ENTER = 0.30
TAU_EXIT = 0.15


@dataclass(frozen=True)
class Config:
    calibration_n: int = 20000

    @staticmethod
    def quick() -> Config:
        return Config(calibration_n=2000)


def _near_boundary_trajectory() -> list[float]:
    """Rise past tau_enter, oscillate inside the [tau_exit, tau_enter] dead-band, then decay."""
    rise = [0.02, 0.08, 0.18, 0.32]
    oscillation = [0.30 + 0.06 * math.sin(i) for i in range(16)]  # wobbles around ~0.30
    decay = [0.20, 0.12, 0.05, 0.02]
    return rise + oscillation + decay


def _count_intervention_entries_hysteresis(traj: list[float]) -> int:
    breaker = CircuitBreaker(tau_enter=TAU_ENTER, tau_exit=TAU_EXIT)
    entries, prev_tripped = 0, False
    for b in traj:
        tripped = breaker.check(b, 0.0).state is BreakerState.INTERVENTION
        if tripped and not prev_tripped:
            entries += 1
        prev_tripped = tripped
    return entries


def _count_intervention_entries_binary(traj: list[float]) -> int:
    entries, prev = 0, False
    for b in traj:
        tripped = b >= TAU_ENTER
        if tripped and not prev:
            entries += 1
        prev = tripped
    return entries


def run_hysteresis_vs_binary() -> dict[str, Any]:
    traj = _near_boundary_trajectory()
    return {
        "trajectory": traj,
        "tau_enter": TAU_ENTER,
        "tau_exit": TAU_EXIT,
        "hysteresis_entries": _count_intervention_entries_hysteresis(traj),
        "binary_entries": _count_intervention_entries_binary(traj),
    }


def _run_machine(traj: list[float]) -> list[str]:
    machine = ControlMachine(
        CircuitBreaker(tau_enter=TAU_ENTER, tau_exit=TAU_EXIT), cooldown_steps=1, max_retries=2
    )
    return [machine.step(b, 0.0).state.value for b in traj]


def run_lifecycle() -> dict[str, Any]:
    # A spike that then clears -> should recover to normal.
    recovering = [0.05, 0.4, 0.4, 0.05, 0.05, 0.05]
    # A spike that never clears -> should escalate.
    persistent = [0.05, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    return {
        "recovering_states": _run_machine(recovering),
        "persistent_states": _run_machine(persistent),
    }


def run_calibration(cfg: Config) -> dict[str, Any]:
    rng = np.random.default_rng(MASTER_SEED)
    target = 0.02
    control = rng.uniform(0.0, 1.0, size=cfg.calibration_n).tolist()
    config = calibrate_thresholds(control, target_false_halt_rate=target)
    holdout = rng.uniform(0.0, 1.0, size=cfg.calibration_n)
    achieved = float((holdout > config.tau_enter).mean())
    return {
        "target_false_halt_rate": target,
        "tau_enter": config.tau_enter,
        "tau_exit": config.tau_exit,
        "achieved_false_halt_rate": achieved,
    }


def _pyplot() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def make_plots(results: dict[str, Any], outdir: Path) -> bool:
    plt = _pyplot()
    if plt is None:
        return False
    outdir.mkdir(parents=True, exist_ok=True)

    hv = results["hysteresis_vs_binary"]
    traj = hv["trajectory"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(traj, "o-", color="tab:blue", label="B_net")
    ax.axhline(hv["tau_enter"], ls="--", c="tab:red", label="tau_enter")
    ax.axhline(hv["tau_exit"], ls="--", c="tab:green", label="tau_exit")
    ax.axhspan(hv["tau_exit"], hv["tau_enter"], color="gray", alpha=0.12, label="dead-band")
    ax.set_xlabel("superstep")
    ax.set_ylabel("B_net")
    ax.set_title(
        f"Hysteresis enters once ({hv['hysteresis_entries']}x) vs. "
        f"binary thrash ({hv['binary_entries']}x)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "hysteresis-vs-binary.png", dpi=140)
    plt.close(fig)
    return True


def run_all(cfg: Config) -> dict[str, Any]:
    return {
        "hysteresis_vs_binary": run_hysteresis_vs_binary(),
        "lifecycle": run_lifecycle(),
        "calibration": run_calibration(cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 control study")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("docs/studies"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = Config.quick() if args.quick else Config()
    start = time.perf_counter()
    results = run_all(cfg)
    results["runtime_seconds"] = time.perf_counter() - start

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "phase3-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plotted = False if args.no_plots else make_plots(results, args.out / "figures")

    print(json.dumps(results, indent=2))
    print(f"\nruntime: {results['runtime_seconds']:.2f}s  plots: {'yes' if plotted else 'skipped'}")


if __name__ == "__main__":
    main()
