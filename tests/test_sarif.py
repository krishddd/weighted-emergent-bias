"""SARIF export tests, including validation against the bundled SARIF 2.1.0 schema.

The schema-validation test is what lets the project honestly claim "SARIF 2.1.0" rather than
"SARIF-inspired" (DESIGN §2c).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from weighted_emergent_bias import (
    AuditTrail,
    BiasScore,
    BreakerAction,
    BreakerDecision,
    BreakerState,
    DivergenceMethod,
    TaskMode,
    to_sarif,
)

_SCHEMA = Path(__file__).parent / "fixtures" / "sarif-2.1.0.schema.json"


def _populated_trail() -> AuditTrail:
    trail = AuditTrail()
    trail.record_score(
        BiasScore(
            effect_size=5.0,
            ci_low=3.0,
            ci_high=7.0,
            p_value=0.001,
            method=DivergenceMethod.JENSEN_SHANNON,
            task_mode=TaskMode.CHOICE,
            n_samples=8,
            raw_divergence=0.6,
            null_mean=0.1,
            null_std=0.1,
            axis="gender",
        ),
        node="router",
    )
    trail.record_decision(
        BreakerDecision(BreakerState.INTERVENTION, BreakerAction.REROUTE, 0.7, 0.6, "spike"),
        node="router",
    )
    return trail


class TestSarifStructure:
    def test_version_and_schema(self) -> None:
        doc = to_sarif(AuditTrail())
        assert doc["version"] == "2.1.0"
        assert doc["$schema"].endswith("sarif-2.1.0.json")
        assert doc["runs"][0]["tool"]["driver"]["name"] == "weighted-emergent-bias"

    def test_results_and_levels(self) -> None:
        doc = to_sarif(_populated_trail())
        results = doc["runs"][0]["results"]
        assert {r["ruleId"] for r in results} == {"bias/gender", "breaker/reroute"}
        levels = {r["ruleId"]: r["level"] for r in results}
        assert levels["bias/gender"] == "warning"  # significant
        assert levels["breaker/reroute"] == "error"

    def test_proceed_decisions_are_not_findings(self) -> None:
        trail = AuditTrail()
        trail.record_decision(
            BreakerDecision(BreakerState.NORMAL, BreakerAction.PROCEED, 0.0, 0.1, "ok")
        )
        assert to_sarif(trail)["runs"][0]["results"] == []

    def test_rules_cover_referenced_rule_ids(self) -> None:
        doc = to_sarif(_populated_trail())
        rule_ids = {r["id"] for r in doc["runs"][0]["tool"]["driver"]["rules"]}
        result_ids = {r["ruleId"] for r in doc["runs"][0]["results"]}
        assert result_ids <= rule_ids

    def test_location_names_the_node(self) -> None:
        doc = to_sarif(_populated_trail())
        loc = doc["runs"][0]["results"][0]["locations"][0]["logicalLocations"][0]
        assert loc["name"] == "router"


class TestSchemaValidation:
    def test_export_validates_against_sarif_2_1_0_schema(self) -> None:
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
        jsonschema.validate(to_sarif(_populated_trail()), schema)  # raises if invalid

    def test_empty_trail_also_validates(self) -> None:
        jsonschema = pytest.importorskip("jsonschema")
        schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
        jsonschema.validate(to_sarif(AuditTrail()), schema)
