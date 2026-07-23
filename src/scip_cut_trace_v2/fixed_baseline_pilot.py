"""Evaluate the pre-registered fixed-baseline train pilot gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TIE_ORDER = ("efficacy-rank", "adaptive-score", "random-rank")
MINIMUM_INTERVENTION_INSTANCES_V1 = 3
MINIMUM_INTERVENTION_INSTANCES_V2 = 5


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_pilot_gate(
    plan: dict[str, Any],
    causal_manifest: dict[str, Any],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    contract = plan["experiment_contract"]
    actions = tuple(contract["actions"])
    instance_ids = tuple(instance["instance_id"] for instance in plan["instances"])
    causal_instance_ids = tuple(
        instance["instance_id"] for instance in causal_manifest["per_instance"]
    )
    checks = {
        "plan_pre_registered": plan.get("status")
        == "pre_registered_before_active_outcomes",
        "actions_match": tuple(causal_manifest["actions"]) == actions,
        "seeds_match": tuple(causal_manifest["seeds"])
        == tuple(contract["seeds"]),
        "time_limit_matches": float(causal_manifest["time_limit"])
        == float(contract["time_limit_seconds"]),
        "node_limit_matches": causal_manifest.get("node_limit")
        == contract.get("node_limit"),
        "intervention_scope_matches": causal_manifest.get("intervention_scope")
        == contract["intervention_scope"],
        "instances_match": causal_instance_ids == instance_ids,
        "statistics_actions_match": set(statistics["actions"]) == set(actions),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Pilot artifacts violate the pre-registration: {failed}")

    expected_pairs = len(instance_ids) * len(contract["seeds"])
    plan_schema_version = int(plan.get("schema_version", 1))
    minimum_intervention_instances = (
        MINIMUM_INTERVENTION_INSTANCES_V2
        if plan_schema_version >= 2
        else MINIMUM_INTERVENTION_INSTANCES_V1
    )
    action_results = {}
    for action in actions:
        result = statistics["actions"][action]
        intervention_instances = sum(
            instance["intervention_pairs"] > 0 for instance in result["per_instance"]
        )
        performance = result["penalized_time"]
        action_checks = {
            "all_native_arms_complete": result["native_complete_pairs"]
            == expected_pairs,
            "zero_instance_safety_failures": result["safety_failure_instances"]
            == 0,
            f"selected_set_changes_on_at_least_{minimum_intervention_instances}_instances": (
                intervention_instances >= minimum_intervention_instances
            ),
            "more_instance_wins_than_losses": (
                performance["instance_wins"] > performance["instance_losses"]
            ),
        }
        if plan_schema_version >= 2:
            interval = performance["cluster_bootstrap_interval"]
            action_checks["penalized_time_ci95_upper_below_one"] = (
                interval["upper"] is not None and interval["upper"] < 1.0
            )
        else:
            action_checks["penalized_time_point_ratio_below_one"] = (
                performance["geometric_mean_ratio_treatment_over_native"] is not None
                and performance["geometric_mean_ratio_treatment_over_native"] < 1.0
            )
        action_results[action] = {
            "passed": all(action_checks.values()),
            "checks": action_checks,
            "expected_native_complete_pairs": expected_pairs,
            "observed_native_complete_pairs": result["native_complete_pairs"],
            "intervention_instances": intervention_instances,
            "safety_failure_instances": result["safety_failure_instances"],
            "penalized_time_ratio": performance[
                "geometric_mean_ratio_treatment_over_native"
            ],
            "instance_wins": performance["instance_wins"],
            "instance_losses": performance["instance_losses"],
        }

    passing = [action for action in actions if action_results[action]["passed"]]
    order = {action: index for index, action in enumerate(TIE_ORDER)}
    selected = (
        min(
            passing,
            key=lambda action: (
                action_results[action]["penalized_time_ratio"],
                order[action],
            ),
        )
        if passing
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "machine-readable application of the pre-registered fixed-baseline train pilot gate",
        "artifact_contract_checks": checks,
        "plan_schema_version": plan_schema_version,
        "minimum_intervention_instances": minimum_intervention_instances,
        "tie_order": list(TIE_ORDER),
        "actions": action_results,
        "selected_action_for_larger_train_cohort": selected,
        "passed": selected is not None,
        "decision": (
            f"advance {selected} to a separately pre-registered larger train cohort"
            if selected is not None
            else "no fixed baseline advances; do not enlarge or tune these actions on pilot outcomes"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--causal-manifest", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = (args.plan, args.causal_manifest, args.statistics)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    decision = evaluate_pilot_gate(*payloads)
    decision["sources"] = {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in zip(("plan", "causal_manifest", "statistics"), paths)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": decision["passed"],
                "selected_action": decision[
                    "selected_action_for_larger_train_cohort"
                ],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
