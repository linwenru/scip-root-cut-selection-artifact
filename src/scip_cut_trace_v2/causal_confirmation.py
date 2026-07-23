"""Evaluate a pre-registered single-action causal confirmation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
COMPLETE_STATUSES = frozenset(("optimal", "infeasible", "unbounded", "inforunbd"))
DEFAULT_PLAN = Path(
    "data/manifests/causal_first_run_efficacy_cohort_c_plan_v1.json"
)
DEFAULT_EXPERIMENT = Path(
    "data/manifests/causal_action_efficacy_first_run_cohort_c_v1.json"
)
DEFAULT_OUTPUT = Path(
    "data/manifests/causal_first_run_efficacy_cohort_c_confirmation_v1.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_contract(plan: dict[str, Any], experiment: dict[str, Any]) -> str:
    actions = plan["experiment_contract"]["actions"]
    if len(actions) != 1:
        raise ValueError("confirmation requires exactly one planned action")
    if experiment["actions"] != actions:
        raise ValueError("experiment actions do not match the pre-registered plan")
    contract = plan["experiment_contract"]
    if experiment["seeds"] != contract["seeds"]:
        raise ValueError("experiment seeds do not match the pre-registered plan")
    if experiment["time_limit"] != contract["time_limit_seconds"]:
        raise ValueError("experiment time limit does not match the pre-registered plan")
    if experiment["node_limit"] != contract["node_limit"]:
        raise ValueError("experiment node limit does not match the pre-registered plan")
    if experiment["intervention_scope"] != contract["intervention_scope"]:
        raise ValueError("experiment intervention scope does not match the plan")
    planned_instances = [entry["instance_id"] for entry in plan["instances"]]
    observed_instances = [entry["instance_id"] for entry in experiment["per_instance"]]
    if observed_instances != planned_instances:
        raise ValueError("experiment instances or ordering do not match the plan")
    return actions[0]


def evaluate_confirmation(
    plan: dict[str, Any], experiment: dict[str, Any]
) -> dict[str, Any]:
    action = _validate_contract(plan, experiment)
    planned_seeds = plan["experiment_contract"]["seeds"]
    attributable_pairs = 0
    native_incomplete_pairs = 0
    eligible_attributable_pairs = 0
    valid_eligible_pairs = 0
    attributable_unsafe = []
    total_contexts = 0
    total_interventions = 0
    per_instance = []

    for instance in experiment["per_instance"]:
        pairs = instance["pairs"]
        if [pair["seed"] for pair in pairs] != planned_seeds:
            raise ValueError(f"seed mismatch for {instance['instance_id']}")
        valid_savings = []
        eligible_pairs = 0
        ineligible_attributable_pairs = 0
        unsafe_pairs = 0
        for pair in pairs:
            result = pair["actions"][action]
            comparison = result["comparison"]
            selector = result["selector"]
            contexts = selector.get("context_records", [])
            interventions = int(selector["interventions"])
            total_contexts += len(contexts)
            total_interventions += interventions
            if len(contexts) != 1:
                raise ValueError(
                    f"expected one context for {instance['instance_id']} seed {pair['seed']}"
                )
            if interventions not in (0, 1):
                raise ValueError("first-run confirmation observed multiple interventions")
            if not pair["initial_context"]["matching_across_actions"]:
                raise ValueError("pre-intervention contexts did not match")

            native_complete = pair["native_outcome"]["status"] in COMPLETE_STATUSES
            treatment_complete = result["outcome"]["status"] in COMPLETE_STATUSES
            eligible = bool(comparison["eligible"])
            eligible_pairs += int(eligible)
            if not native_complete:
                native_incomplete_pairs += 1
                continue
            attributable_pairs += 1
            if not eligible:
                ineligible_attributable_pairs += 1
                continue
            eligible_attributable_pairs += 1
            if not treatment_complete:
                unsafe_pairs += 1
                attributable_unsafe.append(
                    {
                        "instance_id": instance["instance_id"],
                        "seed": pair["seed"],
                        "native_status": pair["native_outcome"]["status"],
                        "treatment_status": result["outcome"]["status"],
                        "native_lp_iterations": pair["native_outcome"]["lp_iterations"],
                        "treatment_lp_iterations": result["outcome"]["lp_iterations"],
                    }
                )
                continue
            if not comparison["valid"]:
                raise ValueError("completed attributable treatment pair is not valid")
            valid_eligible_pairs += 1
            valid_savings.append(
                float(
                    comparison["metrics"]["lp_iterations"]["relative_saving"]
                )
            )

        mean_saving = sum(valid_savings) / len(valid_savings) if valid_savings else 0.0
        per_instance.append(
            {
                "instance_id": instance["instance_id"],
                "eligible_pairs": eligible_pairs,
                "ineligible_attributable_pairs": ineligible_attributable_pairs,
                "valid_eligible_pairs": len(valid_savings),
                "attributable_unsafe_pairs": unsafe_pairs,
                "mean_relative_lp_iteration_saving_valid_eligible": mean_saving,
                "fixed_action_outcome": (
                    "win" if mean_saving > 0.0 else "loss" if mean_saving < 0.0 else "tie"
                ),
            }
        )

    mean_saving = sum(
        entry["mean_relative_lp_iteration_saving_valid_eligible"]
        for entry in per_instance
    ) / len(per_instance)
    wins = sum(entry["fixed_action_outcome"] == "win" for entry in per_instance)
    losses = sum(entry["fixed_action_outcome"] == "loss" for entry in per_instance)
    ties = sum(entry["fixed_action_outcome"] == "tie" for entry in per_instance)
    worst = min(
        entry["mean_relative_lp_iteration_saving_valid_eligible"]
        for entry in per_instance
    )
    minimum_pairs = int(plan["confirmation_gate"]["minimum_attributable_pairs"])
    checks = {
        "minimum_attributable_pairs_met": attributable_pairs >= minimum_pairs,
        "zero_attributable_unsafe_treatment_pairs": not attributable_unsafe,
        "instance_equal_mean_relative_lp_iteration_saving_positive": mean_saving > 0.0,
        "more_instance_wins_than_losses": wins > losses,
        "no_instance_below_minus_0_25": worst >= -0.25,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "action": action,
        "attribution_contract": (
            "only pairs with a completed native arm can attribute treatment safety; "
            "benefit uses completed eligible pairs and ineligible instances tie at native"
        ),
        "data": {
            "instances": len(per_instance),
            "planned_pairs": len(per_instance) * len(planned_seeds),
            "attributable_pairs": attributable_pairs,
            "native_incomplete_pairs": native_incomplete_pairs,
            "eligible_attributable_pairs": eligible_attributable_pairs,
            "valid_eligible_pairs": valid_eligible_pairs,
            "attributable_unsafe_pairs": len(attributable_unsafe),
            "contexts": total_contexts,
            "interventions": total_interventions,
        },
        "fixed_action_summary": {
            "instance_equal_mean_relative_lp_iteration_saving": mean_saving,
            "instance_wins": wins,
            "instance_losses": losses,
            "instance_ties": ties,
            "worst_instance_mean_relative_lp_iteration_saving": worst,
        },
        "attributable_unsafe": attributable_unsafe,
        "per_instance": per_instance,
        "confirmation_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "consequence": (
                plan["confirmation_gate"]["passing_consequence"]
                if all(checks.values())
                else plan["confirmation_gate"]["failure_consequence"]
            ),
        },
    }


def run_confirmation(plan_path: Path, experiment_path: Path, output_path: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    experiment_path = experiment_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    result = evaluate_confirmation(plan, experiment)
    result.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan": str(plan_path),
            "plan_sha256": _sha256_file(plan_path),
            "experiment": str(experiment_path),
            "experiment_sha256": _sha256_file(experiment_path),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_confirmation(args.plan, args.experiment, args.output)
    print(json.dumps(result["confirmation_gate"], sort_keys=True))
    summary = result["fixed_action_summary"]
    print(
        f"single_action_confirmation: attributable={result['data']['attributable_pairs']} "
        f"unsafe={result['data']['attributable_unsafe_pairs']} "
        f"mean_relative_lp_saving="
        f"{summary['instance_equal_mean_relative_lp_iteration_saving']:.6f} "
        f"wins/ties/losses={summary['instance_wins']}/"
        f"{summary['instance_ties']}/{summary['instance_losses']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
