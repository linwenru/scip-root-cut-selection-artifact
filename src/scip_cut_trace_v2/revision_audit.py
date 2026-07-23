"""Build the deterministic evidence audit used for the major revision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .observational import PROJECT_ROOT


SCHEMA_VERSION = 2
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
DEFAULT_ACTIVE_PLAN = MANIFEST_DIR / "publication_active_plan_v1.json"
DEFAULT_ROOT_MANIFEST = MANIFEST_DIR / "root_observational_v1.json"
DEFAULT_ONLINE_DATASET = MANIFEST_DIR / "ranking_imitation_online_v1.json"
DEFAULT_ONLINE_MODEL = MANIFEST_DIR / "ranking_imitation_online_xgb_v1.json"
DEFAULT_PARITY = MANIFEST_DIR / "causal_publication_parity_v1.json"
DEFAULT_FIXED_RESULTS = MANIFEST_DIR / "causal_publication_fixed_baselines_v1.json"
DEFAULT_ITT = MANIFEST_DIR / "paper_statistics_publication_fixed_baselines_itt_v2.json"
DEFAULT_SMOKE = MANIFEST_DIR / "causal_online_imitation_smoke_v1.json"
DEFAULT_REVISION_MODEL = (
    MANIFEST_DIR / "ranking_imitation_online_xgb_revision_v1.json"
)
DEFAULT_PRECISION_PLAN = MANIFEST_DIR / "major_revision_precision_plan_v1.json"
DEFAULT_SHADOW_OVERHEAD = (
    MANIFEST_DIR / "major_revision_shadow_overhead_statistics_v1.json"
)
DEFAULT_ASSIGNMENTS = (
    MANIFEST_DIR / "evaluation_protocols" / "instance_assignments.csv"
)
DEFAULT_OUTPUT = MANIFEST_DIR / "major_revision_audit_v1.json"
DEFAULT_COHORT_CSV = PROJECT_ROOT / "paper" / "supplementary_publication_cohort.csv"

DEVELOPMENT_MANIFESTS = {
    "action_oracle_pilot": MANIFEST_DIR / "causal_action_oracle_pilot_v1.json",
    "cohort_a": MANIFEST_DIR / "causal_action_oracle_first_run_cohort_a_v1.json",
    "cohort_b": MANIFEST_DIR / "causal_action_oracle_first_run_cohort_b_v1.json",
    "cohort_c": MANIFEST_DIR / "causal_action_efficacy_first_run_cohort_c_v1.json",
    "fixed_baseline_train_pilot": (
        MANIFEST_DIR / "causal_fixed_baseline_train_pilot_v1.json"
    ),
    "fixed_baseline_disjoint_pilot": (
        MANIFEST_DIR / "causal_fixed_baseline_disjoint_train_pilot_v2.json"
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _read_assignments(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _instance_ids(manifest: dict[str, Any]) -> set[str]:
    return {
        str(record["instance_id"])
        for record in manifest.get("per_instance", ())
    }


def _group_summary(
    instance_names: Iterable[str], assignment_by_name: dict[str, dict[str, str]]
) -> dict[str, Any]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for instance_name in sorted(instance_names):
        group = assignment_by_name[instance_name]["official_group"]
        by_group[group].append(instance_name)
    repeated = {
        group: names for group, names in sorted(by_group.items()) if len(names) > 1
    }
    return {
        "instances": sum(len(names) for names in by_group.values()),
        "official_groups": len(by_group),
        "repeated_groups": repeated,
    }


def _cohort_overlap(
    active_instances: list[dict[str, Any]],
    development_manifests: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queue_ids = {
        name: _instance_ids(manifest)
        for name, manifest in development_manifests.items()
    }
    active_ids = {str(record["instance_id"]) for record in active_instances}
    overlap_by_queue = {
        name: sorted(active_ids & instance_ids)
        for name, instance_ids in queue_ids.items()
    }
    previously_evaluated = set().union(*queue_ids.values()) & active_ids
    cohort_rows = []
    for record in active_instances:
        instance_id = str(record["instance_id"])
        prior_queues = sorted(
            name for name, instance_ids in queue_ids.items() if instance_id in instance_ids
        )
        cohort_rows.append(
            {
                "instance_id": instance_id,
                "instance_name": record["instance_name"],
                "official_group": record.get("official_group") or "",
                "evaluation_stratum": record["evaluation_stratum"],
                "sampling_stratum": record.get("sampling_stratum", ""),
                "source_trace_elapsed_seconds": record[
                    "source_trace_elapsed_seconds"
                ],
                "previously_evaluated_in_development": bool(prior_queues),
                "prior_development_queues": ";".join(prior_queues),
            }
        )
    return (
        {
            "active_instances": len(active_ids),
            "overlap_by_queue": {
                name: {"count": len(ids), "instance_ids": ids}
                for name, ids in overlap_by_queue.items()
            },
            "unique_instances_previously_evaluated": len(previously_evaluated),
            "unique_previously_evaluated_instance_ids": sorted(previously_evaluated),
            "previously_unevaluated_instances": len(active_ids - previously_evaluated),
            "previously_unevaluated_instance_ids": sorted(
                active_ids - previously_evaluated
            ),
            "inference": (
                "The active cohort is a selected training-split development cohort, "
                "not an independent generalization cohort."
            ),
        },
        cohort_rows,
    )


def _annotate_sampling_strata(
    active_instances: list[dict[str, Any]], strata: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    annotated = []
    offset = 0
    for stratum in strata:
        count = int(stratum["selected_instances"])
        for record in active_instances[offset : offset + count]:
            annotated.append({**record, "sampling_stratum": stratum["name"]})
        offset += count
    if offset != len(active_instances):
        raise ValueError(
            "Sampling-stratum counts do not cover the active publication cohort"
        )
    return annotated


def _parity_audit(parity: dict[str, Any]) -> dict[str, Any]:
    exceptions = []
    for instance in parity["per_instance"]:
        for pair in instance["pairs"]:
            if pair["classification"] == "passed":
                continue
            exceptions.append(
                {
                    "instance_id": instance["instance_id"],
                    "seed": pair["seed"],
                    "candidate_arm": pair["candidate_arm"],
                    "classification": pair["classification"],
                    "native_status": pair["native_outcome"]["status"],
                    "candidate_status": pair["candidate_outcome"]["status"],
                }
            )
    return {
        "predeclared_strict_all_pair_gate_passed": bool(parity["passed"]),
        "conditional_complete_callback_evidence_passed": bool(
            parity["diagnostic_complete_evidence_passed"]
        ),
        "pairs_per_arm": {
            arm: summary["pairs"] for arm, summary in parity["arm_summary"].items()
        },
        "complete_callback_exercised_pairs_per_arm": {
            arm: summary["complete_callback_exercised_pairs"]
            for arm, summary in parity["arm_summary"].items()
        },
        "exceptions": exceptions,
        "manuscript_status": (
            "conditional diagnostic parity only; the strict predeclared gate failed"
        ),
    }


def _itt_audit(analysis: dict[str, Any]) -> dict[str, Any]:
    actions = {}
    for action, result in analysis["actions"].items():
        penalized = result["penalized_time"]
        interval = penalized["cluster_bootstrap_interval"]
        safety = result["conditional_policy_safety"]
        actions[action] = {
            "instances": penalized["instances"],
            "pairs": penalized["pairs"],
            "ratio_treatment_over_native": penalized[
                "geometric_mean_ratio_treatment_over_native"
            ],
            "ci95": [interval["lower"], interval["upper"]],
            "instance_wins_ties_losses": [
                penalized["instance_wins"],
                penalized["instance_ties"],
                penalized["instance_losses"],
            ],
            "intervention_pairs": result["intervention_pairs"],
            "policy_fallback_pairs": result["policy_fallback_pairs"],
            "outcome_pair_counts": result["outcome_pair_counts"],
            "conditional_policy_safety": safety,
            "gate_passed": result["gate"]["passed"],
            "native_complete_secondary_ratio": result[
                "native_complete_secondary"
            ]["geometric_mean_ratio_treatment_over_native"],
        }
    return {
        "primary_estimand": analysis["analysis_contract"]["primary_metric"],
        "primary_time_field": analysis["analysis_contract"][
            "primary_time_field"
        ],
        "full_arm_wall_time_sensitivity_available": analysis[
            "analysis_contract"
        ]["full_arm_wall_time_sensitivity_available"],
        "full_arm_wall_time_limitation": analysis["analysis_contract"][
            "full_arm_wall_time_limitation"
        ],
        "actions": actions,
        "all_predeclared_blocks_retained": all(
            action["instances"] == 40 and action["pairs"] == 120
            for action in actions.values()
        ),
    }


def build_revision_audit(
    *,
    active_plan_path: Path = DEFAULT_ACTIVE_PLAN,
    root_manifest_path: Path = DEFAULT_ROOT_MANIFEST,
    online_dataset_path: Path = DEFAULT_ONLINE_DATASET,
    online_model_path: Path = DEFAULT_ONLINE_MODEL,
    parity_path: Path = DEFAULT_PARITY,
    fixed_results_path: Path = DEFAULT_FIXED_RESULTS,
    itt_path: Path = DEFAULT_ITT,
    smoke_path: Path = DEFAULT_SMOKE,
    revision_model_path: Path = DEFAULT_REVISION_MODEL,
    precision_plan_path: Path = DEFAULT_PRECISION_PLAN,
    shadow_overhead_path: Path = DEFAULT_SHADOW_OVERHEAD,
    assignments_path: Path = DEFAULT_ASSIGNMENTS,
    development_paths: dict[str, Path] = DEVELOPMENT_MANIFESTS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    active_plan = _load_json(active_plan_path)
    root_manifest = _load_json(root_manifest_path)
    online_dataset = _load_json(online_dataset_path)
    online_model = _load_json(online_model_path)
    parity = _load_json(parity_path)
    fixed_results = _load_json(fixed_results_path)
    itt = _load_json(itt_path)
    smoke = _load_json(smoke_path)
    revision_model = _load_json(revision_model_path)
    precision_plan = _load_json(precision_plan_path)
    shadow_overhead = _load_json(shadow_overhead_path)
    development = {
        name: _load_json(path) for name, path in development_paths.items()
    }
    assignments = _read_assignments(assignments_path)
    assignment_by_name = {row["instance_name"]: row for row in assignments}

    active_experiment = active_plan["active_fixed_baselines"]
    active_instances = _annotate_sampling_strata(
        active_experiment["instances"], active_experiment["sampling_strata"]
    )
    overlap, cohort_rows = _cohort_overlap(active_instances, development)
    totals = root_manifest["totals"]
    train_matrix = online_dataset["matrices"]["train"]
    assigned_group_ood_val = [
        row
        for row in assignments
        if row["original_split"] == "val"
        and row["evaluation_stratum"] == "unseen_family"
    ]
    evaluated_group_ood_names = set(
        online_model["evaluation"]["official_group_ood_val"]["per_instance"]
    )
    learned_action = smoke["action_summary"]["xgb-imitation-rank"]

    source_paths = {
        "active_plan": active_plan_path,
        "root_observational": root_manifest_path,
        "online_dataset": online_dataset_path,
        "online_model": online_model_path,
        "parity": parity_path,
        "fixed_results": fixed_results_path,
        "itt_reanalysis": itt_path,
        "learned_smoke": smoke_path,
        "post_review_model": revision_model_path,
        "precision_plan": precision_plan_path,
        "shadow_overhead": shadow_overhead_path,
        "assignments": assignments_path,
        **{
            f"development_{name}": path
            for name, path in development_paths.items()
        },
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post-review diagnostic audit; not a preregistration",
        "source_artifacts": {
            name: _source_record(path) for name, path in source_paths.items()
        },
        "observational_scope": {
            "source_root_rows": totals["source_root_rows"],
            "optional_root_candidate_occurrences": totals[
                "root_candidate_occurrences"
            ],
            "forced_root_rows": totals["root_forced_rows"],
            "root_row_accounting_closes": (
                totals["source_root_rows"]
                == totals["root_candidate_occurrences"]
                + totals["root_forced_rows"]
            ),
            "online_actionable_definition": [
                "first not-yet-used root decision in a SCIP run",
                "pre-action LP state is present",
                "at least two logical optional candidates",
                "no ambiguous source cut_id collision",
            ],
            "online_actionable_decisions": totals["policy_eligible_decisions"],
            "online_actionable_candidates": totals["policy_eligible_candidates"],
            "offline_pairwise_informative_definition": (
                "post-action native labels contain at least one applied and one "
                "non-applied candidate"
            ),
            "train_actionable_queries": train_matrix["groups"],
            "train_pairwise_informative_queries": train_matrix[
                "effective_pair_groups"
            ],
            "train_all_applied_queries_excluded_from_pairwise_fit": (
                train_matrix["groups"] - train_matrix["effective_pair_groups"]
            ),
            "interpretation": (
                "Post-action labels define metric/training informativeness only; "
                "they are not part of the online actionability condition."
            ),
        },
        "validation_grouping": {
            "assigned_official_group_ood": _group_summary(
                (row["instance_name"] for row in assigned_group_ood_val),
                assignment_by_name,
            ),
            "evaluated_eligible_official_group_ood": _group_summary(
                evaluated_group_ood_names, assignment_by_name
            ),
            "bootstrap_issue": (
                "The published interval resampled evaluated instances, while official "
                "Group is the highest shared clustering level."
            ),
        },
        "model_selection": {
            "model_kind": "XGBoost rank:ndcg imitation model",
            "label": "native SCIP cut application",
            "best_iteration": online_model["model"]["best_iteration"],
            "early_stopping_dataset": "official_group_ood_val",
            "early_stopping_metric": online_model["training"]["parameters"][
                "eval_metric"
            ],
            "same_dataset_used_for_reported_interval": True,
            "stage_gate_passed": online_model["stage_gate"]["passed"],
            "required_revision": (
                "select the boosting round inside training groups, freeze it, then "
                "evaluate once with Group-level resampling"
            ),
        },
        "learned_policy_online_evidence": {
            "publication_fixed_actions": fixed_results["actions"],
            "learned_action_in_publication_experiment": (
                "xgb-imitation-rank" in fixed_results["actions"]
            ),
            "smoke_test_instances": smoke["instances"],
            "smoke_test_eligible_pairs": learned_action["eligible_pairs"],
            "smoke_test_policy_fallback_pairs": learned_action[
                "policy_fallback_pairs"
            ],
            "complete_solve_effect_established": False,
            "inference": (
                "The learned ranker has integration evidence only and has not received "
                "an independent complete-solve efficacy test."
            ),
        },
        "post_review_model_selection": {
            "boosting_rounds": revision_model["model"]["frozen_boost_rounds"],
            "round_selection_external_validation_used": revision_model[
                "training"
            ]["round_selection"]["external_validation_used"],
            "primary_offline_metric": revision_model["inference_contract"][
                "primary_offline_metric"
            ],
            "bootstrap_cluster": revision_model["inference_contract"][
                "bootstrap_cluster"
            ],
            "official_group_ood_selection_overlap": revision_model[
                "evaluation"
            ]["official_group_ood_val"]["comparison"]["selection_overlap"],
            "seen_group_selection_overlap": revision_model["evaluation"][
                "seen_family_val"
            ]["comparison"]["selection_overlap"],
            "stage_gate": revision_model["stage_gate"],
        },
        "parity": _parity_audit(parity),
        "publication_cohort_overlap": overlap,
        "publication_cohort_grouping": {
            "instances": len(active_instances),
            "distinct_group_keys": len(
                {record["group_key"] for record in active_instances}
            ),
            "instances_with_official_group": sum(
                bool(record.get("official_group")) for record in active_instances
            ),
            "distinct_nonempty_official_groups": len(
                {
                    record["official_group"]
                    for record in active_instances
                    if record.get("official_group")
                }
            ),
            "officially_ungrouped_instances": sum(
                not record.get("official_group") for record in active_instances
            ),
            "wording": (
                "40 distinct group keys: 30 nonempty official Groups and 10 "
                "instance-specific keys for officially ungrouped cases"
            ),
        },
        "fixed_policy_intention_to_treat": _itt_audit(itt),
        "prospective_precision": {
            "available_test": precision_plan["available_sealed_test"],
            "recommended_future_design": precision_plan[
                "recommended_future_design"
            ],
            "decision": precision_plan["decision"],
        },
        "same_path_shadow_overhead": {
            "instances": shadow_overhead["instances"],
            "pairs": shadow_overhead["pairs"],
            "structural_gate_passed": shadow_overhead[
                "structural_gate_passed"
            ],
            "policy_pair_structural_matches": shadow_overhead[
                "structural_matching_pairs"
            ],
            "full_path_structural_matches": shadow_overhead[
                "full_path_structural_matching_pairs"
            ],
            "shadow_evaluation_pairs": shadow_overhead[
                "shadow_evaluation_pairs"
            ],
            "proposed_selected_set_change_pairs": shadow_overhead[
                "proposed_selected_set_change_pairs"
            ],
            "itt_par2": shadow_overhead["itt_par2"],
            "full_path_exposure_sensitivity": shadow_overhead[
                "full_path_exposure_sensitivity"
            ],
            "analysis_contract": shadow_overhead["analysis_contract"],
            "timing_seconds": shadow_overhead["timing_seconds"],
            "inference": shadow_overhead["inference"],
        },
        "revision_decisions": {
            "current_generalization_claim_supported": False,
            "current_reproducible_claim_supported": False,
            "current_learned_complete_solve_claim_supported": False,
            "existing_fixed_policy_result": (
                "No fixed policy passes the predeclared joint gate under full-block ITT; "
                "the intervals remain compatible with benefit and harm."
            ),
        },
    }
    return audit, cohort_rows


def write_cohort_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort-csv", type=Path, default=DEFAULT_COHORT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit, cohort_rows = build_revision_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_cohort_csv(args.cohort_csv, cohort_rows)
    summary = {
        "output": str(args.output),
        "cohort_csv": str(args.cohort_csv),
        "strict_parity_passed": audit["parity"][
            "predeclared_strict_all_pair_gate_passed"
        ],
        "active_cohort_prior_overlap": audit["publication_cohort_overlap"][
            "unique_instances_previously_evaluated"
        ],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
