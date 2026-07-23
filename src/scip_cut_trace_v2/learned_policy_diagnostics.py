"""Produce non-gating diagnostics for the frozen learned-policy pilot."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .causal_harness import COMPLETE_STATUSES
from .learned_policy_pilot import (
    ACTIVE_ARM,
    DEFAULT_PLAN,
    DEFAULT_RESULT,
    PROJECT_ROOT,
    SHADOW_ARM,
    _load,
    _manifest_path,
    _penalized,
    _sha256_file,
)
from .miplib_collection_metadata import parse_collection_html


SCHEMA_VERSION = 1
DEFAULT_FROZEN_STATISTICS = (
    PROJECT_ROOT / "data" / "manifests" / "learned_policy_pilot_statistics_v1.json"
)
DEFAULT_COLLECTION_HTML = (
    PROJECT_ROOT / "vendor" / "miplib2017_collection_snapshot" / "set_collection.html"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "manifests" / "learned_policy_pilot_diagnostics_v1.json"
)
MECHANICAL_CHECKS = (
    "arm_order",
    "same_instance_sha256",
    "same_seed",
    "same_parameters",
    "same_runtime_versions",
    "same_objective_sense",
    "known_intervention_scope",
    "run_budget_respected",
    "one_record_per_intervention",
    "context_budget_respected",
    "interventions_have_context",
)


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def _parse_objective(value: str) -> float | None:
    normalized = value.strip().removesuffix("*")
    try:
        return float(normalized)
    except ValueError:
        return None


def _official_objectives(collection_html: Path) -> dict[str, float]:
    return {
        row["instance"]: objective
        for row in parse_collection_html(collection_html)
        for objective in (_parse_objective(row["objective"]),)
        if objective is not None
    }


def _geometric_mean_ratio(log_ratios: Iterable[float]) -> float | None:
    values = tuple(log_ratios)
    return math.exp(sum(values) / len(values)) if values else None


def _arm_ratio_logs(
    result: dict[str, Any],
    numerator_arm: str,
    denominator_arm: str,
    field: str,
    time_limit: float,
) -> list[float]:
    group_logs = []
    for instance in result["per_instance"]:
        seed_logs = []
        for pair in instance["pairs"]:
            numerator = (
                pair["native_outcome"]
                if numerator_arm == "native"
                else pair["actions"][numerator_arm]["outcome"]
            )
            denominator = (
                pair["native_outcome"]
                if denominator_arm == "native"
                else pair["actions"][denominator_arm]["outcome"]
            )
            seed_logs.append(
                math.log(
                    _penalized(numerator, field, time_limit)
                    / _penalized(denominator, field, time_limit)
                )
            )
        group_logs.append(sum(seed_logs) / len(seed_logs))
    return group_logs


def _median(values: Iterable[float]) -> float | None:
    materialized = tuple(values)
    return float(statistics.median(materialized)) if materialized else None


def build_diagnostics(
    result_path: Path,
    plan_path: Path,
    frozen_statistics_path: Path,
    collection_html: Path,
) -> dict[str, Any]:
    result = _load(result_path)
    plan = _load(plan_path)
    frozen = _load(frozen_statistics_path)
    if frozen["source_result_sha256"] != _sha256_file(result_path):
        raise ValueError("Frozen statistics do not identify the supplied result")
    if frozen["source_plan_sha256"] != _sha256_file(plan_path):
        raise ValueError("Frozen statistics do not identify the supplied plan")
    if frozen["passed"]:
        raise ValueError("Post-hoc failure diagnostics require a frozen No-Go result")

    time_limit = float(plan["experiment"]["time_limit_seconds"])
    official_objectives = _official_objectives(collection_html)
    status_counts = {
        arm: Counter() for arm in ("native", SHADOW_ARM, ACTIVE_ARM)
    }
    shadow_classes = Counter()
    shadow_failed_checks = Counter()
    completed_shadow_pairs = 0
    completed_shadow_full_matches = 0
    active_intervention_pairs = 0
    active_fallback_pairs = 0
    mechanical_mismatch_pairs = []
    objective_mismatch_pairs = []
    native_complete_active_incomplete = []
    active_complete_native_incomplete = []
    both_incomplete_same_status = []
    context_observed_pairs = 0
    context_mismatch_pairs = 0
    arm_overheads: dict[str, list[float]] = {
        arm: [] for arm in ("native", SHADOW_ARM, ACTIVE_ARM)
    }
    model_load_times: dict[str, list[float]] = {
        SHADOW_ARM: [],
        ACTIVE_ARM: [],
    }
    policy_compute_times: dict[str, list[float]] = {
        SHADOW_ARM: [],
        ACTIVE_ARM: [],
    }

    for instance in result["per_instance"]:
        instance_id = instance["instance_id"]
        for pair in instance["pairs"]:
            seed = int(pair["seed"])
            native = pair["native_outcome"]
            shadow_record = pair["actions"][SHADOW_ARM]
            active_record = pair["actions"][ACTIVE_ARM]
            shadow = shadow_record["outcome"]
            active = active_record["outcome"]
            outcomes = {
                "native": native,
                SHADOW_ARM: shadow,
                ACTIVE_ARM: active,
            }
            for arm, outcome in outcomes.items():
                status_counts[arm][outcome["status"]] += 1
                arm_overheads[arm].append(
                    float(outcome["arm_wall_time_seconds"])
                    - float(outcome["solving_time"])
                )
            for arm, record in (
                (SHADOW_ARM, shadow_record),
                (ACTIVE_ARM, active_record),
            ):
                timing = record["selector"]["timing_seconds"]
                model_load_times[arm].append(float(timing["model_load"]))
                policy_compute_times[arm].append(float(timing["policy_compute"]))

            context = pair["initial_context"]
            if context["all_actions_observed"]:
                context_observed_pairs += 1
                context_mismatch_pairs += not context["matching_across_actions"]

            shadow_comparison = shadow_record["comparison"]
            native_complete = native["status"] in COMPLETE_STATUSES
            shadow_complete = shadow["status"] in COMPLETE_STATUSES
            if native_complete and shadow_complete:
                completed_shadow_pairs += 1
                completed_shadow_full_matches += shadow_comparison["safe"]
            if shadow_comparison["safe"]:
                shadow_classes["full_safety_match"] += 1
            elif (
                not native_complete
                and not shadow_complete
                and native["status"] == shadow["status"]
            ):
                shadow_classes["same_incomplete_limit_status"] += 1
            else:
                shadow_classes["other_full_safety_mismatch"] += 1
            shadow_failed_checks.update(
                name
                for name, passed in shadow_comparison["safety_checks"].items()
                if not passed
            )

            active_comparison = active_record["comparison"]
            checks = active_comparison["safety_checks"]
            failed_mechanical = [name for name in MECHANICAL_CHECKS if not checks[name]]
            if failed_mechanical:
                mechanical_mismatch_pairs.append(
                    {
                        "instance_id": instance_id,
                        "seed": seed,
                        "failed_checks": failed_mechanical,
                    }
                )
            interventions = int(active_record["selector"]["interventions"])
            active_intervention_pairs += interventions > 0
            active_fallback_pairs += interventions == 0
            active_complete = active["status"] in COMPLETE_STATUSES
            if native_complete and not active_complete:
                native_complete_active_incomplete.append(
                    {
                        "instance_id": instance_id,
                        "seed": seed,
                        "native_status": native["status"],
                        "active_status": active["status"],
                    }
                )
            elif active_complete and not native_complete:
                active_complete_native_incomplete.append(
                    {
                        "instance_id": instance_id,
                        "seed": seed,
                        "native_status": native["status"],
                        "active_status": active["status"],
                    }
                )
            elif (
                not native_complete
                and not active_complete
                and native["status"] == active["status"]
            ):
                both_incomplete_same_status.append(
                    {
                        "instance_id": instance_id,
                        "seed": seed,
                        "status": native["status"],
                    }
                )
            if native_complete and active_complete and (
                not _close(native["primal_bound"], active["primal_bound"])
                or not _close(native["dual_bound"], active["dual_bound"])
            ):
                official = official_objectives.get(instance_id)
                native_value = float(native["primal_bound"])
                active_value = float(active["primal_bound"])
                closer = None
                if official is not None:
                    native_distance = abs(native_value - official)
                    active_distance = abs(active_value - official)
                    closer = (
                        "active"
                        if active_distance < native_distance
                        else "native"
                        if native_distance < active_distance
                        else "tie"
                    )
                objective_mismatch_pairs.append(
                    {
                        "instance_id": instance_id,
                        "seed": seed,
                        "native_objective": native_value,
                        "active_objective": active_value,
                        "official_collection_objective": official,
                        "closer_to_official": closer,
                        "official_details_url": (
                            f"https://miplib.zib.de/instance_details_{instance_id}.html"
                        ),
                    }
                )

    performance = {}
    for field in ("arm_wall_time_seconds", "solving_time"):
        performance[field] = {}
        for numerator, denominator, label in (
            (ACTIVE_ARM, "native", "active_over_native"),
            (SHADOW_ARM, "native", "shadow_over_native"),
            (ACTIVE_ARM, SHADOW_ARM, "active_over_shadow"),
        ):
            performance[field][label] = _geometric_mean_ratio(
                _arm_ratio_logs(
                    result, numerator, denominator, field, time_limit
                )
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post-hoc diagnostic; frozen Go/No-Go decision unchanged",
        "source_artifacts": {
            "plan": {
                "path": _manifest_path(plan_path),
                "sha256": _sha256_file(plan_path),
            },
            "result": {
                "path": _manifest_path(result_path),
                "sha256": _sha256_file(result_path),
            },
            "frozen_statistics": {
                "path": _manifest_path(frozen_statistics_path),
                "sha256": _sha256_file(frozen_statistics_path),
            },
            "official_collection_html": {
                "path": _manifest_path(collection_html),
                "sha256": _sha256_file(collection_html),
                "url": "https://miplib.zib.de/set_collection.html",
            },
        },
        "frozen_decision": {
            "passed": frozen["passed"],
            "decision": frozen["decision"],
            "gate_checks": frozen["gate_checks"],
            "primary_full_process_wall_time": frozen[
                "primary_full_process_wall_time"
            ],
            "secondary_scip_solving_time": frozen["secondary_scip_solving_time"],
        },
        "coverage": {
            "group_keys": len(result["per_instance"]),
            "paired_blocks": sum(
                len(instance["pairs"]) for instance in result["per_instance"]
            ),
            "arm_records": sum(
                3 * len(instance["pairs"]) for instance in result["per_instance"]
            ),
            "active_intervention_pairs": active_intervention_pairs,
            "active_fallback_pairs": active_fallback_pairs,
            "pre_action_context_observed_pairs": context_observed_pairs,
            "pre_action_context_mismatch_pairs": context_mismatch_pairs,
        },
        "status_counts": {
            arm: dict(sorted(counts.items())) for arm, counts in status_counts.items()
        },
        "shadow_diagnostic": {
            "classifications": dict(sorted(shadow_classes.items())),
            "completed_pairs": completed_shadow_pairs,
            "completed_pairs_with_full_safety_match": completed_shadow_full_matches,
            "failed_check_counts": dict(sorted(shadow_failed_checks.items())),
            "interpretation": (
                "The frozen full-safety gate includes completion and terminal structural "
                "counters. Same-limit incomplete pairs are not evidence that shadow changed "
                "the selected cut set."
            ),
        },
        "active_safety_diagnostic": {
            "mechanical_mismatch_pairs": mechanical_mismatch_pairs,
            "objective_mismatch_pairs": objective_mismatch_pairs,
            "native_complete_active_incomplete_pairs": native_complete_active_incomplete,
            "active_complete_native_incomplete_pairs": active_complete_native_incomplete,
            "both_incomplete_same_status_pairs": both_incomplete_same_status,
            "interpretation": (
                "Objective disagreement remains a frozen safety-gate failure. The official "
                "Collection objective is reported only as a post-hoc attribution diagnostic."
            ),
        },
        "performance_decomposition": performance,
        "implementation_timing_seconds": {
            "median_wall_minus_scip_solving": {
                arm: _median(values) for arm, values in arm_overheads.items()
            },
            "median_model_load": {
                arm: _median(values) for arm, values in model_load_times.items()
            },
            "median_policy_compute": {
                arm: _median(values) for arm, values in policy_compute_times.items()
            },
        },
        "interpretation_boundary": (
            "These diagnostics were computed after the frozen No-Go result. They explain "
            "failure modes but neither alter the gate nor authorize model tuning, test-set "
            "opening, or another learned-policy iteration."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--frozen-statistics", type=Path, default=DEFAULT_FROZEN_STATISTICS
    )
    parser.add_argument("--collection-html", type=Path, default=DEFAULT_COLLECTION_HTML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing diagnostic manifest")
    diagnostics = build_diagnostics(
        args.result.resolve(),
        args.plan.resolve(),
        args.frozen_statistics.resolve(),
        args.collection_html.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "frozen_passed": diagnostics["frozen_decision"]["passed"],
                "objective_mismatches": len(
                    diagnostics["active_safety_diagnostic"][
                        "objective_mismatch_pairs"
                    ]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
