"""Quantify instance support required for a future learned-policy experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .observational import PROJECT_ROOT
from .paper_statistics import exact_binomial_upper_bound


SCHEMA_VERSION = 3
DEFAULT_ITT = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "paper_statistics_publication_fixed_baselines_itt_v2.json"
)
DEFAULT_ASSIGNMENTS = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "evaluation_protocols"
    / "instance_assignments.csv"
)
DEFAULT_TEST_LIST = (
    PROJECT_ROOT / "vendor" / "tracer_snapshot" / "split" / "test.test"
)
DEFAULT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "miplib2017_benchmark_metadata_2026-07-16.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "major_revision_precision_plan_v1.json"
)


def required_zero_failure_instances(
    maximum_failure_probability: float, confidence: float = 0.95
) -> int:
    if not 0.0 < maximum_failure_probability < 1.0:
        raise ValueError("maximum failure probability must be in (0, 1)")
    trials = 1
    while True:
        upper = exact_binomial_upper_bound(0, trials, confidence)
        if upper is not None and upper <= maximum_failure_probability:
            return trials
        trials += 1


def required_paired_instances(
    log_ratio_standard_deviation: float,
    target_ratio: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Normal-approximation size for a one-sided paired log-ratio test."""
    return required_paired_instances_for_log_ratio_test(
        log_ratio_standard_deviation,
        null_ratio=1.0,
        alternative_ratio=target_ratio,
        alpha=alpha,
        power=power,
    )


def required_paired_instances_for_log_ratio_test(
    log_ratio_standard_deviation: float,
    *,
    null_ratio: float,
    alternative_ratio: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Normal-approximation size for a lower-is-better paired log-ratio test."""
    if log_ratio_standard_deviation <= 0.0:
        raise ValueError("log-ratio standard deviation must be positive")
    if not 0.0 < alternative_ratio < null_ratio:
        raise ValueError("alternative ratio must be positive and below null ratio")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be in (0, 1)")
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - alpha) + normal.inv_cdf(power)
    effect = math.log(null_ratio) - math.log(alternative_ratio)
    return math.ceil((critical * log_ratio_standard_deviation / effect) ** 2)


def minimum_detectable_ratio(
    log_ratio_standard_deviation: float,
    instances: int,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    if instances <= 0:
        raise ValueError("instances must be positive")
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - alpha) + normal.inv_cdf(power)
    detectable_log_effect = critical * log_ratio_standard_deviation / math.sqrt(
        instances
    )
    return math.exp(-detectable_log_effect)


def _log_ratio_standard_deviations(analysis: dict[str, Any]) -> dict[str, float]:
    return {
        action: statistics.stdev(
            instance["mean_log_penalized_time_ratio"]
            for instance in result["per_instance"]
        )
        for action, result in analysis["actions"].items()
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _independent_group_key(instance_name: str, official_group: str) -> str:
    return official_group or f"officially-ungrouped:{instance_name}"


def _sealed_test_group_support(
    assignments_path: Path,
    test_list_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    assignments = _read_csv(assignments_path)
    metadata = {
        row["instance_name"]: row for row in _read_csv(metadata_path)
    }
    test_names = [
        line.strip()
        for line in test_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assigned_test_names = {
        row["instance_name"]
        for row in assignments
        if row["original_split"] == "test"
    }
    if len(test_names) != len(assigned_test_names) or set(test_names) != assigned_test_names:
        raise ValueError("test.test and instance_assignments.csv disagree")

    keys_by_split: dict[str, set[str]] = defaultdict(set)
    test_members: dict[str, list[str]] = defaultdict(list)
    for row in assignments:
        instance_name = row["instance_name"]
        if instance_name not in metadata:
            raise ValueError(f"missing MIPLIB metadata for {instance_name}")
        metadata_group = metadata[instance_name]["group"]
        if row["official_group"] != metadata_group:
            raise ValueError(f"Group metadata disagree for {instance_name}")
        group_key = _independent_group_key(instance_name, metadata_group)
        keys_by_split[row["original_split"]].add(group_key)
        if row["original_split"] == "test":
            test_members[group_key].append(instance_name)

    repeated_test_keys = {
        key: sorted(names)
        for key, names in sorted(test_members.items())
        if len(names) > 1
    }
    seen_training_keys = sorted(keys_by_split["test"] & keys_by_split["train"])
    unseen_training_keys = sorted(keys_by_split["test"] - keys_by_split["train"])
    return {
        "source_test_list": str(test_list_path.resolve()),
        "source_group_metadata": str(metadata_path.resolve()),
        "instances": len(test_names),
        "distinct_group_keys": len(keys_by_split["test"]),
        "group_key_definition": (
            "nonempty official MIPLIB Group; officially ungrouped instances use "
            "one instance-specific key"
        ),
        "repeated_test_group_keys": repeated_test_keys,
        "group_keys_seen_in_training": len(seen_training_keys),
        "seen_training_group_keys": seen_training_keys,
        "group_keys_unseen_in_training": len(unseen_training_keys),
        "unseen_training_group_keys": unseen_training_keys,
        "strict_group_ood": not seen_training_keys,
    }


def _split_counts(assignments_path: Path) -> dict[str, int]:
    rows = _read_csv(assignments_path)
    return {
        split: sum(row["original_split"] == split for row in rows)
        for split in ("train", "val", "test")
    }


def build_precision_plan(
    itt_path: Path = DEFAULT_ITT,
    assignments_path: Path = DEFAULT_ASSIGNMENTS,
    test_list_path: Path = DEFAULT_TEST_LIST,
    metadata_path: Path = DEFAULT_METADATA,
) -> dict[str, Any]:
    analysis = json.loads(itt_path.read_text(encoding="utf-8"))
    standard_deviations = _log_ratio_standard_deviations(analysis)
    planning_standard_deviation = max(standard_deviations.values())
    split_counts = _split_counts(assignments_path)
    test_instances = split_counts["test"]
    test_support = _sealed_test_group_support(
        assignments_path, test_list_path, metadata_path
    )
    test_group_keys = test_support["distinct_group_keys"]
    effect_sizes = {}
    for relative_improvement in (0.05, 0.10, 0.15):
        target_ratio = 1.0 - relative_improvement
        effect_sizes[f"{relative_improvement:.0%}"] = {
            "target_ratio": target_ratio,
            "required_instances_one_sided_alpha_0.05_power_0.80": (
                required_paired_instances(
                    planning_standard_deviation, target_ratio
                )
            ),
        }
    safety_sizes = {}
    for maximum_probability in (0.025, 0.05, 0.10):
        required = required_zero_failure_instances(maximum_probability)
        safety_sizes[f"{maximum_probability:.1%}"] = {
            "required_zero_failure_instances": required,
            "achieved_upper_bound": exact_binomial_upper_bound(0, required),
        }
    instance_assumption_upper = exact_binomial_upper_bound(0, test_instances)
    instance_assumption_ratio = minimum_detectable_ratio(
        planning_standard_deviation, test_instances
    )
    group_cluster_upper = exact_binomial_upper_bound(0, test_group_keys)
    group_cluster_ratio = minimum_detectable_ratio(
        planning_standard_deviation, test_group_keys
    )
    superiority_instances = effect_sizes["10%"][
        "required_instances_one_sided_alpha_0.05_power_0.80"
    ]
    noninferiority_instances = required_paired_instances_for_log_ratio_test(
        planning_standard_deviation,
        null_ratio=1.05,
        alternative_ratio=1.00,
    )
    safety_instances = safety_sizes["5.0%"]["required_zero_failure_instances"]
    superiority_and_safety_instances = max(
        superiority_instances, safety_instances
    )
    dual_purpose_instances = max(
        superiority_instances, noninferiority_instances, safety_instances
    )
    seeds = 3
    main_comparison_arms = 2
    diagnostic_arms = 3
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post-review prospective planning; not applied retroactively",
        "variance_source": {
            "source": str(itt_path.resolve()),
            "instance_log_ratio_standard_deviation_by_fixed_policy": (
                standard_deviations
            ),
            "conservative_planning_standard_deviation": (
                planning_standard_deviation
            ),
            "limitation": (
                "fixed-policy variability is a proxy because no multi-instance learned "
                "online experiment exists"
            ),
        },
        "performance_precision": {
            "estimand": (
                "instance-equal geometric mean paired PAR-2 ratio of learned policy "
                "to native; historical SCIP-time variability is used only as a "
                "planning proxy for a future implementation-aware endpoint"
            ),
            "alpha": 0.05,
            "power": 0.80,
            "one_sided": True,
            "normal_approximation_formula": (
                "ceil((sigma * (z_(1-alpha) + z_power) / "
                "(log(null_ratio) - log(alternative_ratio)))^2)"
            ),
            "superiority_test": {
                "null_hypothesis": "ratio >= 1.00",
                "planning_alternative": "ratio = 0.90",
                "required_instances": superiority_instances,
                "interpretation": (
                    "approximately 80% power when the true ratio is 0.90; this is "
                    "not an 80% probability of proving an improvement of at least 10%"
                ),
            },
            "noninferiority_test": {
                "null_hypothesis": "ratio >= 1.05",
                "planning_alternative": "ratio = 1.00",
                "required_instances": noninferiority_instances,
                "interpretation": (
                    "approximately 80% power to exclude the 5% harm margin when the "
                    "true ratio is 1.00"
                ),
            },
            "effect_sizes": effect_sizes,
        },
        "safety_precision": {
            "unit": "independent Group key",
            "confidence": 0.95,
            "acceptance_rule": (
                "observe zero failures and require the one-sided exact 95% "
                "Clopper-Pearson upper bound to be no greater than 0.05"
            ),
            "required_instances_for_five_percent_cap": safety_instances,
            "interpretation": (
                "a confidence-bound requirement, not an 80%-power calculation"
            ),
            "maximum_failure_probability_scenarios": safety_sizes,
        },
        "available_sealed_test": {
            **test_support,
            "instance_independence_sensitivity": {
                "independent_units": test_instances,
                "assumption": "all 35 test instances are mutually independent",
                "zero_failure_upper_bound": instance_assumption_upper,
                "minimum_detectable_ratio": instance_assumption_ratio,
                "minimum_detectable_relative_improvement": (
                    1.0 - instance_assumption_ratio
                ),
            },
            "official_group_cluster_sensitivity": {
                "independent_units": test_group_keys,
                "zero_failure_upper_bound": group_cluster_upper,
                "minimum_detectable_ratio": group_cluster_ratio,
                "minimum_detectable_relative_improvement": (
                    1.0 - group_cluster_ratio
                ),
            },
            "group_ood_limitation": (
                "10 test Group keys also occur in training, so the sealed split "
                "cannot support a strict Group-OOD claim as a whole"
            ),
            "adequate_for_10_percent_improvement_and_5_percent_failure_cap": False,
        },
        "recommended_future_design": {
            "superiority_and_safety_minimum_instances": (
                superiority_and_safety_instances
            ),
            "dual_purpose_minimum_independent_instances": dual_purpose_instances,
            "minimum_independent_instances": dual_purpose_instances,
            "independent_unit_definition": (
                "official MIPLIB Group key; officially ungrouped instances use "
                "one instance-specific key"
            ),
            "additional_independent_group_keys_beyond_sealed_test": max(
                0, dual_purpose_instances - test_group_keys
            ),
            "additional_group_ood_keys_beyond_current_unseen_test": max(
                0,
                dual_purpose_instances
                - test_support["group_keys_unseen_in_training"],
            ),
            "seeds_per_instance": seeds,
            "arms": [
                "native",
                "xgb-imitation-shadow",
                "xgb-imitation-rank",
            ],
            "two_arm_main_comparison_runs": (
                dual_purpose_instances * seeds * main_comparison_arms
            ),
            "three_arm_including_shadow_runs": (
                dual_purpose_instances * seeds * diagnostic_arms
            ),
            "superiority_and_safety_two_arm_runs": (
                superiority_and_safety_instances * seeds * main_comparison_arms
            ),
            "superiority_and_safety_three_arm_including_shadow_runs": (
                superiority_and_safety_instances * seeds * diagnostic_arms
            ),
            "minimum_meaningful_improvement": 0.10,
            "maximum_acceptable_group_key_failure_probability": 0.05,
            "performance_noninferiority_margin": 0.05,
            "primary_multiplicity": (
                "one learned-versus-native policy contrast; superiority and "
                "noninferiority serve distinct predeclared claims, while "
                "shadow-versus-native remains a structural and overhead diagnostic"
            ),
            "group_ood_requirement": (
                "a future Group-OOD queue must use evaluation Group keys that are "
                "disjoint from every development and training Group key"
            ),
            "maximum_basis": {
                "superiority_instances": superiority_instances,
                "noninferiority_instances": noninferiority_instances,
                "zero_failure_safety_instances": safety_instances,
                "maximum": dual_purpose_instances,
            },
        },
        "decision": (
            "Do not unseal the 35-instance test merely to answer the review: it "
            "contains only 34 independent Group keys, including 10 seen in training, "
            "and is insufficient for the proposed superiority, noninferiority, and "
            "safety claims. A superiority-plus-safety study needs "
            f"{superiority_and_safety_instances} independent Group keys; a "
            "dual-purpose design that can also establish 5% noninferiority under a "
            f"true ratio of 1.00 needs {dual_purpose_instances}."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--itt", type=Path, default=DEFAULT_ITT)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--test-list", type=Path, default=DEFAULT_TEST_LIST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_precision_plan(
        args.itt.resolve(),
        args.assignments.resolve(),
        args.test_list.resolve(),
        args.metadata.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "available_test": plan["available_sealed_test"],
                "recommended_instances": plan["recommended_future_design"][
                    "minimum_independent_instances"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
