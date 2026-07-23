"""Compute frozen, instance-equal statistics from causal suite manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 4
COMPLETE_STATUSES = frozenset(("optimal", "infeasible", "unbounded", "inforunbd"))
DESCRIPTIVE_SOLVER_METRICS = (
    "nodes",
    "total_nodes",
    "lp_iterations",
    "lp_count",
    "cuts_applied",
    "primal_dual_integral",
)


def _number(value: Any) -> float:
    return float(value)


def _numbers_close(left: Any, right: Any) -> bool:
    try:
        left_number = _number(left)
        right_number = _number(right)
    except (TypeError, ValueError):
        return left == right
    if math.isnan(left_number) or math.isnan(right_number):
        return math.isnan(left_number) and math.isnan(right_number)
    if math.isinf(left_number) or math.isinf(right_number):
        return left_number == right_number
    return math.isclose(left_number, right_number, rel_tol=1e-9, abs_tol=1e-9)


def _complete(outcome: dict[str, Any]) -> bool:
    return outcome.get("status") in COMPLETE_STATUSES


def _solver_equivalent(
    native: dict[str, Any], treatment: dict[str, Any]
) -> bool:
    return (
        _complete(native)
        and _complete(treatment)
        and native.get("status") == treatment.get("status")
        and native.get("objective_sense") == treatment.get("objective_sense")
        and _numbers_close(native.get("primal_bound"), treatment.get("primal_bound"))
        and _numbers_close(native.get("dual_bound"), treatment.get("dual_bound"))
    )


def _penalized_time(outcome: dict[str, Any], time_limit: float) -> float:
    if _complete(outcome):
        return max(float(outcome["solving_time"]), 1e-12)
    return 2.0 * time_limit


def _log_binomial_cdf(failures: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return -math.inf if failures < trials else 0.0
    log_probability = math.log(probability)
    log_complement = math.log1p(-probability)
    terms = [
        math.lgamma(trials + 1)
        - math.lgamma(count + 1)
        - math.lgamma(trials - count + 1)
        + count * log_probability
        + (trials - count) * log_complement
        for count in range(failures + 1)
    ]
    maximum = max(terms)
    return maximum + math.log(sum(math.exp(term - maximum) for term in terms))


def exact_binomial_upper_bound(
    failures: int, trials: int, confidence: float = 0.95
) -> float | None:
    """Return the one-sided Clopper-Pearson upper confidence bound."""
    if trials < 0 or not 0 <= failures <= trials:
        raise ValueError("failures and trials must satisfy 0 <= failures <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if trials == 0:
        return None
    if failures == trials:
        return 1.0
    alpha_log = math.log1p(-confidence)
    low, high = 0.0, 1.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _log_binomial_cdf(failures, trials, midpoint) > alpha_log:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def cluster_bootstrap_geometric_ratio(
    instance_log_ratios: Iterable[float],
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    values = np.asarray(tuple(instance_log_ratios), dtype=np.float64)
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if values.size == 0:
        return {
            "replicates": replicates,
            "seed": seed,
            "confidence": confidence,
            "lower": None,
            "upper": None,
        }
    generator = np.random.default_rng(seed)
    sampled = generator.choice(values, size=(replicates, values.size), replace=True)
    ratios = np.exp(sampled.mean(axis=1))
    tail = (1.0 - confidence) / 2.0
    return {
        "replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "lower": float(np.quantile(ratios, tail)),
        "upper": float(np.quantile(ratios, 1.0 - tail)),
    }


def _action_bootstrap_seed(seed: int, action: str) -> int:
    digest = hashlib.sha256(action.encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], "big")) % (2**32)


def _paired_solver_metrics(
    native: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, dict[str, float | None]]:
    records: dict[str, dict[str, float | None]] = {}
    for metric in DESCRIPTIVE_SOLVER_METRICS:
        try:
            native_value = float(native[metric])
            treatment_value = float(treatment[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(native_value) or not math.isfinite(treatment_value):
            continue
        delta = treatment_value - native_value
        records[metric] = {
            "native": native_value,
            "treatment": treatment_value,
            "delta_treatment_minus_native": delta,
            "relative_change_over_abs_native": (
                delta / abs(native_value) if native_value != 0.0 else None
            ),
        }
    return records


def _instance_solver_metric_summary(
    pair_solver_metrics: list[dict[str, dict[str, float | None]]],
) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for metric in DESCRIPTIVE_SOLVER_METRICS:
        values = [
            record[metric]
            for record in pair_solver_metrics
            if metric in record
        ]
        if not values:
            continue
        native_mean = float(np.mean([value["native"] for value in values]))
        treatment_mean = float(
            np.mean([value["treatment"] for value in values])
        )
        relative_changes = [
            value["relative_change_over_abs_native"]
            for value in values
            if value["relative_change_over_abs_native"] is not None
        ]
        summary[metric] = {
            "pairs": len(values),
            "native_seed_mean": native_mean,
            "treatment_seed_mean": treatment_mean,
            "delta_treatment_minus_native": treatment_mean - native_mean,
            "mean_pair_relative_change_over_abs_native": (
                float(np.mean(relative_changes)) if relative_changes else None
            ),
        }
    return summary


def _aggregate_solver_metrics(
    instances: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | str | None]]:
    result: dict[str, dict[str, float | int | str | None]] = {}
    for metric in DESCRIPTIVE_SOLVER_METRICS:
        records = [
            instance["descriptive_solver_metrics"][metric]
            for instance in instances
            if metric in instance["descriptive_solver_metrics"]
        ]
        if not records:
            continue
        deltas = np.asarray(
            [record["delta_treatment_minus_native"] for record in records],
            dtype=np.float64,
        )
        native_means = np.asarray(
            [record["native_seed_mean"] for record in records], dtype=np.float64
        )
        treatment_means = np.asarray(
            [record["treatment_seed_mean"] for record in records], dtype=np.float64
        )
        relative_changes = [
            record["mean_pair_relative_change_over_abs_native"]
            for record in records
            if record["mean_pair_relative_change_over_abs_native"] is not None
        ]
        tolerance = 1e-12
        result[metric] = {
            "estimand": "descriptive instance-equal mean over all predeclared blocks",
            "censoring_note": "values at the time limit are retained; no censoring correction or causal claim is attached",
            "instances": len(records),
            "pairs": int(sum(record["pairs"] for record in records)),
            "native_instance_equal_mean": float(np.mean(native_means)),
            "treatment_instance_equal_mean": float(np.mean(treatment_means)),
            "ratio_of_instance_equal_means_treatment_over_native": (
                float(np.mean(treatment_means) / np.mean(native_means))
                if float(np.mean(native_means)) != 0.0
                else None
            ),
            "mean_delta_treatment_minus_native": float(np.mean(deltas)),
            "median_instance_delta_treatment_minus_native": float(
                np.median(deltas)
            ),
            "mean_instance_relative_change_over_abs_native": (
                float(np.mean(relative_changes)) if relative_changes else None
            ),
            "instances_treatment_lower": int(np.sum(deltas < -tolerance)),
            "instances_equal": int(np.sum(np.abs(deltas) <= tolerance)),
            "instances_treatment_higher": int(np.sum(deltas > tolerance)),
        }
    return result


def _instance_action_record(
    instance: dict[str, Any], action: str, time_limit: float
) -> dict[str, Any]:
    pair_records = []
    pair_solver_metrics = []
    itt_log_ratios = []
    native_complete_log_ratios = []
    safety_failures = 0
    interventions = 0
    policy_fallbacks = 0
    outcome_counts = {
        "both_complete": 0,
        "native_complete_treatment_incomplete": 0,
        "native_incomplete_treatment_complete": 0,
        "both_incomplete": 0,
    }
    for pair in instance["pairs"]:
        if action not in pair["actions"]:
            raise ValueError(
                f"Instance {instance['instance_id']} lacks action {action!r}"
            )
        native = pair["native_outcome"]
        treatment_record = pair["actions"][action]
        treatment = treatment_record["outcome"]
        comparison = treatment_record.get("comparison", {})
        selector = treatment_record.get("selector", {})
        native_complete = _complete(native)
        treatment_complete = _complete(treatment)
        equivalent = _solver_equivalent(native, treatment) if native_complete else False
        safety_failure = native_complete and not equivalent
        intervention = int(selector.get("interventions", 0)) > 0
        policy_fallback = bool(
            comparison.get("policy_fallback", not intervention)
        )
        solver_metrics = _paired_solver_metrics(native, treatment)
        pair_solver_metrics.append(solver_metrics)
        native_penalized_time = _penalized_time(native, time_limit)
        treatment_penalized_time = _penalized_time(treatment, time_limit)
        itt_ratio = treatment_penalized_time / native_penalized_time
        itt_log_ratios.append(math.log(itt_ratio))
        native_complete_ratio = None
        if native_complete:
            native_complete_ratio = itt_ratio
            native_complete_log_ratios.append(math.log(native_complete_ratio))
        if native_complete and treatment_complete:
            outcome_class = "both_complete"
        elif native_complete:
            outcome_class = "native_complete_treatment_incomplete"
        elif treatment_complete:
            outcome_class = "native_incomplete_treatment_complete"
        else:
            outcome_class = "both_incomplete"
        outcome_counts[outcome_class] += 1
        safety_failures += int(safety_failure)
        interventions += int(intervention)
        policy_fallbacks += int(policy_fallback)
        pair_records.append(
            {
                "seed": pair["seed"],
                "native_status": native.get("status"),
                "treatment_status": treatment.get("status"),
                "native_complete": native_complete,
                "treatment_complete": treatment_complete,
                "outcome_class": outcome_class,
                "solver_equivalent": equivalent,
                "safety_failure": safety_failure,
                "intervention": intervention,
                "policy_fallback": policy_fallback,
                "native_penalized_time": native_penalized_time,
                "treatment_penalized_time": treatment_penalized_time,
                "itt_penalized_time_ratio_treatment_over_native": itt_ratio,
                "native_complete_penalized_time_ratio_treatment_over_native": (
                    native_complete_ratio
                ),
            }
        )
    if not itt_log_ratios:
        raise ValueError(f"Instance {instance['instance_id']} contains no seed pairs")
    mean_log_itt_ratio = sum(itt_log_ratios) / len(itt_log_ratios)
    itt_ratio = math.exp(mean_log_itt_ratio)
    mean_log_native_complete_ratio = (
        sum(native_complete_log_ratios) / len(native_complete_log_ratios)
        if native_complete_log_ratios
        else None
    )
    native_complete_ratio = (
        math.exp(mean_log_native_complete_ratio)
        if mean_log_native_complete_ratio is not None
        else None
    )
    descriptive_solver_metrics = _instance_solver_metric_summary(
        pair_solver_metrics
    )
    return {
        "instance_id": instance["instance_id"],
        "instance_sha256": instance.get("instance_sha256"),
        "pairs": pair_records,
        "total_pairs": len(pair_records),
        "native_complete_pairs": len(native_complete_log_ratios),
        "native_incomplete_pairs": len(pair_records) - len(native_complete_log_ratios),
        "treatment_complete_pairs": sum(
            int(record["treatment_complete"]) for record in pair_records
        ),
        "treatment_incomplete_pairs": sum(
            int(not record["treatment_complete"]) for record in pair_records
        ),
        **{f"{key}_pairs": value for key, value in outcome_counts.items()},
        "safety_failure_pairs": safety_failures,
        "safety_failure_instance": safety_failures > 0,
        "intervention_pairs": interventions,
        "policy_fallback_pairs": policy_fallbacks,
        "mean_log_penalized_time_ratio": mean_log_itt_ratio,
        "geometric_mean_penalized_time_ratio": itt_ratio,
        "relative_time_saving": 1.0 - itt_ratio,
        "native_complete_secondary": {
            "mean_log_penalized_time_ratio": mean_log_native_complete_ratio,
            "geometric_mean_penalized_time_ratio": native_complete_ratio,
            "relative_time_saving": (
                1.0 - native_complete_ratio
                if native_complete_ratio is not None
                else None
            ),
        },
        "descriptive_solver_metrics": descriptive_solver_metrics,
    }


def analyze_action(
    manifest: dict[str, Any],
    action: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    time_limit = float(manifest["time_limit"])
    instances = [
        _instance_action_record(instance, action, time_limit)
        for instance in manifest["per_instance"]
    ]
    conditional_safety_instances = [
        instance for instance in instances if instance["native_complete_pairs"] > 0
    ]
    itt_log_ratios = [
        instance["mean_log_penalized_time_ratio"] for instance in instances
    ]
    point_ratio = math.exp(sum(itt_log_ratios) / len(itt_log_ratios))
    action_seed = _action_bootstrap_seed(bootstrap_seed, action)
    interval = cluster_bootstrap_geometric_ratio(
        itt_log_ratios, bootstrap_replicates, action_seed, confidence
    )
    native_complete_log_ratios = [
        instance["native_complete_secondary"]["mean_log_penalized_time_ratio"]
        for instance in conditional_safety_instances
    ]
    native_complete_point_ratio = (
        math.exp(
            sum(native_complete_log_ratios) / len(native_complete_log_ratios)
        )
        if native_complete_log_ratios
        else None
    )
    native_complete_interval = cluster_bootstrap_geometric_ratio(
        native_complete_log_ratios,
        bootstrap_replicates,
        _action_bootstrap_seed(bootstrap_seed, f"{action}:native-complete"),
        confidence,
    )
    safety_failures = sum(
        instance["safety_failure_instance"]
        for instance in conditional_safety_instances
    )
    wins = sum(
        instance["geometric_mean_penalized_time_ratio"] < 1.0
        for instance in instances
    )
    ties = sum(
        math.isclose(
            instance["geometric_mean_penalized_time_ratio"],
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for instance in instances
    )
    losses = len(instances) - wins - ties
    native_complete_wins = sum(
        instance["native_complete_secondary"][
            "geometric_mean_penalized_time_ratio"
        ]
        < 1.0
        for instance in conditional_safety_instances
    )
    native_complete_ties = sum(
        math.isclose(
            instance["native_complete_secondary"][
                "geometric_mean_penalized_time_ratio"
            ],
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for instance in conditional_safety_instances
    )
    native_complete_losses = (
        len(conditional_safety_instances)
        - native_complete_wins
        - native_complete_ties
    )
    descriptive_solver_metrics = _aggregate_solver_metrics(instances)
    for instance in instances:
        instance.pop("descriptive_solver_metrics", None)
    gate_checks = {
        "zero_instance_safety_failures": safety_failures == 0,
        "penalized_time_ci95_upper_below_one": (
            interval["upper"] is not None and interval["upper"] < 1.0
        ),
        "more_instance_wins_than_losses": wins > losses,
    }
    return {
        "action": action,
        "instances_predeclared": len(instances),
        "instances_analyzed_intention_to_treat": len(instances),
        "instances_with_native_complete_pair": len(conditional_safety_instances),
        "native_incomplete_instances": (
            len(instances) - len(conditional_safety_instances)
        ),
        "pairs": sum(instance["total_pairs"] for instance in instances),
        "native_complete_pairs": sum(
            instance["native_complete_pairs"] for instance in instances
        ),
        "intervention_pairs": sum(
            instance["intervention_pairs"] for instance in instances
        ),
        "policy_fallback_pairs": sum(
            instance["policy_fallback_pairs"] for instance in instances
        ),
        "outcome_pair_counts": {
            key: sum(instance[f"{key}_pairs"] for instance in instances)
            for key in (
                "both_complete",
                "native_complete_treatment_incomplete",
                "native_incomplete_treatment_complete",
                "both_incomplete",
            )
        },
        "safety_failure_pairs": sum(
            instance["safety_failure_pairs"] for instance in instances
        ),
        "safety_failure_instances": safety_failures,
        "safety_failure_rate_instance": (
            safety_failures / len(conditional_safety_instances)
            if conditional_safety_instances
            else None
        ),
        "safety_failure_rate_upper_one_sided": exact_binomial_upper_bound(
            safety_failures, len(conditional_safety_instances), confidence
        ),
        "conditional_policy_safety": {
            "estimand": (
                "conditional ITT policy safety; fallback is part of the assigned "
                "policy and no selection-change attribution is claimed"
            ),
            "population_rule": (
                "instances with at least one native-complete predeclared seed block"
            ),
            "includes_policy_fallbacks": True,
            "causal_attribution_to_selection_change": False,
            "instances": len(conditional_safety_instances),
            "native_complete_pairs": sum(
                instance["native_complete_pairs"]
                for instance in conditional_safety_instances
            ),
            "policy_fallback_pairs_in_population_instances": sum(
                instance["policy_fallback_pairs"]
                for instance in conditional_safety_instances
            ),
            "failure_rule": (
                "an instance fails if any native-complete pair has an incomplete or "
                "terminally nonequivalent policy outcome"
            ),
            "failure_instances": safety_failures,
            "failure_rate": (
                safety_failures / len(conditional_safety_instances)
                if conditional_safety_instances
                else None
            ),
            "upper_one_sided_exact_clopper_pearson": (
                exact_binomial_upper_bound(
                    safety_failures,
                    len(conditional_safety_instances),
                    confidence,
                )
            ),
        },
        "penalized_time": {
            "estimand": "intention-to-treat over every predeclared seed block",
            "par_factor": 2,
            "pairs": sum(instance["total_pairs"] for instance in instances),
            "instances": len(instances),
            "geometric_mean_ratio_treatment_over_native": point_ratio,
            "relative_saving": 1.0 - point_ratio if point_ratio is not None else None,
            "cluster_bootstrap_interval": interval,
            "instance_wins": wins,
            "instance_ties": ties,
            "instance_losses": losses,
        },
        "native_complete_secondary": {
            "estimand": "secondary analysis conditional on at least one native-complete seed block per instance",
            "par_factor": 2,
            "pairs": sum(
                instance["native_complete_pairs"] for instance in instances
            ),
            "instances": len(conditional_safety_instances),
            "geometric_mean_ratio_treatment_over_native": (
                native_complete_point_ratio
            ),
            "relative_saving": (
                1.0 - native_complete_point_ratio
                if native_complete_point_ratio is not None
                else None
            ),
            "cluster_bootstrap_interval": native_complete_interval,
            "instance_wins": native_complete_wins,
            "instance_ties": native_complete_ties,
            "instance_losses": native_complete_losses,
        },
        "descriptive_solver_metrics": descriptive_solver_metrics,
        "gate": {"passed": bool(gate_checks) and all(gate_checks.values()), "checks": gate_checks},
        "per_instance": instances,
    }


def analyze_manifest(
    manifest: dict[str, Any],
    actions: Iterable[str] | None = None,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260717,
    confidence: float = 0.95,
) -> dict[str, Any]:
    available_actions = tuple(manifest.get("actions", ()))
    selected_actions = tuple(actions) if actions is not None else available_actions
    if not selected_actions:
        raise ValueError("the causal manifest contains no actions")
    unsupported = [action for action in selected_actions if action not in available_actions]
    if unsupported:
        raise ValueError(f"actions are absent from the causal manifest: {unsupported}")
    if bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    source_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    full_wall_pairs = 0
    total_action_pairs = 0
    for instance in manifest["per_instance"]:
        for pair in instance["pairs"]:
            native = pair["native_outcome"]
            for action in selected_actions:
                treatment = pair["actions"][action]["outcome"]
                total_action_pairs += 1
                full_wall_pairs += int(
                    "arm_wall_time_seconds" in native
                    and "arm_wall_time_seconds" in treatment
                )
    full_wall_available = (
        total_action_pairs > 0 and full_wall_pairs == total_action_pairs
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_content_sha256": source_hash,
        "analysis_contract": {
            "independent_unit": "instance",
            "seed_handling": "mean paired log ratios within instance",
            "primary_population": "every predeclared instance-seed block, irrespective of native completion or realized intervention",
            "primary_metric": "intention-to-treat PAR-2 penalized solving-time ratio",
            "primary_time_field": "SCIP-reported solving_time",
            "primary_time_interpretation": (
                "excludes Python process setup and any integration work outside "
                "SCIP's reported solving clock"
            ),
            "full_arm_wall_time_sensitivity_available": full_wall_available,
            "full_arm_wall_time_pairs_available": full_wall_pairs,
            "full_arm_wall_time_pairs_required": total_action_pairs,
            "full_arm_wall_time_limitation": (
                None
                if full_wall_available
                else "historical fixed-policy records did not retain paired full-arm wall time"
            ),
            "aggregation": "instance-equal geometric mean",
            "bootstrap": "resample instances with all nested seed outcomes",
            "confidence": confidence,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "secondary_metric": "native-complete conditional PAR-2 ratio for diagnosis only",
            "descriptive_solver_metrics": list(DESCRIPTIVE_SOLVER_METRICS),
            "descriptive_metric_caveat": "work counters and primal-dual integral are reported without censoring correction and are not promoted to primary causal estimands",
            "safety_estimand": (
                "conditional ITT policy safety including fallback; not attributable "
                "to realized selection changes"
            ),
            "safety_unit": "instance with at least one native-complete pair",
            "safety_interval": "one-sided exact Clopper-Pearson",
        },
        "actions": {
            action: analyze_action(
                manifest,
                action,
                bootstrap_replicates,
                bootstrap_seed,
                confidence,
            )
            for action in selected_actions
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--actions", nargs="+")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    analysis = analyze_manifest(
        manifest,
        args.actions,
        args.bootstrap_replicates,
        args.bootstrap_seed,
        args.confidence,
    )
    analysis["source_manifest"] = str(args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        action: result["gate"]["passed"]
        for action, result in analysis["actions"].items()
    }
    print(json.dumps({"gate_passed_by_action": summary, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
