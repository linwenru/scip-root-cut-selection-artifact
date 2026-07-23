"""Freeze, run, and analyze the post-review learned-path shadow experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .causal_harness import COMPLETE_STATUSES, run_action_oracle_suite
from .observational import PROJECT_ROOT
from .paper_statistics import cluster_bootstrap_geometric_ratio


SCHEMA_VERSION = 2
PLAN_KEY = "major-revision-shadow-overhead-v1-20260720"
SHADOW_ACTION = "xgb-imitation-shadow"
DEFAULT_SOURCE_PLAN = (
    PROJECT_ROOT / "data" / "manifests" / "publication_strengthening_plan_v1.json"
)
DEFAULT_MODEL_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "ranking_imitation_online_xgb_revision_v1.json"
)
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "major_revision_shadow_overhead_plan_v1.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "major_revision_shadow_overhead_v1"
DEFAULT_RESULT = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "causal_major_revision_shadow_overhead_v1.json"
)
DEFAULT_STATISTICS = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "major_revision_shadow_overhead_statistics_v1.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _balanced_schedule(
    instances: list[dict[str, Any]], seeds: list[int]
) -> list[dict[str, Any]]:
    blocks = [
        {
            "instance_id": instance["instance_id"],
            "seed": seed,
            "digest": hashlib.sha256(
                f"{PLAN_KEY}|{instance['instance_id']}|{seed}".encode("utf-8")
            ).hexdigest(),
        }
        for instance in instances
        for seed in seeds
    ]
    ordered = sorted(blocks, key=lambda block: block["digest"])
    schedule = []
    for index, block in enumerate(ordered):
        arm_order = (
            ["native", SHADOW_ACTION]
            if index % 2 == 0
            else [SHADOW_ACTION, "native"]
        )
        schedule.append(
            {
                "instance_id": block["instance_id"],
                "seed": block["seed"],
                "arm_order": arm_order,
            }
        )
    return schedule


def build_plan(
    source_plan_path: Path = DEFAULT_SOURCE_PLAN,
    model_manifest_path: Path = DEFAULT_MODEL_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    result_path: Path = DEFAULT_RESULT,
    statistics_path: Path = DEFAULT_STATISTICS,
) -> dict[str, Any]:
    source_plan = _load(source_plan_path)
    model_manifest = _load(model_manifest_path)
    instances = source_plan["parity"]["instances"]
    seeds = list(source_plan["parity"]["seeds"])
    schedule = _balanced_schedule(instances, seeds)
    first_arm_counts = Counter(record["arm_order"][0] for record in schedule)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen before shadow-overhead outcomes",
        "purpose": (
            "measure structural neutrality and total implementation overhead of the "
            "same native-hybrid, context, feature, and XGBoost inference path"
        ),
        "data_boundary": {
            "selection_split": "train",
            "validation_split": "untouched",
            "test_split": "sealed",
            "independent_unit": "MPS instance",
        },
        "source_artifacts": {
            "publication_strengthening_plan": {
                "path": str(source_plan_path.resolve()),
                "sha256": _sha256_file(source_plan_path),
            },
            "learned_model_manifest": {
                "path": str(model_manifest_path.resolve()),
                "sha256": _sha256_file(model_manifest_path),
                "model_sha256": model_manifest["model"]["sha256"],
                "offline_stage_gate_passed": model_manifest["stage_gate"][
                    "passed"
                ],
            },
        },
        "experiment": {
            "instances": instances,
            "seeds": seeds,
            "time_limit_seconds": float(source_plan["parity"]["time_limit_seconds"]),
            "node_limit": source_plan["parity"]["node_limit"],
            "intervention_scope": "first-run-only",
            "arms": ["native", SHADOW_ACTION],
            "execution_order_key": PLAN_KEY,
            "execution_schedule": schedule,
            "output_dir": str(output_dir.resolve()),
            "result_manifest": str(result_path.resolve()),
            "statistics_manifest": str(statistics_path.resolve()),
            "max_workers": 1,
        },
        "analysis_contract": {
            "primary_population": "all 36 predeclared instance-seed blocks",
            "structural_gate": (
                "all same-status fields, bounds, nodes, LP iterations, LP counts, "
                "and cuts applied match; no shadow arm changes the selected set"
            ),
            "overhead_estimands": [
                "full-block ITT PAR-2 solving-time ratio shadow/native",
                "full-block ITT PAR-2 Python arm-wall-time ratio shadow/native",
                "model-load, callback, context, hybrid, and policy timing components",
            ],
            "callback_coverage": (
                "reported descriptively and not used to remove any block"
            ),
            "inference": (
                "implementation-path overhead and neutrality only; no learned-policy "
                "efficacy or generalization claim"
            ),
        },
        "checks": {
            "twelve_training_instances": len(instances) == 12,
            "three_seeds": seeds == [0, 1, 2],
            "thirty_six_blocks": len(schedule) == 36,
            "balanced_first_arm": first_arm_counts
            == {"native": 18, SHADOW_ACTION: 18},
            "test_remains_sealed": True,
            "model_was_not_authorized_for_efficacy": not model_manifest[
                "stage_gate"
            ]["passed"],
        },
    }
    if not all(plan["checks"].values()):
        failed = [name for name, passed in plan["checks"].items() if not passed]
        raise ValueError(f"Shadow overhead plan checks failed: {failed}")
    return plan


def _schedule_map(plan: dict[str, Any]) -> dict[tuple[str, int], tuple[str, ...]]:
    return {
        (record["instance_id"], int(record["seed"])): tuple(record["arm_order"])
        for record in plan["experiment"]["execution_schedule"]
    }


def run_plan(plan_path: Path, reuse_existing: bool = False) -> dict[str, Any]:
    plan = _load(plan_path)
    if not all(plan["checks"].values()):
        raise ValueError("Frozen shadow-overhead plan has failed checks")
    source = plan["source_artifacts"]
    model_path = Path(source["learned_model_manifest"]["path"])
    if _sha256_file(model_path) != source["learned_model_manifest"]["sha256"]:
        raise ValueError("Learned model manifest changed after the plan was frozen")
    experiment = plan["experiment"]
    instances = [Path(record["instance"]) for record in experiment["instances"]]
    return run_action_oracle_suite(
        instances,
        [SHADOW_ACTION],
        experiment["seeds"],
        experiment["time_limit_seconds"],
        experiment["node_limit"],
        Path(experiment["output_dir"]),
        Path(experiment["result_manifest"]),
        reuse_existing,
        experiment["intervention_scope"],
        model_path,
        experiment["max_workers"],
        None,
        _schedule_map(plan),
    )


def _penalized(outcome: dict[str, Any], field: str, time_limit: float) -> float:
    if outcome["status"] in COMPLETE_STATUSES:
        return max(float(outcome[field]), 1e-12)
    return 2.0 * time_limit


def _timing_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"records": 0, "mean": None, "median": None, "p95": None, "sum": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "records": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "sum": sum(values),
    }


def analyze_result(
    result_path: Path,
    plan_path: Path,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260720,
) -> dict[str, Any]:
    result = _load(result_path)
    plan = _load(plan_path)
    time_limit = float(result["time_limit"])
    metric_instance_logs: dict[str, list[float]] = {
        "solving_time": [],
        "arm_wall_time_seconds": [],
    }
    exposure_metric_instance_logs: dict[str, list[float]] = {
        "solving_time": [],
        "arm_wall_time_seconds": [],
    }
    pair_counts = Counter()
    structural_matches = 0
    full_path_structural_matches = 0
    shadow_evaluations = 0
    proposed_changes = 0
    timing_values: dict[str, list[float]] = defaultdict(list)
    per_instance = []
    for instance in result["per_instance"]:
        local_logs = {metric: [] for metric in metric_instance_logs}
        local_exposure_logs = {
            metric: [] for metric in exposure_metric_instance_logs
        }
        local_records = []
        for pair in instance["pairs"]:
            native = pair["native_outcome"]
            action = pair["actions"][SHADOW_ACTION]
            shadow = action["outcome"]
            comparison = action["comparison"]
            selector = action["selector"]
            native_complete = native["status"] in COMPLETE_STATUSES
            shadow_complete = shadow["status"] in COMPLETE_STATUSES
            if native_complete and shadow_complete:
                pair_counts["both_complete"] += 1
            elif native_complete:
                pair_counts["native_complete_shadow_incomplete"] += 1
            elif shadow_complete:
                pair_counts["native_incomplete_shadow_complete"] += 1
            else:
                pair_counts["both_incomplete"] += 1
            ignored_completeness = {"native_complete", "treatment_complete"}
            structural_match = all(
                passed
                for name, passed in comparison["safety_checks"].items()
                if name not in ignored_completeness
            )
            structural_matches += int(structural_match)
            evaluations = int(selector.get("shadow_evaluations", 0))
            shadow_evaluations += int(evaluations > 0)
            full_path_structural_matches += int(
                evaluations > 0 and structural_match
            )
            proposed = any(
                record.get("proposed_changed_selected_cuts", 0) > 0
                for record in selector.get("shadow_records", [])
            )
            proposed_changes += int(proposed)
            for name, seconds in selector.get("timing_seconds", {}).items():
                timing_values[name].append(float(seconds))
            metric_ratios = {}
            for metric in local_logs:
                native_value = _penalized(native, metric, time_limit)
                shadow_value = _penalized(shadow, metric, time_limit)
                ratio = shadow_value / native_value
                local_logs[metric].append(math.log(ratio))
                if evaluations > 0:
                    local_exposure_logs[metric].append(math.log(ratio))
                metric_ratios[metric] = ratio
            local_records.append(
                {
                    "seed": pair["seed"],
                    "native_status": native["status"],
                    "shadow_status": shadow["status"],
                    "structural_match": structural_match,
                    "shadow_evaluation": evaluations > 0,
                    "proposed_selected_set_change": proposed,
                    "ratios": metric_ratios,
                }
            )
        instance_metrics = {}
        exposure_instance_metrics = {}
        for metric, logs in local_logs.items():
            mean_log = statistics.mean(logs)
            metric_instance_logs[metric].append(mean_log)
            instance_metrics[metric] = math.exp(mean_log)
            exposure_logs = local_exposure_logs[metric]
            if exposure_logs:
                exposure_mean_log = statistics.mean(exposure_logs)
                exposure_metric_instance_logs[metric].append(exposure_mean_log)
                exposure_instance_metrics[metric] = math.exp(exposure_mean_log)
        per_instance.append(
            {
                "instance_id": instance["instance_id"],
                "pairs": local_records,
                "geometric_mean_ratios": instance_metrics,
                "full_path_geometric_mean_ratios": exposure_instance_metrics,
            }
        )

    total_pairs = sum(len(instance["pairs"]) for instance in per_instance)
    metric_results = {}
    exposure_metric_results = {}
    for index, (metric, logs) in enumerate(metric_instance_logs.items()):
        metric_results[metric] = {
            "instances": len(logs),
            "pairs": total_pairs,
            "geometric_mean_ratio_shadow_over_native": math.exp(
                statistics.mean(logs)
            ),
            "cluster_bootstrap_interval": cluster_bootstrap_geometric_ratio(
                logs,
                bootstrap_replicates,
                bootstrap_seed + index,
            ),
        }
        exposure_logs = exposure_metric_instance_logs[metric]
        exposure_metric_results[metric] = {
            "instances": len(exposure_logs),
            "pairs": shadow_evaluations,
            "geometric_mean_ratio_shadow_over_native": math.exp(
                statistics.mean(exposure_logs)
            ),
            "cluster_bootstrap_interval": cluster_bootstrap_geometric_ratio(
                exposure_logs,
                bootstrap_replicates,
                bootstrap_seed + 100 + index,
            ),
        }
    analysis_contract = {
        **plan["analysis_contract"],
        "itt_population": "all 36 fallback-inclusive predeclared pairs",
        "exposure_population": (
            "35 pairs in which the complete feature-and-inference path executed"
        ),
        "bootstrap_type": "paired percentile instance-cluster bootstrap",
        "bootstrap_unit": "12 MPS instances",
        "seed_handling": (
            "average paired log ratios over nested seeds within each instance"
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "itt_bootstrap_seeds": {
            "solving_time": bootstrap_seed,
            "arm_wall_time_seconds": bootstrap_seed + 1,
        },
        "exposure_bootstrap_seeds": {
            "solving_time": bootstrap_seed + 100,
            "arm_wall_time_seconds": bootstrap_seed + 101,
        },
    }
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_plan": str(plan_path.resolve()),
        "source_plan_sha256": _sha256_file(plan_path),
        "source_result": str(result_path.resolve()),
        "source_result_sha256": _sha256_file(result_path),
        "analysis_contract": analysis_contract,
        "instances": len(per_instance),
        "pairs": total_pairs,
        "outcome_pair_counts": dict(sorted(pair_counts.items())),
        "structural_matching_pairs": structural_matches,
        "full_path_structural_matching_pairs": full_path_structural_matches,
        "structural_gate_passed": structural_matches
        == sum(len(instance["pairs"]) for instance in per_instance),
        "shadow_evaluation_pairs": shadow_evaluations,
        "proposed_selected_set_change_pairs": proposed_changes,
        "timing_seconds": {
            name: _timing_summary(values)
            for name, values in sorted(timing_values.items())
        },
        "itt_par2": metric_results,
        "full_path_exposure_sensitivity": exposure_metric_results,
        "per_instance": per_instance,
        "inference": (
            "No structural difference was observed in the 36 fallback-inclusive "
            "policy pairs; only 35 pairs executed the complete learned path. The "
            "time analyses quantify implementation-path overhead, not learned-ranking "
            "efficacy or generalization."
        ),
    }
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    run.add_argument("--reuse-existing", action="store_true")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    analyze.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    analyze.add_argument("--output", type=Path, default=DEFAULT_STATISTICS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        payload = build_plan()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"checks": payload["checks"], "output": str(args.output)}))
        return 0
    if args.command == "run":
        result = run_plan(args.plan.resolve(), args.reuse_existing)
        print(
            json.dumps(
                {
                    "result": result["experiment"],
                    "instances": result["instances"],
                    "all_actions_safe": result["all_actions_safe"],
                }
            )
        )
        return 0
    analysis = analyze_result(args.result.resolve(), args.plan.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "structural_gate_passed": analysis["structural_gate_passed"],
                "pairs": analysis["pairs"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
