"""Freeze and execute the publication-strengthening causal experiment queues."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .causal_harness import run_action_oracle_suite, run_structural_parity_suite
from .observational import PROJECT_ROOT, _instance_stem


SCHEMA_VERSION = 1
DEFAULT_ASSIGNMENTS = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "evaluation_protocols"
    / "instance_assignments.csv"
)
DEFAULT_ROOT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "root_observational_v1.json"
DEFAULT_SOURCE_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "trace_source"
    / "benchmark_output_train"
    / "summary.csv"
)
DEFAULT_INSTANCES_DIR = PROJECT_ROOT / "data" / "raw" / "miplib2017"
DEFAULT_MODEL_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "ranking_imitation_online_xgb_v1.json"
)
DEFAULT_PLAN = PROJECT_ROOT / "data" / "manifests" / "publication_strengthening_plan_v1.json"
DEFAULT_ACTIVE_PLAN = (
    PROJECT_ROOT / "data" / "manifests" / "publication_active_plan_v1.json"
)
ACTIVE_EXECUTION_ORDER_KEY = "publication-active-v1-20260718"

PARITY_INSTANCE_COUNT = 12
ACTIVE_INSTANCE_COUNT = 40
SEEDS = (0, 1, 2)
FIXED_ACTIONS = ("random-rank", "efficacy-rank", "adaptive-score")


@dataclass(frozen=True)
class CandidateInstance:
    instance_id: str
    instance_name: str
    path: Path
    official_group: str
    evaluation_stratum: str
    source_trace_elapsed_seconds: float
    policy_eligible_decisions: int
    policy_eligible_candidates: int
    policy_eligible_applied_labels: int

    @property
    def group_key(self) -> str:
        return self.official_group or f"officially-ungrouped:{self.instance_id}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def discover_candidates(
    assignments_path: Path,
    root_manifest_path: Path,
    source_summary_path: Path,
    instances_dir: Path,
) -> list[CandidateInstance]:
    assignments = {
        _instance_stem(row["instance_name"]): row
        for row in _read_csv(assignments_path)
        if row["original_split"] == "train"
    }
    summary = {
        row["name"]: row
        for row in _read_csv(source_summary_path)
        if row["status"] == "ok"
    }
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    candidates = []
    for instance_id, assignment in assignments.items():
        source = summary.get(instance_id)
        quality = root_manifest["instances"].get(assignment["instance_name"], {})
        counts = quality.get("counts", {})
        instance_path = instances_dir / assignment["instance_name"]
        if (
            source is None
            or counts.get("policy_eligible_decisions", 0) <= 0
            or not instance_path.is_file()
        ):
            continue
        candidates.append(
            CandidateInstance(
                instance_id=instance_id,
                instance_name=assignment["instance_name"],
                path=instance_path.resolve(),
                official_group=assignment["official_group"],
                evaluation_stratum=assignment["evaluation_stratum"],
                source_trace_elapsed_seconds=float(source["elapsed"]),
                policy_eligible_decisions=int(counts["policy_eligible_decisions"]),
                policy_eligible_candidates=int(counts["policy_eligible_candidates"]),
                policy_eligible_applied_labels=int(
                    counts["policy_eligible_applied_labels"]
                ),
            )
        )
    return sorted(
        candidates,
        key=lambda item: (item.source_trace_elapsed_seconds, item.instance_id),
    )


def select_distinct_groups(
    candidates: Iterable[CandidateInstance], count: int, used_groups: set[str] | None = None
) -> list[CandidateInstance]:
    used = set() if used_groups is None else used_groups
    selected = []
    for candidate in candidates:
        if candidate.group_key in used:
            continue
        selected.append(candidate)
        used.add(candidate.group_key)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} distinct groups available; {count} required")


def select_completion_enriched(
    candidates: list[CandidateInstance], count: int
) -> tuple[list[CandidateInstance], list[dict[str, Any]]]:
    """Select 30 completion-enriched cases plus 10 full-range hard cases."""
    if count != 40:
        raise ValueError("The v1 active cohort size is frozen at 40")
    used: set[str] = set()
    completion_enriched = select_distinct_groups(candidates, 30, used)
    remaining = [candidate for candidate in candidates if candidate.group_key not in used]
    hard_coverage = []
    for bin_index in range(10):
        start = bin_index * len(remaining) // 10
        stop = (bin_index + 1) * len(remaining) // 10
        chosen = select_distinct_groups(remaining[start:stop], 1, used)[0]
        hard_coverage.append(chosen)
    metadata = [
        {
            "name": "completion_enriched",
            "sampling": "first 30 distinct group keys by source trace runtime and instance ID",
            "selected_instances": len(completion_enriched),
            "minimum_source_trace_seconds": min(
                item.source_trace_elapsed_seconds for item in completion_enriched
            ),
            "maximum_source_trace_seconds": max(
                item.source_trace_elapsed_seconds for item in completion_enriched
            ),
        },
        {
            "name": "hardness_coverage",
            "sampling": (
                "partition the remaining runtime-ordered pool into ten equal-rank bins "
                "and take the first unused group key from each bin"
            ),
            "selected_instances": len(hard_coverage),
            "minimum_source_trace_seconds": min(
                item.source_trace_elapsed_seconds for item in hard_coverage
            ),
            "maximum_source_trace_seconds": max(
                item.source_trace_elapsed_seconds for item in hard_coverage
            ),
        },
    ]
    return completion_enriched + hard_coverage, metadata


def _instance_record(candidate: CandidateInstance) -> dict[str, Any]:
    return {
        "instance_id": candidate.instance_id,
        "instance_name": candidate.instance_name,
        "instance": str(candidate.path),
        "instance_sha256": _sha256_file(candidate.path),
        "official_group": candidate.official_group or None,
        "group_key": candidate.group_key,
        "evaluation_stratum": candidate.evaluation_stratum,
        "source_trace_elapsed_seconds": candidate.source_trace_elapsed_seconds,
        "policy_eligible_decisions": candidate.policy_eligible_decisions,
        "policy_eligible_candidates": candidate.policy_eligible_candidates,
        "policy_eligible_applied_labels": candidate.policy_eligible_applied_labels,
    }


def build_plan(
    assignments_path: Path,
    root_manifest_path: Path,
    source_summary_path: Path,
    instances_dir: Path,
    online_model_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    candidates = discover_candidates(
        assignments_path, root_manifest_path, source_summary_path, instances_dir
    )
    parity = select_distinct_groups(candidates, PARITY_INSTANCE_COUNT)
    active, runtime_strata = select_completion_enriched(candidates, ACTIVE_INSTANCE_COUNT)
    model_manifest = json.loads(
        online_model_manifest_path.read_text(encoding="utf-8")
    )
    model_gate_passed = bool(model_manifest["stage_gate"]["passed"])
    if model_gate_passed:
        raise ValueError(
            "This negative-result v1 plan assumes the frozen online imitation gate failed; "
            "a passing model requires a new protocol version before active evaluation"
        )

    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "pre_registered_before_publication_strengthening_runs",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "increase independent-instance support for instrumentation parity and fixed "
            "root-ranking negative-result estimation without opening validation or test"
        ),
        "data_boundary": {
            "selection_split": "train",
            "validation_split": "sealed",
            "test_split": "sealed",
            "independent_unit": "MPS instance",
            "paired_repeated_measurements": "SCIP seeds nested within instance",
        },
        "source_artifacts": {
            "assignments": {
                "path": str(assignments_path.resolve()),
                "sha256": _sha256_file(assignments_path),
            },
            "root_observational_manifest": {
                "path": str(root_manifest_path.resolve()),
                "sha256": _sha256_file(root_manifest_path),
            },
            "source_trace_summary": {
                "path": str(source_summary_path.resolve()),
                "sha256": _sha256_file(source_summary_path),
            },
            "online_model_manifest": {
                "path": str(online_model_manifest_path.resolve()),
                "sha256": _sha256_file(online_model_manifest_path),
                "offline_stage_gate_passed": model_gate_passed,
            },
        },
        "selection_contract": {
            "eligible_pool": (
                "original train split, successful source trace, at least one leakage-safe "
                "policy-eligible root decision, and locally verified MPS file"
            ),
            "outcome_blinding": (
                "cohorts use only official Group, source trace runtime, eligibility counts, "
                "and instance ID; no new parity or fixed-baseline treatment outcome"
            ),
            "group_rule": (
                "at most one instance per official MIPLIB Group; each officially ungrouped "
                "instance is its own group key"
            ),
            "parity_sampling": (
                "sort eligible instances by source trace elapsed seconds then instance ID; "
                "take the first 12 distinct group keys"
            ),
            "active_sampling": (
                "take the first 30 distinct group keys by source runtime, then partition "
                "the remaining runtime-ordered pool into ten bins and take one unused "
                "group key per bin"
            ),
        },
        "parity": {
            "instances": [_instance_record(item) for item in parity],
            "candidate_arms": ["noop", "direct-hybrid"],
            "seeds": list(SEEDS),
            "time_limit_seconds": 60.0,
            "node_limit": None,
            "output_dir": str((PROJECT_ROOT / "experiments" / "publication_parity_v1").resolve()),
            "manifest": str((PROJECT_ROOT / "data" / "manifests" / "causal_publication_parity_v1.json").resolve()),
            "gate": "every structural field matches and every candidate callback is exercised",
        },
        "active_fixed_baselines": {
            "instances": [_instance_record(item) for item in active],
            "sampling_strata": runtime_strata,
            "actions": list(FIXED_ACTIONS),
            "seeds": list(SEEDS),
            "time_limit_seconds": 300.0,
            "node_limit": None,
            "intervention_scope": "first-run-only",
            "output_dir": str((PROJECT_ROOT / "experiments" / "publication_fixed_baselines_v1").resolve()),
            "manifest": str((PROJECT_ROOT / "data" / "manifests" / "causal_publication_fixed_baselines_v1.json").resolve()),
            "estimand": (
                "instance-equal paired PAR-2 solving-time ratio; seeds aggregate within "
                "instance before instances receive equal weight"
            ),
            "role": (
                "precision-strengthening negative-result estimate on train instances; it "
                "does not authorize validation/test advancement"
            ),
        },
        "learned_baseline": {
            "arm": "xgb-imitation-rank",
            "active_authorized": False,
            "reason": (
                "the frozen online-compatible model failed the pre-existing official-Group-"
                "OOD confidence gate"
            ),
            "allowed_use": "single-instance integration smoke only; no performance claim",
        },
        "checks": {
            "eligible_pool_at_least_40": len(candidates) >= ACTIVE_INSTANCE_COUNT,
            "parity_instances": len(parity) == PARITY_INSTANCE_COUNT,
            "active_instances": len(active) == ACTIVE_INSTANCE_COUNT,
            "parity_group_keys_unique": len({item.group_key for item in parity})
            == len(parity),
            "active_group_keys_unique": len({item.group_key for item in active})
            == len(active),
            "three_paired_seeds": len(SEEDS) == 3,
            "validation_and_test_sealed": True,
            "learned_model_not_active_after_failed_gate": not model_gate_passed,
        },
    }
    if not all(plan["checks"].values()):
        failed = [name for name, passed in plan["checks"].items() if not passed]
        raise ValueError(f"Publication plan checks failed: {failed}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return plan


def build_balanced_arm_schedule(
    instances: list[dict[str, Any]],
    seeds: list[int],
    actions: list[str],
    key: str,
) -> list[dict[str, Any]]:
    """Assign four balanced carryover sequences to hash-randomized blocks."""
    arms = ("native", *actions)
    if len(arms) != 4:
        raise ValueError("Publication V1 freezes exactly four active arms")
    base = (0, 1, 3, 2)
    sequences = [
        tuple(arms[(index + rotation) % 4] for index in base)
        for rotation in range(4)
    ]
    blocks = [
        (instance["instance_id"], int(seed))
        for instance in instances
        for seed in seeds
    ]
    blocks.sort(
        key=lambda block: hashlib.sha256(
            f"{key}\0{block[0]}\0{block[1]}".encode("utf-8")
        ).digest()
    )
    return [
        {
            "instance_id": instance_id,
            "seed": seed,
            "design_row": index % 4,
            "arm_order": list(sequences[index % 4]),
        }
        for index, (instance_id, seed) in enumerate(blocks)
    ]


def _schedule_position_counts(
    schedule: list[dict[str, Any]],
) -> dict[str, dict[int, int]]:
    counts: dict[str, dict[int, int]] = {}
    for block in schedule:
        for position, arm in enumerate(block["arm_order"]):
            counts.setdefault(arm, {}).setdefault(position, 0)
            counts[arm][position] += 1
    return counts


def build_active_plan(parent_plan_path: Path, output_path: Path) -> dict[str, Any]:
    """Freeze resource isolation and balanced arm order before active outcomes."""
    parent = json.loads(parent_plan_path.read_text(encoding="utf-8"))
    contract = json.loads(json.dumps(parent["active_fixed_baselines"]))
    result_path = Path(contract["manifest"])
    if result_path.exists():
        raise ValueError(
            f"Active result already exists; execution plan cannot be frozen now: {result_path}"
        )
    schedule = build_balanced_arm_schedule(
        contract["instances"],
        contract["seeds"],
        contract["actions"],
        ACTIVE_EXECUTION_ORDER_KEY,
    )
    contract.update(
        {
            "max_workers": 1,
            "resource_isolation": (
                "one SCIP process at a time; each SCIP arm uses one solver and LP thread"
            ),
            "execution_order_key": ACTIVE_EXECUTION_ORDER_KEY,
            "execution_order": (
                "assign SHA-256-randomized instance-seed blocks to four balanced "
                "Williams/Latin sequence rows"
            ),
            "execution_schedule": schedule,
        }
    )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "pre_registered_before_active_fixed_baseline_outcomes",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_plan": str(parent_plan_path.resolve()),
        "parent_plan_sha256": _sha256_file(parent_plan_path),
        "reason_for_separate_execution_plan": (
            "wall-clock is the primary outcome, so concurrent SCIP arms and a fixed "
            "native-first order are prohibited"
        ),
        "data_boundary": parent["data_boundary"],
        "active_fixed_baselines": contract,
        "checks": {
            "parent_plan_pre_registered": parent["status"]
            == "pre_registered_before_publication_strengthening_runs",
            "active_result_absent": not result_path.exists(),
            "single_process_execution": contract["max_workers"] == 1,
            "three_paired_seeds": contract["seeds"] == list(SEEDS),
            "forty_instances": len(contract["instances"]) == ACTIVE_INSTANCE_COUNT,
            "arm_order_key_frozen": bool(contract["execution_order_key"]),
            "arm_positions_exactly_balanced": all(
                count == 30
                for arm_counts in _schedule_position_counts(schedule).values()
                for count in arm_counts.values()
            ),
            "validation_and_test_sealed": True,
        },
    }
    if not all(plan["checks"].values()):
        failed = [name for name, passed in plan["checks"].items() if not passed]
        raise ValueError(f"Active execution plan checks failed: {failed}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return plan


def _verify_plan_instances(records: list[dict[str, Any]]) -> list[Path]:
    paths = []
    for record in records:
        path = Path(record["instance"])
        if not path.is_file() or _sha256_file(path) != record["instance_sha256"]:
            raise ValueError(f"Planned instance is missing or changed: {path}")
        paths.append(path)
    return paths


def run_parity_from_plan(
    plan_path: Path, reuse_existing: bool, max_workers: int
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    contract = plan["parity"]
    result = run_structural_parity_suite(
        _verify_plan_instances(contract["instances"]),
        contract["candidate_arms"],
        contract["seeds"],
        contract["time_limit_seconds"],
        contract["node_limit"],
        Path(contract["output_dir"]),
        Path(contract["manifest"]),
        reuse_existing,
        max_workers,
    )
    result["publication_plan"] = str(plan_path.resolve())
    result["publication_plan_sha256"] = _sha256_file(plan_path)
    Path(contract["manifest"]).write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def run_active_from_plan(
    plan_path: Path, reuse_existing: bool, max_workers: int
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    contract = plan["active_fixed_baselines"]
    if max_workers != contract["max_workers"]:
        raise ValueError(
            f"Active plan requires jobs={contract['max_workers']}; received {max_workers}"
        )
    schedule = {
        (block["instance_id"], int(block["seed"])): tuple(block["arm_order"])
        for block in contract["execution_schedule"]
    }
    result = run_action_oracle_suite(
        _verify_plan_instances(contract["instances"]),
        contract["actions"],
        contract["seeds"],
        contract["time_limit_seconds"],
        contract["node_limit"],
        Path(contract["output_dir"]),
        Path(contract["manifest"]),
        reuse_existing,
        contract["intervention_scope"],
        None,
        max_workers,
        contract["execution_order_key"],
        schedule,
    )
    result["publication_plan"] = str(plan_path.resolve())
    result["publication_plan_sha256"] = _sha256_file(plan_path)
    Path(contract["manifest"]).write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze the train-only experiment plan")
    freeze.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    freeze.add_argument("--root-manifest", type=Path, default=DEFAULT_ROOT_MANIFEST)
    freeze.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
    freeze.add_argument("--instances-dir", type=Path, default=DEFAULT_INSTANCES_DIR)
    freeze.add_argument("--online-model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    freeze.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    freeze_active = subparsers.add_parser(
        "freeze-active", help="freeze active resource and arm-order controls"
    )
    freeze_active.add_argument("--parent-plan", type=Path, default=DEFAULT_PLAN)
    freeze_active.add_argument("--output", type=Path, default=DEFAULT_ACTIVE_PLAN)
    parity = subparsers.add_parser("run-parity")
    parity.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parity.add_argument("--reuse-existing", action="store_true")
    parity.add_argument("--jobs", type=int, default=1)
    active = subparsers.add_parser("run-active")
    active.add_argument("--plan", type=Path, default=DEFAULT_ACTIVE_PLAN)
    active.add_argument("--reuse-existing", action="store_true")
    active.add_argument("--jobs", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        plan = build_plan(
            args.assignments.resolve(),
            args.root_manifest.resolve(),
            args.source_summary.resolve(),
            args.instances_dir.resolve(),
            args.online_model_manifest.resolve(),
            args.output.resolve(),
        )
        print(json.dumps(plan["checks"], sort_keys=True))
        return 0
    if args.command == "freeze-active":
        plan = build_active_plan(args.parent_plan.resolve(), args.output.resolve())
        print(json.dumps(plan["checks"], sort_keys=True))
        return 0
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    result = (
        run_parity_from_plan(args.plan.resolve(), args.reuse_existing, args.jobs)
        if args.command == "run-parity"
        else run_active_from_plan(args.plan.resolve(), args.reuse_existing, args.jobs)
    )
    print(json.dumps({"passed": result.get("passed"), "manifest": result.get("experiment")}, sort_keys=True))
    return 0 if args.command == "run-active" or result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
