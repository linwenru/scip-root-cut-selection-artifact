"""Build leakage-safe root-node observational tables from supplied traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from .audit import SPLITS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSIGNMENTS = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "evaluation_protocols"
    / "instance_assignments.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "root_observational_v1"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "root_observational_v1.json"
SCHEMA_VERSION = 1
BUILDER_REVISION = 3

CANDIDATE_COPY_FIELDS = (
    "cut_name",
    "origin_type",
    "score",
    "nnz",
    "rhs",
    "lhs",
    "constant",
    "coeff_norm_l2",
    "coeff_norm_l1",
    "coeff_max_abs",
    "coeff_min_abs",
    "coeff_mean_abs",
    "coeff_std_abs",
    "efficacy",
    "obj_parallelism",
    "cutoff_distance",
    "n_int_cols",
    "is_local",
    "is_modifiable",
    "is_removable",
    "is_integral",
    "in_global_cutpool",
)
CANDIDATE_SIGNATURE_FIELDS = CANDIDATE_COPY_FIELDS
CANDIDATE_MODEL_FEATURES = (
    "score_rank_pre",
    "score_rank_fraction_pre",
    "score",
    "nnz",
    "rhs",
    "lhs",
    "constant",
    "coeff_norm_l2",
    "coeff_norm_l1",
    "coeff_max_abs",
    "coeff_min_abs",
    "coeff_mean_abs",
    "coeff_std_abs",
    "efficacy",
    "obj_parallelism",
    "cutoff_distance",
    "cutoff_distance_available",
    "n_int_cols",
    "is_local",
    "is_modifiable",
    "is_removable",
    "is_integral",
    "in_global_cutpool",
    "origin_type",
)
CANDIDATE_OUTPUT_FIELDS = (
    "instance_name",
    "original_split",
    "official_group",
    "evaluation_stratum",
    "decision_id",
    "logical_candidate_id",
    "source_cut_id",
    "source_signature_sha256",
    "candidate_multiplicity",
    "first_original_index",
    "source_applied_labels_consistent",
    "source_cut_id_collision",
    "observed_label_ambiguous",
    "observed_logical_is_applied",
    "is_policy_eligible_decision",
    "score_rank_pre",
    "score_rank_fraction_pre",
    "cutoff_distance_available",
) + CANDIDATE_COPY_FIELDS

PRE_STATE_FIELDS = (
    "pre_lp_status",
    "n_lp_rows_pre",
    "n_lp_cols_pre",
    "lp_obj_val_pre",
    "lp_iterations_total_pre",
    "lp_iterations_node_pre",
    "dual_bound_pre",
    "primal_bound_pre",
    "gap_pre",
    "n_cuts_applied_pre",
    "n_cuts_generated_node_pre",
    "n_open_nodes_pre",
)
DECISION_MODEL_FEATURES = (
    "run_number",
    "sep_round_node",
    "sep_round_run",
    "sep_round_global",
    "lp_round_node",
    "lp_round_run",
    "lp_round_global",
    "n_candidate_occurrences",
    "n_logical_candidates",
    "pre_state_available",
) + PRE_STATE_FIELDS
DECISION_OUTPUT_FIELDS = (
    "instance_name",
    "original_split",
    "official_group",
    "evaluation_stratum",
    "decision_id",
    "run_number",
    "node_number",
    "node_depth",
    "sep_round_node",
    "sep_round_run",
    "sep_round_global",
    "lp_round_node",
    "lp_round_run",
    "lp_round_global",
    "is_first_root_decision_in_run",
    "is_policy_eligible_decision",
    "n_candidate_occurrences",
    "n_logical_candidates",
    "n_forced_cuts",
    "n_duplicate_occurrences",
    "n_observed_applied_logical_cuts",
    "has_source_cut_id_collision",
    "pre_state_available",
) + PRE_STATE_FIELDS

PROHIBITED_MODEL_FIELDS = (
    "rank",
    "is_selected",
    "is_forced",
    "is_applied",
    "observed_logical_is_applied",
    "original_index",
    "lp_position",
    "coeff_sparsity_ratio",
    "solving_time",
    "post_lp_status",
    "n_lp_rows_post",
    "n_lp_cols_post",
    "lp_obj_val_post",
    "delta_lp_obj_val",
    "lp_obj_improvement_ratio",
    "delta_lp_iterations_total",
    "dual_bound_post",
    "delta_dual_bound",
    "primal_bound_post",
    "delta_primal_bound",
    "gap_post",
    "delta_gap",
    "relative_gap_improvement",
    "n_cuts_applied_post",
    "delta_n_cuts_applied",
)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _as_int(value: str) -> int:
    return int(float(value))


def _score(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return result if math.isfinite(result) else -math.inf


def _instance_stem(instance_name: str) -> str:
    for suffix in (".mps.gz", ".mps"):
        if instance_name.endswith(suffix):
            return instance_name[: -len(suffix)]
    return instance_name


def _manifest_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _decision_key(row: dict[str, str]) -> tuple[int, int, int]:
    return (
        _as_int(row["run_number"]),
        _as_int(row["node_number"]),
        _as_int(row["sep_round_node"]),
    )


def _decision_id(instance_name: str, key: tuple[int, int, int]) -> str:
    run, node, sep_round = key
    return f"{instance_name}|run={run}|node={node}|sep={sep_round}"


def _signature_sha256(signature: tuple[str, ...]) -> str:
    encoded = json.dumps(signature, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _logical_candidate_id(
    decision_id: str, source_cut_id: str, source_signature_sha256: str
) -> str:
    value = (
        f"{decision_id}|source_cut_id={source_cut_id}|signature={source_signature_sha256}"
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def load_assignments(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"instance_name", "original_split", "official_group", "evaluation_stratum"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Assignment file lacks required columns: {path}")
    names = [row["instance_name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate instance in assignment file: {path}")
    return rows


def _load_pre_states(path: Path) -> tuple[dict[tuple[int, int, int], dict[str, str]], int]:
    states = {}
    duplicate_keys = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _as_int(row["node_depth"]) != 0:
                continue
            key = _decision_key(row)
            if key in states:
                duplicate_keys += 1
                continue
            states[key] = {field: row.get(field, "") for field in PRE_STATE_FIELDS}
    return states, duplicate_keys


def _atomic_gzip_writer(path: Path, compresslevel: int):
    temporary = path.with_name(path.name + ".tmp")
    handle = gzip.open(
        temporary,
        "wt",
        newline="",
        encoding="utf-8",
        compresslevel=compresslevel,
    )
    return temporary, handle


def _source_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _completed_summary_matches(summary_path: Path, candidate_path: Path) -> bool:
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("schema_version") == SCHEMA_VERSION
        and summary.get("builder_revision") == BUILDER_REVISION
        and summary.get("source_candidate") == _source_signature(candidate_path)
    )


def _collapse_group(
    rows: list[dict[str, str]],
    assignment: dict[str, str],
    pre_state: dict[str, str] | None,
    first_root_decision: bool,
    eligible_already_used: bool,
) -> tuple[dict[str, object], list[dict[str, object]], Counter]:
    counts = Counter()
    first = rows[0]
    key = _decision_key(first)
    decision_id = _decision_id(assignment["instance_name"], key)
    optional_rows = [row for row in rows if not _as_bool(row["is_forced"])]
    forced_rows = [row for row in rows if _as_bool(row["is_forced"])]
    grouped_by_cut_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in optional_rows:
        source_cut_id = row["cut_id"]
        if not source_cut_id:
            source_cut_id = f"missing@{row['original_index']}"
            counts["missing_source_cut_id_rows"] += 1
        grouped_by_cut_id[source_cut_id].append(row)

    logical_rows = []
    has_cut_id_collision = False
    collision_cut_ids = 0
    for source_cut_id, cut_id_occurrences in grouped_by_cut_id.items():
        by_signature: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in cut_id_occurrences:
            signature = tuple(row.get(field, "") for field in CANDIDATE_SIGNATURE_FIELDS)
            by_signature[signature].append(row)
        cut_id_collision = len(by_signature) > 1
        has_cut_id_collision |= cut_id_collision
        collision_cut_ids += cut_id_collision
        for signature, occurrences in by_signature.items():
            representative = min(occurrences, key=lambda row: _as_int(row["original_index"]))
            applied_values = {_as_bool(row["is_applied"]) for row in occurrences}
            logical_rows.append(
                {
                    "representative": representative,
                    "source_cut_id": source_cut_id,
                    "source_signature_sha256": _signature_sha256(signature),
                    "multiplicity": len(occurrences),
                    "cut_id_collision": cut_id_collision,
                    "labels_consistent": len(applied_values) == 1,
                    "observed_applied": any(applied_values),
                }
            )

    logical_rows.sort(
        key=lambda item: (
            -_score(item["representative"]["score"]),
            _as_int(item["representative"]["original_index"]),
            item["source_cut_id"],
        )
    )
    is_policy_eligible = (
        not eligible_already_used
        and len(logical_rows) >= 2
        and not has_cut_id_collision
        and pre_state is not None
    )
    output_candidates = []
    denominator = max(len(logical_rows) - 1, 1)
    for rank, item in enumerate(logical_rows, start=1):
        representative = item["representative"]
        record = {
            "instance_name": assignment["instance_name"],
            "original_split": assignment["original_split"],
            "official_group": assignment["official_group"],
            "evaluation_stratum": assignment["evaluation_stratum"],
            "decision_id": decision_id,
            "logical_candidate_id": _logical_candidate_id(
                decision_id, item["source_cut_id"], item["source_signature_sha256"]
            ),
            "source_cut_id": item["source_cut_id"],
            "source_signature_sha256": item["source_signature_sha256"],
            "candidate_multiplicity": item["multiplicity"],
            "first_original_index": _as_int(representative["original_index"]),
            "source_applied_labels_consistent": item["labels_consistent"],
            "source_cut_id_collision": item["cut_id_collision"],
            "observed_label_ambiguous": item["cut_id_collision"],
            "observed_logical_is_applied": item["observed_applied"],
            "is_policy_eligible_decision": is_policy_eligible,
            "score_rank_pre": rank,
            "score_rank_fraction_pre": (rank - 1) / denominator,
            "cutoff_distance_available": bool(representative.get("cutoff_distance", "")),
        }
        record.update({field: representative.get(field, "") for field in CANDIDATE_COPY_FIELDS})
        output_candidates.append(record)

    state = pre_state or {field: "" for field in PRE_STATE_FIELDS}
    decision = {
        "instance_name": assignment["instance_name"],
        "original_split": assignment["original_split"],
        "official_group": assignment["official_group"],
        "evaluation_stratum": assignment["evaluation_stratum"],
        "decision_id": decision_id,
        "run_number": key[0],
        "node_number": key[1],
        "node_depth": _as_int(first["node_depth"]),
        "sep_round_node": key[2],
        "sep_round_run": _as_int(first["sep_round_run"]),
        "sep_round_global": _as_int(first["sep_round_global"]),
        "lp_round_node": _as_int(first["lp_round_node"]),
        "lp_round_run": _as_int(first["lp_round_run"]),
        "lp_round_global": _as_int(first["lp_round_global"]),
        "is_first_root_decision_in_run": first_root_decision,
        "is_policy_eligible_decision": is_policy_eligible,
        "n_candidate_occurrences": len(optional_rows),
        "n_logical_candidates": len(logical_rows),
        "n_forced_cuts": len(forced_rows),
        "n_duplicate_occurrences": len(optional_rows) - len(logical_rows),
        "n_observed_applied_logical_cuts": sum(
            item["observed_applied"] for item in logical_rows
        ),
        "has_source_cut_id_collision": has_cut_id_collision,
        "pre_state_available": pre_state is not None,
        **state,
    }
    counts["root_candidate_occurrences"] += len(optional_rows)
    counts["root_forced_rows"] += len(forced_rows)
    counts["logical_candidates"] += len(logical_rows)
    counts["duplicate_occurrences_collapsed"] += len(optional_rows) - len(logical_rows)
    counts["decisions_with_exact_duplicate_candidates"] += any(
        item["multiplicity"] > 1 for item in logical_rows
    )
    counts["decisions_with_source_cut_id_collision"] += has_cut_id_collision
    counts["source_cut_id_collisions"] += collision_cut_ids
    counts["source_cut_id_collision_candidates"] += sum(
        item["cut_id_collision"] for item in logical_rows
    )
    counts["logical_applied_labels"] += decision["n_observed_applied_logical_cuts"]
    counts["unambiguous_logical_applied_labels"] += sum(
        item["observed_applied"] and not item["cut_id_collision"] for item in logical_rows
    )
    counts["inconsistent_duplicate_label_candidates"] += sum(
        not item["labels_consistent"] for item in logical_rows
    )
    counts["missing_cutoff_distance_candidates"] += sum(
        not record["cutoff_distance_available"] for record in output_candidates
    )
    counts["policy_eligible_decisions"] += is_policy_eligible
    counts["policy_eligible_candidates"] += len(logical_rows) if is_policy_eligible else 0
    counts["policy_eligible_applied_labels"] += (
        decision["n_observed_applied_logical_cuts"] if is_policy_eligible else 0
    )
    counts["policy_eligible_decisions_without_pre_state"] += (
        is_policy_eligible and pre_state is None
    )
    counts["policy_eligible_decisions_with_cut_id_collision"] += (
        is_policy_eligible and has_cut_id_collision
    )
    counts["decisions_with_pre_state"] += pre_state is not None
    return decision, output_candidates, counts


def build_instance(
    source_instance_dir: Path,
    output_instance_dir: Path,
    assignment: dict[str, str],
    compresslevel: int = 1,
) -> dict[str, object]:
    candidate_path = source_instance_dir / "candidate_cuts.csv"
    transition_path = source_instance_dir / "sep_round_transitions.csv"
    if not candidate_path.is_file() or not transition_path.is_file():
        raise FileNotFoundError(f"Missing source CSV in {source_instance_dir}")
    output_instance_dir.mkdir(parents=True, exist_ok=True)
    candidates_output = output_instance_dir / "root_candidates.csv.gz"
    decisions_output = output_instance_dir / "root_decisions.csv.gz"
    candidates_tmp, candidates_handle = _atomic_gzip_writer(candidates_output, compresslevel)
    decisions_tmp, decisions_handle = _atomic_gzip_writer(decisions_output, compresslevel)

    started = time.time()
    counts = Counter()
    pre_states, duplicate_transition_keys = _load_pre_states(transition_path)
    candidate_writer = csv.DictWriter(
        candidates_handle, fieldnames=CANDIDATE_OUTPUT_FIELDS, lineterminator="\n"
    )
    decision_writer = csv.DictWriter(
        decisions_handle, fieldnames=DECISION_OUTPUT_FIELDS, lineterminator="\n"
    )
    candidate_writer.writeheader()
    decision_writer.writeheader()
    current_key = None
    current_rows: list[dict[str, str]] = []
    seen_decision_keys = set()
    root_seen_runs = set()
    eligible_used_runs = set()

    def flush() -> None:
        nonlocal current_key, current_rows
        if not current_rows:
            return
        if current_key in seen_decision_keys:
            raise ValueError(f"Non-contiguous duplicate decision key: {current_key}")
        seen_decision_keys.add(current_key)
        run = current_key[0]
        decision, candidates, group_counts = _collapse_group(
            current_rows,
            assignment,
            pre_states.get(current_key),
            run not in root_seen_runs,
            run in eligible_used_runs,
        )
        root_seen_runs.add(run)
        if decision["is_policy_eligible_decision"]:
            eligible_used_runs.add(run)
        decision_writer.writerow(decision)
        candidate_writer.writerows(candidates)
        counts.update(group_counts)
        counts["root_decisions"] += 1
        current_key = None
        current_rows = []

    try:
        with candidate_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "run_number",
                "node_number",
                "node_depth",
                "sep_round_node",
                "root",
                "cut_id",
                "is_forced",
                "is_applied",
                "original_index",
                "score",
            } | set(CANDIDATE_COPY_FIELDS)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Candidate schema mismatch: {candidate_path}")
            for row in reader:
                counts["source_candidate_rows"] += 1
                is_root = _as_bool(row["root"])
                depth_is_root = _as_int(row["node_depth"]) == 0
                counts["root_flag_depth_mismatches"] += is_root != depth_is_root
                if not is_root or not depth_is_root:
                    flush()
                    continue
                counts["source_root_rows"] += 1
                key = _decision_key(row)
                if current_key != key:
                    flush()
                    current_key = key
                current_rows.append(row)
        flush()
        candidates_handle.close()
        decisions_handle.close()
        os.replace(candidates_tmp, candidates_output)
        os.replace(decisions_tmp, decisions_output)
    except Exception:
        candidates_handle.close()
        decisions_handle.close()
        candidates_tmp.unlink(missing_ok=True)
        decisions_tmp.unlink(missing_ok=True)
        raise

    summary = {
        "schema_version": SCHEMA_VERSION,
        "builder_revision": BUILDER_REVISION,
        "instance_name": assignment["instance_name"],
        "original_split": assignment["original_split"],
        "official_group": assignment["official_group"],
        "evaluation_stratum": assignment["evaluation_stratum"],
        "source_candidate": _source_signature(candidate_path),
        "source_transition": _source_signature(transition_path),
        "duplicate_root_transition_keys": duplicate_transition_keys,
        "counts": dict(counts),
        "elapsed_seconds": round(time.time() - started, 3),
        "output_bytes": {
            "root_candidates.csv.gz": candidates_output.stat().st_size,
            "root_decisions.csv.gz": decisions_output.stat().st_size,
        },
    }
    (output_instance_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _aggregate_protocol_counts(
    assignments: list[dict[str, str]], summaries: dict[str, dict[str, object]]
) -> dict[str, dict[str, Counter]]:
    result: dict[str, dict[str, Counter]] = {
        protocol: {split: Counter() for split in SPLITS}
        for protocol in ("official_group_ood", "seen_family", "officially_ungrouped")
    }
    stratum_by_protocol = {
        "official_group_ood": "unseen_family",
        "seen_family": "seen_family",
        "officially_ungrouped": "officially_ungrouped",
    }
    for assignment in assignments:
        split = assignment["original_split"]
        summary = summaries[assignment["instance_name"]]
        counts = summary["counts"]
        for protocol, target_stratum in stratum_by_protocol.items():
            if split == "train" or assignment["evaluation_stratum"] == target_stratum:
                result[protocol][split].update(counts)
                result[protocol][split]["instances"] += 1
    return {
        protocol: {split: dict(counts) for split, counts in by_split.items()}
        for protocol, by_split in result.items()
    }


def schema_contract() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "unit_of_candidate_row": "one logical source cut_id within one root decision",
        "unit_of_decision_row": "one root cut-selector callback",
        "candidate_columns": list(CANDIDATE_OUTPUT_FIELDS),
        "decision_columns": list(DECISION_OUTPUT_FIELDS),
        "candidate_model_features": list(CANDIDATE_MODEL_FEATURES),
        "decision_model_features": list(DECISION_MODEL_FEATURES),
        "categorical_model_features": ["origin_type", "pre_lp_status"],
        "observational_label": "observed_logical_is_applied",
        "label_semantics": (
            "SCIP added at least one occurrence of this logical source cut_id to the LP; "
            "this is an imitation label, not a causal benefit label."
        ),
        "prohibited_model_fields": list(PROHIBITED_MODEL_FIELDS),
        "identity_limit": (
            "logical_candidate_id is stable for this trace snapshot only because source_cut_id "
            "was produced by Python's process-randomized 32-bit hash; the observable signature "
            "separates detected within-round hash collisions."
        ),
        "policy_eligibility": (
            "the first root decision in each run with pre-state, at least two logical candidates, "
            "and no source cut-id collision"
        ),
    }


def quality_checks(
    totals: Counter, instance_count: int, expected_instance_count: int
) -> dict[str, bool]:
    model_features = set(CANDIDATE_MODEL_FEATURES) | set(DECISION_MODEL_FEATURES)
    return {
        "all_assigned_instances_built": instance_count == expected_instance_count,
        "root_row_accounting_closes": totals["source_root_rows"]
        == totals["root_candidate_occurrences"] + totals["root_forced_rows"],
        "logical_candidate_accounting_closes": totals["root_candidate_occurrences"]
        == totals["logical_candidates"] + totals["duplicate_occurrences_collapsed"],
        "root_flag_matches_node_depth": totals["root_flag_depth_mismatches"] == 0,
        "duplicate_occurrence_labels_are_consistent": totals[
            "inconsistent_duplicate_label_candidates"
        ]
        == 0,
        "eligible_decisions_have_pre_state": totals[
            "policy_eligible_decisions_without_pre_state"
        ]
        == 0,
        "eligible_decisions_have_no_cut_id_collision": totals[
            "policy_eligible_decisions_with_cut_id_collision"
        ]
        == 0,
        "model_features_exclude_prohibited_fields": model_features.isdisjoint(
            PROHIBITED_MODEL_FIELDS
        ),
    }


def build_dataset(
    source: Path,
    assignments_path: Path,
    output_dir: Path,
    manifest_path: Path,
    compresslevel: int = 1,
    resume: bool = False,
    limit: int | None = None,
) -> dict[str, object]:
    assignments = load_assignments(assignments_path)
    if limit is not None:
        assignments = assignments[:limit]
    summaries = {}
    started = time.time()
    for position, assignment in enumerate(assignments, start=1):
        split = assignment["original_split"]
        stem = _instance_stem(assignment["instance_name"])
        source_instance_dir = source / f"benchmark_output_{split}" / stem
        output_instance_dir = output_dir / split / stem
        summary_path = output_instance_dir / "build_summary.json"
        candidate_path = source_instance_dir / "candidate_cuts.csv"
        if resume and _completed_summary_matches(summary_path, candidate_path):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            action = "reused"
        else:
            summary = build_instance(
                source_instance_dir,
                output_instance_dir,
                assignment,
                compresslevel=compresslevel,
            )
            action = "built"
        summaries[assignment["instance_name"]] = summary
        counts = summary["counts"]
        print(
            f"[{position}/{len(assignments)}] {action} {assignment['instance_name']} "
            f"root_rows={counts.get('source_root_rows', 0)} "
            f"logical={counts.get('logical_candidates', 0)}",
            flush=True,
        )

    totals = Counter()
    for summary in summaries.values():
        totals.update(summary["counts"])
    checks = quality_checks(totals, len(summaries), len(assignments))
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Observational dataset quality checks failed: {failed}")
    manifest = {
        "builder_revision": BUILDER_REVISION,
        "schema": schema_contract(),
        "source": _manifest_path(source),
        "assignments": _manifest_path(assignments_path),
        "output_dir": _manifest_path(output_dir),
        "instance_count": len(summaries),
        "elapsed_seconds_this_invocation": round(time.time() - started, 3),
        "instance_processing_seconds": round(
            sum(summary["elapsed_seconds"] for summary in summaries.values()), 3
        ),
        "totals": dict(totals),
        "quality_checks": checks,
        "instance_quality": {
            "without_root_decisions": sorted(
                name
                for name, summary in summaries.items()
                if not summary["counts"].get("root_decisions", 0)
            ),
            "without_policy_eligible_decisions": sorted(
                name
                for name, summary in summaries.items()
                if not summary["counts"].get("policy_eligible_decisions", 0)
            ),
        },
        "output_bytes": sum(
            sum(summary["output_bytes"].values()) for summary in summaries.values()
        ),
        "protocol_counts": _aggregate_protocol_counts(assignments, summaries),
        "instances": summaries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--compression-level", type=int, choices=range(1, 10), default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_dataset(
        args.source.resolve(),
        args.assignments.resolve(),
        args.output_dir.resolve(),
        args.manifest.resolve(),
        compresslevel=args.compression_level,
        resume=args.resume,
        limit=args.limit,
    )
    print(json.dumps(manifest["totals"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
