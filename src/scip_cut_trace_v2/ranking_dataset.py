"""Audit eligible root decisions and build instance-balanced ranking matrices."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .observational import (
    CANDIDATE_MODEL_FEATURES,
    DECISION_MODEL_FEATURES,
    PROHIBITED_MODEL_FIELDS,
    PROJECT_ROOT,
    _instance_stem,
    load_assignments,
)


DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "root_observational_v1"
DEFAULT_ASSIGNMENTS = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "evaluation_protocols"
    / "instance_assignments.csv"
)
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "root_observational_v1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "datasets" / "ranking_imitation_v1"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "ranking_imitation_v1.json"
DEFAULT_ANALYSIS = PROJECT_ROOT / "data" / "manifests" / "eligible_ranking_analysis_v1.json"
MATRIX_SCHEMA_VERSION = 1

CATEGORICAL_FEATURES = ("origin_type", "pre_lp_status")
BOOLEAN_FEATURES = {
    "cutoff_distance_available",
    "is_local",
    "is_modifiable",
    "is_removable",
    "is_integral",
    "in_global_cutpool",
    "pre_state_available",
}
NUMERIC_CANDIDATE_FEATURES = tuple(
    feature for feature in CANDIDATE_MODEL_FEATURES if feature not in CATEGORICAL_FEATURES
)
NUMERIC_DECISION_FEATURES = tuple(
    feature for feature in DECISION_MODEL_FEATURES if feature not in CATEGORICAL_FEATURES
)
NUMERIC_FEATURES = NUMERIC_CANDIDATE_FEATURES + NUMERIC_DECISION_FEATURES

SUBSET_ORDER = (
    "train",
    "official_group_ood_val",
    "official_group_ood_test",
    "seen_family_val",
    "seen_family_test",
    "officially_ungrouped_val",
    "officially_ungrouped_test",
)


@dataclass(frozen=True)
class GroupRecord:
    instance_name: str
    original_split: str
    official_group: str
    evaluation_stratum: str
    decision_id: str
    run_number: int
    n_candidates: int
    n_positives: int
    decision: dict[str, str]
    candidates_path: Path


class NumericAudit:
    def __init__(self) -> None:
        self.count = 0
        self.missing = 0
        self.nonfinite = 0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, value: str) -> None:
        self.count += 1
        if value == "":
            self.missing += 1
            return
        try:
            number = float(value)
        except ValueError:
            self.missing += 1
            return
        if not math.isfinite(number):
            self.nonfinite += 1
            return
        self.minimum = min(self.minimum, number)
        self.maximum = max(self.maximum, number)

    @property
    def informative(self) -> bool:
        return self.minimum < self.maximum

    def to_dict(self) -> dict[str, int | float | None | bool]:
        finite = self.count - self.missing - self.nonfinite
        return {
            "count": self.count,
            "finite": finite,
            "missing": self.missing,
            "nonfinite": self.nonfinite,
            "minimum": self.minimum if finite else None,
            "maximum": self.maximum if finite else None,
            "informative_in_training": self.informative,
        }


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _numeric_value(feature: str, value: str) -> float:
    if value == "":
        return math.nan
    if feature in BOOLEAN_FEATURES:
        return float(_as_bool(value))
    try:
        number = float(value)
    except ValueError:
        return math.nan
    return number if math.isfinite(number) else math.nan


def _update_numeric_audit(audit: NumericAudit, feature: str, value: str) -> None:
    if feature in BOOLEAN_FEATURES and value != "":
        audit.update("1" if _as_bool(value) else "0")
    else:
        audit.update(value)


def _manifest_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _subset_name(original_split: str, evaluation_stratum: str) -> str:
    if original_split == "train":
        return "train"
    suffix = "val" if original_split == "val" else "test"
    prefix = {
        "unseen_family": "official_group_ood",
        "seen_family": "seen_family",
        "officially_ungrouped": "officially_ungrouped",
    }[evaluation_stratum]
    return f"{prefix}_{suffix}"


def discover_groups(
    processed_dir: Path, assignments: list[dict[str, str]]
) -> dict[str, list[GroupRecord]]:
    groups = {subset: [] for subset in SUBSET_ORDER}
    for assignment in assignments:
        stem = _instance_stem(assignment["instance_name"])
        instance_dir = processed_dir / assignment["original_split"] / stem
        decisions_path = instance_dir / "root_decisions.csv.gz"
        candidates_path = instance_dir / "root_candidates.csv.gz"
        if not decisions_path.is_file() or not candidates_path.is_file():
            raise FileNotFoundError(f"Missing processed tables for {assignment['instance_name']}")
        subset = _subset_name(
            assignment["original_split"], assignment["evaluation_stratum"]
        )
        with gzip.open(decisions_path, "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not _as_bool(row["is_policy_eligible_decision"]):
                    continue
                if not _as_bool(row["pre_state_available"]):
                    raise ValueError(f"Eligible decision lacks pre-state: {row['decision_id']}")
                if _as_bool(row["has_source_cut_id_collision"]):
                    raise ValueError(f"Eligible decision has cut-ID collision: {row['decision_id']}")
                n_candidates = int(row["n_logical_candidates"])
                n_positives = int(row["n_observed_applied_logical_cuts"])
                if n_candidates < 2:
                    raise ValueError(f"Eligible decision is not rankable: {row['decision_id']}")
                groups[subset].append(
                    GroupRecord(
                        instance_name=assignment["instance_name"],
                        original_split=assignment["original_split"],
                        official_group=assignment["official_group"],
                        evaluation_stratum=assignment["evaluation_stratum"],
                        decision_id=row["decision_id"],
                        run_number=int(row["run_number"]),
                        n_candidates=n_candidates,
                        n_positives=n_positives,
                        decision=row,
                        candidates_path=candidates_path,
                    )
                )
    return groups


def _iter_group_candidates(
    records: list[GroupRecord],
):
    by_path: dict[Path, dict[str, GroupRecord]] = defaultdict(dict)
    for record in records:
        by_path[record.candidates_path][record.decision_id] = record
    for path, records_by_id in by_path.items():
        with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                record = records_by_id.get(row["decision_id"])
                if record is not None:
                    yield record, row


def fit_feature_contract(train_groups: list[GroupRecord]) -> dict[str, object]:
    numeric_audit = {feature: NumericAudit() for feature in NUMERIC_FEATURES}
    categories = {feature: Counter() for feature in CATEGORICAL_FEATURES}

    for group in train_groups:
        for feature in NUMERIC_DECISION_FEATURES:
            _update_numeric_audit(
                numeric_audit[feature], feature, group.decision.get(feature, "")
            )
        categories["pre_lp_status"][group.decision.get("pre_lp_status", "")] += 1

    candidate_rows = 0
    for group, row in _iter_group_candidates(train_groups):
        candidate_rows += 1
        for feature in NUMERIC_CANDIDATE_FEATURES:
            _update_numeric_audit(numeric_audit[feature], feature, row.get(feature, ""))
        categories["origin_type"][row.get("origin_type", "")] += 1

    expected_rows = sum(group.n_candidates for group in train_groups)
    if candidate_rows != expected_rows:
        raise ValueError(
            f"Training candidate count mismatch: observed={candidate_rows} expected={expected_rows}"
        )

    selected_numeric = [
        feature for feature in NUMERIC_FEATURES if numeric_audit[feature].informative
    ]
    dropped_numeric = [feature for feature in NUMERIC_FEATURES if feature not in selected_numeric]
    category_vocabulary = {}
    dropped_categorical = []
    for feature in CATEGORICAL_FEATURES:
        observed = sorted(value for value in categories[feature] if value)
        if len(observed) < 2:
            dropped_categorical.append(feature)
        else:
            category_vocabulary[feature] = observed + ["<UNK>"]

    encoded_names = list(selected_numeric)
    for feature, values in category_vocabulary.items():
        encoded_names.extend(f"{feature}=={value}" for value in values)

    return {
        "fit_scope": "eligible decisions from original training split only",
        "candidate_rows_seen": candidate_rows,
        "decision_groups_seen": len(train_groups),
        "numeric_features": selected_numeric,
        "category_vocabulary": category_vocabulary,
        "encoded_feature_names": encoded_names,
        "dropped_constant_or_empty_numeric": dropped_numeric,
        "dropped_single_category": dropped_categorical,
        "numeric_audit": {
            feature: audit.to_dict() for feature, audit in numeric_audit.items()
        },
        "category_counts": {
            feature: dict(counts) for feature, counts in categories.items()
        },
    }


def _quantiles(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("min", "p25", "median", "p75", "p90", "p95", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": int(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.9)),
        "p95": float(np.quantile(array, 0.95)),
        "max": int(array.max()),
    }


def analyze_subset(
    groups: list[GroupRecord], include_label_statistics: bool = True
) -> dict[str, object]:
    candidate_by_instance = Counter()
    positive_by_instance = Counter()
    decisions_by_instance = Counter()
    total_candidates = 0
    total_positives = 0
    for group in groups:
        candidate_by_instance[group.instance_name] += group.n_candidates
        positive_by_instance[group.instance_name] += group.n_positives
        decisions_by_instance[group.instance_name] += 1
        total_candidates += group.n_candidates
        total_positives += group.n_positives
    ranked = candidate_by_instance.most_common()
    shares = [count / total_candidates for _, count in ranked] if total_candidates else []
    top_instances = []
    for name, count in ranked[:20]:
        record = {
            "instance_name": name,
            "candidate_count": count,
            "candidate_share": count / total_candidates,
            "decision_count": decisions_by_instance[name],
        }
        if include_label_statistics:
            record["positive_count"] = positive_by_instance[name]
        top_instances.append(record)
    result = {
        "instances": len(candidate_by_instance),
        "decision_groups": len(groups),
        "candidates": total_candidates,
        "candidate_count_quantiles": _quantiles([group.n_candidates for group in groups]),
        "top_1_instance_candidate_share": sum(shares[:1]),
        "top_5_instance_candidate_share": sum(shares[:5]),
        "top_10_instance_candidate_share": sum(shares[:10]),
        "effective_instance_count_by_candidates": (
            1.0 / sum(share * share for share in shares) if shares else 0.0
        ),
        "top_instances_by_candidates": top_instances,
    }
    if include_label_statistics:
        result.update(
            {
                "positives": total_positives,
                "positive_rate": total_positives / total_candidates if total_candidates else None,
                "zero_positive_groups": sum(group.n_positives == 0 for group in groups),
                "all_positive_groups": sum(
                    group.n_positives == group.n_candidates for group in groups
                ),
                "effective_pair_groups": sum(
                    0 < group.n_positives < group.n_candidates for group in groups
                ),
                "positive_count_quantiles": _quantiles(
                    [group.n_positives for group in groups]
                ),
            }
        )
    else:
        result["label_statistics"] = "sealed"
    return result


def _matrix_group_weights(
    groups: list[GroupRecord], active: np.ndarray | None = None
) -> np.ndarray:
    if active is None:
        active = np.ones(len(groups), dtype=np.bool_)
    decisions_per_instance = Counter(
        group.instance_name for group, enabled in zip(groups, active) if enabled
    )
    n_instances = len(decisions_per_instance)
    n_groups = int(active.sum())
    if not groups:
        return np.empty(0, dtype=np.float32)
    return np.asarray(
        [
            (
                n_groups / (n_instances * decisions_per_instance[group.instance_name])
                if enabled
                else 0.0
            )
            for group, enabled in zip(groups, active)
        ],
        dtype=np.float32,
    )


def _weights_are_instance_balanced(
    groups: list[GroupRecord], weights: np.ndarray, active: np.ndarray
) -> tuple[bool, float]:
    weight_sum_by_instance = defaultdict(float)
    for group, weight, enabled in zip(groups, weights, active):
        if enabled:
            weight_sum_by_instance[group.instance_name] += float(weight)
    target = float(active.sum()) / len(weight_sum_by_instance) if weight_sum_by_instance else 0.0
    balanced = all(
        math.isclose(weight, target, rel_tol=1e-6, abs_tol=1e-6)
        for weight in weight_sum_by_instance.values()
    )
    return balanced, target


def _group_offsets(groups: list[GroupRecord]) -> np.ndarray:
    sizes = np.asarray([group.n_candidates for group in groups], dtype=np.int64)
    return np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(sizes)))


def build_matrix(
    subset: str,
    groups: list[GroupRecord],
    feature_contract: dict[str, object],
    output_path: Path,
    summarize_labels: bool = True,
) -> dict[str, object]:
    numeric_features = feature_contract["numeric_features"]
    category_vocabulary = feature_contract["category_vocabulary"]
    feature_names = feature_contract["encoded_feature_names"]
    offsets = _group_offsets(groups)
    n_rows = int(offsets[-1])
    n_features = len(feature_names)
    X = np.full((n_rows, n_features), np.nan, dtype=np.float32)
    if category_vocabulary:
        X[:, len(numeric_features) :] = 0.0
    y = np.empty(n_rows, dtype=np.uint8)
    qid = np.empty(n_rows, dtype=np.int32)
    logical_candidate_id = np.empty(n_rows, dtype="U64")
    source_cut_id = np.empty(n_rows, dtype="U32")
    baseline_score_rank_pre = np.empty(n_rows, dtype=np.int32)
    positions = np.zeros(len(groups), dtype=np.int64)
    group_index = {group.decision_id: index for index, group in enumerate(groups)}

    category_offsets = {}
    offset = len(numeric_features)
    for feature, values in category_vocabulary.items():
        category_offsets[feature] = (offset, {value: i for i, value in enumerate(values)})
        offset += len(values)

    for group, candidate in _iter_group_candidates(groups):
        index = group_index[group.decision_id]
        row_index = int(offsets[index] + positions[index])
        positions[index] += 1
        combined = {**group.decision, **candidate}
        for feature_index, feature in enumerate(numeric_features):
            X[row_index, feature_index] = _numeric_value(feature, combined.get(feature, ""))
        for feature, (category_offset, vocabulary) in category_offsets.items():
            value = combined.get(feature, "")
            category_index = vocabulary.get(value, vocabulary["<UNK>"])
            X[row_index, category_offset + category_index] = 1.0
        y[row_index] = _as_bool(candidate["observed_logical_is_applied"])
        qid[row_index] = index
        logical_candidate_id[row_index] = candidate["logical_candidate_id"]
        source_cut_id[row_index] = candidate["source_cut_id"]
        baseline_score_rank_pre[row_index] = int(candidate["score_rank_pre"])

    expected_sizes = np.diff(offsets)
    if not np.array_equal(positions, expected_sizes):
        raise ValueError(
            f"Candidate rows do not match group sizes for {subset}: "
            f"observed={positions.tolist()} expected={expected_sizes.tolist()}"
        )

    all_groups_active = np.ones(len(groups), dtype=np.bool_)
    group_has_effective_pair = np.asarray(
        [0 < group.n_positives < group.n_candidates for group in groups], dtype=np.bool_
    )
    group_weights = _matrix_group_weights(groups, all_groups_active)
    effective_group_weights = _matrix_group_weights(groups, group_has_effective_pair)
    weights_balanced, target_instance_weight = _weights_are_instance_balanced(
        groups, group_weights, all_groups_active
    )
    effective_weights_balanced, effective_target_instance_weight = (
        _weights_are_instance_balanced(groups, effective_group_weights, group_has_effective_pair)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        qid=qid,
        group_ptr=offsets,
        group_sizes=expected_sizes.astype(np.int32),
        group_weight=group_weights,
        group_has_effective_pair=group_has_effective_pair,
        effective_group_weight=effective_group_weights,
        feature_names=np.asarray(feature_names, dtype=str),
        logical_candidate_id=logical_candidate_id,
        source_cut_id=source_cut_id,
        baseline_score_rank_pre=baseline_score_rank_pre,
        group_decision_id=np.asarray([group.decision_id for group in groups], dtype=str),
        group_instance_name=np.asarray([group.instance_name for group in groups], dtype=str),
        group_official_group=np.asarray([group.official_group for group in groups], dtype=str),
        group_original_split=np.asarray([group.original_split for group in groups], dtype=str),
        group_evaluation_stratum=np.asarray(
            [group.evaluation_stratum for group in groups], dtype=str
        ),
        group_run_number=np.asarray([group.run_number for group in groups], dtype=np.int32),
    )

    result = {
        "path": _manifest_path(output_path),
        "sha256": _sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "rows": n_rows,
        "features": n_features,
        "groups": len(groups),
        "instances": len({group.instance_name for group in groups}),
        "group_weights_instance_balanced": weights_balanced,
        "target_total_weight_per_instance": target_instance_weight,
        "effective_group_weights_instance_balanced": effective_weights_balanced,
        "effective_target_total_weight_per_instance": effective_target_instance_weight,
    }
    if summarize_labels:
        positives = int(y.sum())
        top_capture = {}
        for k in (1, 5, 10, 20):
            captured = int(y[baseline_score_rank_pre <= k].sum())
            top_capture[f"top_{k}"] = {
                "captured_positives": captured,
                "positive_recall": captured / positives if positives else None,
            }
        result.update(
            {
                "positives": positives,
                "positive_rate": positives / n_rows if n_rows else None,
                "effective_pair_groups": int(group_has_effective_pair.sum()),
                "baseline_score_top_k_positive_capture": top_capture,
            }
        )
    else:
        result["label_statistics"] = "sealed"
    return result


def build_ranking_dataset(
    processed_dir: Path,
    assignments_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
    manifest_path: Path,
    analysis_path: Path,
) -> dict[str, object]:
    assignments = load_assignments(assignments_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    groups = discover_groups(processed_dir, assignments)
    feature_contract = fit_feature_contract(groups["train"])
    prohibited_overlap = sorted(
        set(feature_contract["encoded_feature_names"]) & set(PROHIBITED_MODEL_FIELDS)
    )
    if prohibited_overlap:
        raise ValueError(f"Prohibited fields entered matrix: {prohibited_overlap}")

    analysis = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "source_manifest": _manifest_path(source_manifest_path),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "subsets": {
            subset: analyze_subset(
                groups[subset], include_label_statistics=not subset.endswith("_test")
            )
            for subset in SUBSET_ORDER
        },
        "feature_contract": feature_contract,
        "interpretation": (
            "observational imitation data only; label means SCIP application, not solve benefit"
        ),
    }
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    matrices = {}
    for subset in SUBSET_ORDER:
        matrices[subset] = build_matrix(
            subset,
            groups[subset],
            feature_contract,
            output_dir / f"{subset}.npz",
            summarize_labels=not subset.endswith("_test"),
        )
    checks = {
        "all_eligible_groups_partitioned": sum(len(value) for value in groups.values())
        == source_manifest["totals"]["policy_eligible_decisions"],
        "all_matrices_instance_balanced": all(
            matrix["group_weights_instance_balanced"] for matrix in matrices.values()
        ),
        "all_effective_group_weights_instance_balanced": all(
            matrix["effective_group_weights_instance_balanced"]
            for matrix in matrices.values()
        ),
        "no_prohibited_model_fields": not prohibited_overlap,
        "test_matrices_constructed_but_not_scored": True,
        "matrix_feature_count_matches_contract": all(
            matrix["features"] == len(feature_contract["encoded_feature_names"])
            for matrix in matrices.values()
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Ranking dataset checks failed: {failed}")
    manifest = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "source_manifest": _manifest_path(source_manifest_path),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "analysis": _manifest_path(analysis_path),
        "analysis_sha256": _sha256_file(analysis_path),
        "output_dir": _manifest_path(output_dir),
        "feature_contract": feature_contract,
        "group_weight_semantics": (
            "one XGBoost ranking weight per decision; normalized so each instance has equal "
            "total weight and mean group weight is one"
        ),
        "test_policy": "matrices constructed and sealed; no test metric computed",
        "matrices": matrices,
        "checks": checks,
        "numpy_version": np.__version__,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_ranking_dataset(
        args.processed_dir.resolve(),
        args.assignments.resolve(),
        args.source_manifest.resolve(),
        args.output_dir.resolve(),
        args.manifest.resolve(),
        args.analysis.resolve(),
    )
    print(json.dumps(manifest["checks"], sort_keys=True))
    for subset, matrix in manifest["matrices"].items():
        print(
            f"{subset}: groups={matrix['groups']} rows={matrix['rows']} "
            f"features={matrix['features']} "
            f"positives={matrix.get('positives', 'sealed')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
