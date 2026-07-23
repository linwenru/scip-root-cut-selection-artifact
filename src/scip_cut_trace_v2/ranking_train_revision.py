"""Train the post-review ranker without using external validation for selection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from .observational import PROJECT_ROOT
from .ranking_train import (
    DEFAULT_SEED,
    PRIMARY_VALIDATION_SUBSET,
    VALIDATION_SUBSETS,
    RankingMatrix,
    _dmatrix,
    _manifest_path,
    _sha256_file,
    anchor_scip_top_candidate,
    evaluate_scores,
    load_effective_matrix,
    subset_matrix,
    training_parameters,
)


SCHEMA_VERSION = 1
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "data" / "datasets" / "ranking_imitation_online_v1"
)
DEFAULT_DATASET_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "ranking_imitation_online_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "models" / "ranking_imitation_online_xgb_revision_v1"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "ranking_imitation_online_xgb_revision_v1.json"
)
DEFAULT_FOLDS = 5


def independent_group_keys(matrix: RankingMatrix) -> np.ndarray:
    """Return official Groups, using one unique key per ungrouped instance."""
    keys = []
    instance_keys: dict[str, str] = {}
    for official_group, instance in zip(
        matrix.group_official_groups, matrix.group_instances
    ):
        instance_name = str(instance)
        group_name = str(official_group)
        key = group_name or f"officially-ungrouped:{instance_name}"
        previous = instance_keys.setdefault(instance_name, key)
        if previous != key:
            raise ValueError(
                f"Instance {instance_name} maps to inconsistent Group keys"
            )
        keys.append(key)
    return np.asarray(keys, dtype=str)


def assign_independent_group_folds(
    matrix: RankingMatrix, n_folds: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if n_folds < 2:
        raise ValueError("At least two folds are required")
    group_keys = independent_group_keys(matrix)
    group_instances: dict[str, set[str]] = defaultdict(set)
    group_rows = defaultdict(int)
    for key, instance, size in zip(
        group_keys, matrix.group_instances, matrix.group_sizes
    ):
        group_instances[str(key)].add(str(instance))
        group_rows[str(key)] += int(size)
    if len(group_instances) < n_folds:
        raise ValueError("Fewer independent Group keys than requested folds")

    fold_groups: list[list[str]] = [[] for _ in range(n_folds)]
    fold_instances = [0] * n_folds
    fold_rows = [0] * n_folds
    ordered = sorted(
        group_instances,
        key=lambda key: (-len(group_instances[key]), -group_rows[key], key),
    )
    for key in ordered:
        fold = min(
            range(n_folds),
            key=lambda index: (fold_instances[index], fold_rows[index], index),
        )
        fold_groups[fold].append(key)
        fold_instances[fold] += len(group_instances[key])
        fold_rows[fold] += group_rows[key]

    key_to_fold = {
        key: fold for fold, keys in enumerate(fold_groups) for key in keys
    }
    fold_ids = np.asarray(
        [key_to_fold[str(key)] for key in group_keys], dtype=np.int16
    )
    summaries = [
        {
            "fold": fold,
            "group_keys": sorted(keys),
            "group_key_count": len(keys),
            "instance_count": fold_instances[fold],
            "row_count": fold_rows[fold],
        }
        for fold, keys in enumerate(fold_groups)
    ]
    return fold_ids, summaries


def _native_budget_overlap_metric(matrix: RankingMatrix):
    def metric(predictions: np.ndarray, _data: xgb.DMatrix) -> tuple[str, float]:
        anchored = anchor_scip_top_candidate(matrix, predictions)
        aggregate, _ = evaluate_scores(matrix, anchored)
        return "instance_native_budget_overlap", aggregate["selection_overlap"]

    return metric


def _fit_round_selection_fold(
    train: RankingMatrix,
    validation: RankingMatrix,
    *,
    seed: int,
    nthread: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> dict[str, Any]:
    parameters = training_parameters(seed, nthread)
    parameters["disable_default_eval_metric"] = True
    evaluation_history: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        parameters,
        _dmatrix(train, nthread),
        num_boost_round=num_boost_round,
        evals=[(_dmatrix(validation, nthread), validation.name)],
        custom_metric=_native_budget_overlap_metric(validation),
        early_stopping_rounds=early_stopping_rounds,
        maximize=True,
        evals_result=evaluation_history,
        verbose_eval=False,
    )
    best_iteration = int(booster.best_iteration)
    return {
        "best_iteration_zero_based": best_iteration,
        "selected_boost_rounds": best_iteration + 1,
        "best_instance_native_budget_overlap": float(booster.best_score),
        "rounds_evaluated": len(
            evaluation_history[validation.name][
                "instance_native_budget_overlap"
            ]
        ),
    }


def select_boost_rounds_inside_training(
    matrix: RankingMatrix,
    *,
    n_folds: int,
    seed: int,
    nthread: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> dict[str, Any]:
    fold_ids, summaries = assign_independent_group_folds(matrix, n_folds)
    fold_results = []
    selected_rounds = []
    for fold in range(n_folds):
        validation_mask = fold_ids == fold
        train = subset_matrix(matrix, ~validation_mask, f"inner_{fold}_train")
        validation = subset_matrix(
            matrix, validation_mask, f"inner_{fold}_validation"
        )
        fit = _fit_round_selection_fold(
            train,
            validation,
            seed=seed + fold,
            nthread=nthread,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        selected_rounds.append(fit["selected_boost_rounds"])
        fold_results.append(
            {
                **summaries[fold],
                "training_instances": len(set(train.group_instances.tolist())),
                "training_queries": train.n_groups,
                "validation_queries": validation.n_groups,
                "fit": fit,
            }
        )
    frozen_rounds = int(np.median(np.asarray(selected_rounds, dtype=np.int64)))
    return {
        "method": (
            "five-fold independent-Group internal selection; maximize anchored "
            "instance-equal recall at each query's observed native cut budget; "
            "freeze the median selected round"
        ),
        "folds": fold_results,
        "selected_rounds_by_fold": selected_rounds,
        "frozen_boost_rounds": frozen_rounds,
        "external_validation_used": False,
    }


def _instance_cluster_map(matrix: RankingMatrix) -> dict[str, str]:
    mapping = {}
    for key, instance in zip(independent_group_keys(matrix), matrix.group_instances):
        name = str(instance)
        previous = mapping.setdefault(name, str(key))
        if previous != str(key):
            raise ValueError(f"Instance {name} spans multiple bootstrap clusters")
    return mapping


def cluster_bootstrap_delta(
    baseline: dict[str, dict[str, float]],
    model: dict[str, dict[str, float]],
    instance_clusters: dict[str, str],
    metric: str,
    *,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    instances = sorted(set(baseline) & set(model))
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for instance in instances:
        by_cluster[instance_clusters[instance]].append(
            model[instance][metric] - baseline[instance][metric]
        )
    clusters = sorted(by_cluster)
    if not clusters:
        raise ValueError("Cannot bootstrap an empty evaluation")
    point_values = np.asarray(
        [value for cluster in clusters for value in by_cluster[cluster]],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        selected = generator.integers(0, len(clusters), size=len(clusters))
        values = [
            value
            for index in selected
            for value in by_cluster[clusters[int(index)]]
        ]
        bootstrap[sample] = float(np.mean(values))
    return {
        "delta": float(point_values.mean()),
        "ci95_lower": float(np.quantile(bootstrap, 0.025)),
        "ci95_upper": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_probability_positive": float(np.mean(bootstrap > 0.0)),
        "instances": len(instances),
        "clusters": len(clusters),
    }


def compare_with_group_clustered_baseline(
    matrix: RankingMatrix,
    model_scores: np.ndarray,
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    baseline_scores = -matrix.baseline_score_rank_pre.astype(np.float64)
    baseline, baseline_by_instance = evaluate_scores(matrix, baseline_scores)
    model, model_by_instance = evaluate_scores(matrix, model_scores)
    instance_clusters = _instance_cluster_map(matrix)
    comparisons = {
        metric: cluster_bootstrap_delta(
            baseline_by_instance,
            model_by_instance,
            instance_clusters,
            metric,
            seed=seed + index,
            samples=bootstrap_samples,
        )
        for index, metric in enumerate(baseline)
    }
    return {
        "instances": len(baseline_by_instance),
        "clusters": len(set(instance_clusters.values())),
        "queries": matrix.n_groups,
        "rows": matrix.n_rows,
        "aggregation": (
            "mean queries within instance, equal-weight instances; bootstrap "
            "resamples independent official-Group keys"
        ),
        "baseline": baseline,
        "model": model,
        "comparison": comparisons,
        "per_instance": {
            instance: {
                "cluster": instance_clusters[instance],
                "baseline": baseline_by_instance[instance],
                "model": model_by_instance[instance],
                "delta": {
                    metric: model_by_instance[instance][metric]
                    - baseline_by_instance[instance][metric]
                    for metric in baseline_by_instance[instance]
                },
            }
            for instance in sorted(baseline_by_instance)
        },
    }


def _stage_gate(evaluation: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = evaluation[PRIMARY_VALIDATION_SUBSET]["comparison"]
    seen = evaluation["seen_family_val"]["comparison"]
    checks = {
        "primary_native_budget_overlap_point_improves": (
            primary["selection_overlap"]["delta"] > 0.0
        ),
        "primary_native_budget_overlap_group_ci95_excludes_zero": (
            primary["selection_overlap"]["ci95_lower"] > 0.0
        ),
        "primary_ndcg10_point_does_not_regress": (
            primary["ndcg@10"]["delta"] >= 0.0
        ),
        "seen_group_native_budget_overlap_does_not_regress": (
            seen["selection_overlap"]["delta"] >= 0.0
        ),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "consequence": (
            "eligible for a separately frozen online complete-solve protocol"
            if passed
            else "retitle as an audit/fixed-intervention study; do not unseal test"
        ),
    }


def train_revision_model(
    *,
    dataset_dir: Path,
    dataset_manifest_path: Path,
    output_dir: Path,
    manifest_path: Path,
    n_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    nthread: int = 4,
    num_boost_round: int = 1200,
    early_stopping_rounds: int = 80,
    bootstrap_samples: int = 5000,
) -> dict[str, Any]:
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if not all(dataset_manifest["checks"].values()):
        raise ValueError("Source online ranking dataset has failed checks")
    train = load_effective_matrix(dataset_dir / "train.npz", "train")
    validation = {
        subset: load_effective_matrix(dataset_dir / f"{subset}.npz", subset)
        for subset in VALIDATION_SUBSETS
    }
    round_selection = select_boost_rounds_inside_training(
        train,
        n_folds=n_folds,
        seed=seed,
        nthread=nthread,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )
    parameters = training_parameters(seed, nthread)
    booster = xgb.train(
        parameters,
        _dmatrix(train, nthread),
        num_boost_round=round_selection["frozen_boost_rounds"],
        verbose_eval=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.ubj"
    booster.save_model(model_path)
    evaluation = {}
    for index, (subset, matrix) in enumerate(validation.items()):
        scores = booster.predict(_dmatrix(matrix, nthread))
        evaluation[subset] = compare_with_group_clustered_baseline(
            matrix,
            anchor_scip_top_candidate(matrix, scores),
            seed=seed + 1000 * (index + 1),
            bootstrap_samples=bootstrap_samples,
        )
    stage_gate = _stage_gate(evaluation)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "post-review online-compatible SCIP-application imitation ranker; "
            "boosting rounds are selected only inside training Groups"
        ),
        "dataset_manifest": _manifest_path(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "dataset_checks": dataset_manifest["checks"],
        "test_policy": "no test matrix was loaded or scored",
        "model": {
            "path": _manifest_path(model_path),
            "sha256": _sha256_file(model_path),
            "best_iteration": round_selection["frozen_boost_rounds"] - 1,
            "frozen_boost_rounds": round_selection["frozen_boost_rounds"],
        },
        "training": {
            "rows": train.n_rows,
            "queries": train.n_groups,
            "instances": len(set(train.group_instances.tolist())),
            "features": len(train.feature_names),
            "parameters": parameters,
            "round_selection": round_selection,
        },
        "evaluation": evaluation,
        "stage_gate": stage_gate,
        "policy_contract": (
            "preserve SCIP rank 1, rerank the tail, and preserve the observed "
            "native selected-cut budget"
        ),
        "inference_contract": {
            "model_selection": "training Groups only",
            "external_validation": "one evaluation after the round count is frozen",
            "bootstrap_cluster": "official Group; ungrouped instances are unique keys",
            "primary_offline_metric": "selection_overlap at each query's native budget",
        },
        "runtime": {"numpy": np.__version__, "xgboost": xgb.__version__},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nthread", type=int, default=4)
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = train_revision_model(
        dataset_dir=args.dataset_dir.resolve(),
        dataset_manifest_path=args.dataset_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        manifest_path=args.manifest.resolve(),
        n_folds=args.folds,
        seed=args.seed,
        nthread=args.nthread,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        bootstrap_samples=args.bootstrap_samples,
    )
    primary = manifest["evaluation"][PRIMARY_VALIDATION_SUBSET]["comparison"]
    print(
        json.dumps(
            {
                "stage_gate": manifest["stage_gate"],
                "frozen_boost_rounds": manifest["model"]["frozen_boost_rounds"],
                "selection_overlap": primary["selection_overlap"],
                "ndcg@10": primary["ndcg@10"],
                "manifest": str(args.manifest),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
