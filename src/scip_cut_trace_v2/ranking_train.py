"""Train and audit the one-intervention-per-run XGBoost imitation baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

from .observational import PROJECT_ROOT


DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "ranking_imitation_v1"
DEFAULT_DATASET_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "ranking_imitation_v1.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "ranking_imitation_xgb_v1"
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "ranking_imitation_xgb_v1.json"
)
TRAIN_SUBSET = "train"
VALIDATION_SUBSETS = (
    "official_group_ood_val",
    "seen_family_val",
    "officially_ungrouped_val",
)
PRIMARY_VALIDATION_SUBSET = "official_group_ood_val"
METRIC_K_VALUES = (1, 5, 10, 20)
MODEL_SCHEMA_VERSION = 1
DEFAULT_SEED = 20260716


@dataclass(frozen=True)
class RankingMatrix:
    name: str
    X: np.ndarray
    y: np.ndarray
    group_ptr: np.ndarray
    group_sizes: np.ndarray
    group_weights: np.ndarray
    group_instances: np.ndarray
    group_official_groups: np.ndarray
    group_decision_ids: np.ndarray
    feature_names: np.ndarray
    baseline_score_rank_pre: np.ndarray

    @property
    def n_groups(self) -> int:
        return len(self.group_sizes)

    @property
    def n_rows(self) -> int:
        return len(self.y)


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


def _active_rows(group_ptr: np.ndarray, active_groups: np.ndarray) -> np.ndarray:
    pieces = [
        np.arange(group_ptr[index], group_ptr[index + 1], dtype=np.int64)
        for index in np.flatnonzero(active_groups)
    ]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def load_effective_matrix(path: Path, name: str | None = None) -> RankingMatrix:
    """Load only queries with at least one positive and one negative label."""
    if path.stem.endswith("_test"):
        raise ValueError(f"Test matrices are sealed and cannot be loaded: {path}")
    with np.load(path) as source:
        active_groups = source["group_has_effective_pair"].astype(np.bool_)
        source_group_ptr = source["group_ptr"]
        row_indices = _active_rows(source_group_ptr, active_groups)
        group_sizes = source["group_sizes"][active_groups].astype(np.int32)
        group_ptr = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(group_sizes, dtype=np.int64))
        )
        matrix = RankingMatrix(
            name=name or path.stem,
            X=source["X"][row_indices].astype(np.float32, copy=False),
            y=source["y"][row_indices].astype(np.uint8, copy=False),
            group_ptr=group_ptr,
            group_sizes=group_sizes,
            group_weights=source["effective_group_weight"][active_groups].astype(
                np.float32, copy=False
            ),
            group_instances=source["group_instance_name"][active_groups].astype(str),
            group_official_groups=source["group_official_group"][active_groups].astype(
                str
            ),
            group_decision_ids=source["group_decision_id"][active_groups].astype(str),
            feature_names=source["feature_names"].astype(str),
            baseline_score_rank_pre=source["baseline_score_rank_pre"][row_indices].astype(
                np.int32, copy=False
            ),
        )
    _validate_matrix(matrix)
    return matrix


def _validate_matrix(matrix: RankingMatrix) -> None:
    if matrix.n_groups == 0:
        raise ValueError(f"No effective ranking queries in {matrix.name}")
    if matrix.X.shape != (matrix.n_rows, len(matrix.feature_names)):
        raise ValueError(f"Feature shape mismatch in {matrix.name}: {matrix.X.shape}")
    if matrix.group_ptr[0] != 0 or matrix.group_ptr[-1] != matrix.n_rows:
        raise ValueError(f"Invalid group boundaries in {matrix.name}")
    if not np.array_equal(np.diff(matrix.group_ptr), matrix.group_sizes):
        raise ValueError(f"Group sizes do not match boundaries in {matrix.name}")
    if len(matrix.group_weights) != matrix.n_groups:
        raise ValueError(f"Ranking weights are not per-query in {matrix.name}")
    if len(matrix.group_instances) != matrix.n_groups:
        raise ValueError(f"Instance metadata does not match queries in {matrix.name}")
    if len(matrix.group_official_groups) != matrix.n_groups:
        raise ValueError(f"Official Group metadata does not match queries in {matrix.name}")
    for start, stop in zip(matrix.group_ptr[:-1], matrix.group_ptr[1:]):
        labels = matrix.y[start:stop]
        if not labels.any() or labels.all():
            raise ValueError(f"Ineffective query remained in {matrix.name}")


def _query_metrics(labels: np.ndarray, order: np.ndarray) -> dict[str, float]:
    ranked = labels[order].astype(np.float64)
    positives = int(labels.sum())
    metrics = {}
    for k in METRIC_K_VALUES:
        limit = min(k, len(labels))
        discounts = 1.0 / np.log2(np.arange(limit, dtype=np.float64) + 2.0)
        dcg = float(np.dot(ranked[:limit], discounts))
        ideal_positives = min(positives, limit)
        idcg = float(discounts[:ideal_positives].sum())
        metrics[f"ndcg@{k}"] = dcg / idcg
        metrics[f"recall@{k}"] = float(ranked[:limit].sum()) / positives
    positive_positions = np.flatnonzero(ranked)
    precision_at_positive = (
        np.arange(1, len(positive_positions) + 1, dtype=np.float64)
        / (positive_positions + 1)
    )
    metrics["average_precision"] = float(precision_at_positive.mean())
    metrics["selection_overlap"] = float(ranked[:positives].sum()) / positives
    return metrics


def evaluate_scores(
    matrix: RankingMatrix, scores: np.ndarray
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if scores.shape != (matrix.n_rows,):
        raise ValueError(f"Score shape mismatch for {matrix.name}: {scores.shape}")
    by_instance: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for group_index, (start, stop) in enumerate(
        zip(matrix.group_ptr[:-1], matrix.group_ptr[1:])
    ):
        local_scores = scores[start:stop]
        tie_break = matrix.baseline_score_rank_pre[start:stop]
        order = np.lexsort((tie_break, -local_scores))
        query = _query_metrics(matrix.y[start:stop], order)
        instance = str(matrix.group_instances[group_index])
        for metric, value in query.items():
            by_instance[instance][metric].append(value)
    per_instance = {
        instance: {
            metric: float(np.mean(values)) for metric, values in metrics.items()
        }
        for instance, metrics in by_instance.items()
    }
    aggregate = {
        metric: float(np.mean([metrics[metric] for metrics in per_instance.values()]))
        for metric in next(iter(per_instance.values()))
    }
    return aggregate, per_instance


def anchor_scip_top_candidate(
    matrix: RankingMatrix, scores: np.ndarray
) -> np.ndarray:
    """Preserve SCIP's rank-1 candidate and rerank only the remaining tail."""
    if scores.shape != (matrix.n_rows,):
        raise ValueError(f"Score shape mismatch for {matrix.name}: {scores.shape}")
    anchored = scores.astype(np.float64, copy=True)
    for start, stop in zip(matrix.group_ptr[:-1], matrix.group_ptr[1:]):
        ranks = matrix.baseline_score_rank_pre[start:stop]
        top = np.flatnonzero(ranks == 1)
        if len(top) != 1:
            raise ValueError(f"Expected one SCIP rank-1 candidate in {matrix.name}")
        local_scores = anchored[start:stop]
        maximum = float(np.max(local_scores))
        anchored[start + int(top[0])] = maximum + max(abs(maximum), 1.0)
    return anchored


def _bootstrap_delta(
    baseline: dict[str, dict[str, float]],
    model: dict[str, dict[str, float]],
    metric: str,
    seed: int,
    samples: int,
) -> dict[str, float]:
    instances = sorted(set(baseline) & set(model))
    if not instances:
        raise ValueError("Cannot bootstrap without shared instances")
    deltas = np.asarray(
        [model[name][metric] - baseline[name][metric] for name in instances],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    batch_size = 512
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, len(deltas), size=(stop - start, len(deltas)))
        bootstrap[start:stop] = deltas[indices].mean(axis=1)
    return {
        "delta": float(deltas.mean()),
        "ci95_lower": float(np.quantile(bootstrap, 0.025)),
        "ci95_upper": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_probability_positive": float(np.mean(bootstrap > 0.0)),
        "instances": len(instances),
    }


def compare_with_baseline(
    matrix: RankingMatrix,
    model_scores: np.ndarray,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    baseline_scores = -matrix.baseline_score_rank_pre.astype(np.float64)
    baseline, baseline_by_instance = evaluate_scores(matrix, baseline_scores)
    model, model_by_instance = evaluate_scores(matrix, model_scores)
    comparisons = {
        metric: _bootstrap_delta(
            baseline_by_instance,
            model_by_instance,
            metric,
            seed + index,
            bootstrap_samples,
        )
        for index, metric in enumerate(baseline)
    }
    per_instance = {
        instance: {
            "baseline": baseline_by_instance[instance],
            "model": model_by_instance[instance],
            "delta": {
                metric: model_by_instance[instance][metric]
                - baseline_by_instance[instance][metric]
                for metric in baseline_by_instance[instance]
            },
        }
        for instance in sorted(baseline_by_instance)
    }
    return {
        "instances": len(baseline_by_instance),
        "queries": matrix.n_groups,
        "rows": matrix.n_rows,
        "aggregation": "mean query metric within instance, then mean across instances",
        "baseline": baseline,
        "model": model,
        "comparison": comparisons,
        "per_instance": per_instance,
    }


def _dmatrix(matrix: RankingMatrix, nthread: int) -> xgb.DMatrix:
    safe_feature_names = [f"f{index}" for index in range(len(matrix.feature_names))]
    return xgb.DMatrix(
        matrix.X,
        label=matrix.y,
        weight=matrix.group_weights,
        group=matrix.group_sizes,
        feature_names=safe_feature_names,
        missing=np.nan,
        nthread=nthread,
    )


def instance_balanced_weights(instances: np.ndarray) -> np.ndarray:
    counts = defaultdict(int)
    for instance in instances:
        counts[str(instance)] += 1
    if not counts:
        return np.empty(0, dtype=np.float32)
    n_queries = len(instances)
    n_instances = len(counts)
    return np.asarray(
        [n_queries / (n_instances * counts[str(instance)]) for instance in instances],
        dtype=np.float32,
    )


def subset_matrix(
    matrix: RankingMatrix, selected_groups: np.ndarray, name: str
) -> RankingMatrix:
    selected_groups = np.asarray(selected_groups, dtype=np.bool_)
    if selected_groups.shape != (matrix.n_groups,):
        raise ValueError(f"Invalid group mask for {matrix.name}")
    row_indices = _active_rows(matrix.group_ptr, selected_groups)
    group_sizes = matrix.group_sizes[selected_groups]
    group_ptr = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(group_sizes, dtype=np.int64))
    )
    group_instances = matrix.group_instances[selected_groups]
    result = RankingMatrix(
        name=name,
        X=matrix.X[row_indices],
        y=matrix.y[row_indices],
        group_ptr=group_ptr,
        group_sizes=group_sizes,
        group_weights=instance_balanced_weights(group_instances),
        group_instances=group_instances,
        group_official_groups=matrix.group_official_groups[selected_groups],
        group_decision_ids=matrix.group_decision_ids[selected_groups],
        feature_names=matrix.feature_names,
        baseline_score_rank_pre=matrix.baseline_score_rank_pre[row_indices],
    )
    _validate_matrix(result)
    return result


def training_parameters(seed: int, nthread: int) -> dict[str, object]:
    return {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "tree_method": "hist",
        "lambdarank_pair_method": "mean",
        "lambdarank_num_pair_per_sample": 4,
        "eta": 0.05,
        "max_depth": 4,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "reg_alpha": 0.1,
        "seed": seed,
        "nthread": nthread,
    }


def fit_booster(
    train: RankingMatrix,
    validation: RankingMatrix,
    seed: int,
    nthread: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    verbose_eval: int | bool = 50,
) -> tuple[xgb.Booster, dict[str, object]]:
    parameters = training_parameters(seed, nthread)
    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        parameters,
        _dmatrix(train, nthread),
        num_boost_round=num_boost_round,
        evals=[(_dmatrix(validation, nthread), validation.name)],
        early_stopping_rounds=early_stopping_rounds,
        maximize=True,
        evals_result=evals_result,
        verbose_eval=verbose_eval,
    )
    best_iteration = int(booster.best_iteration)
    best_booster = booster[: best_iteration + 1]
    training = {
        "parameters": parameters,
        "best_iteration": best_iteration,
        "best_validation_ndcg10": float(booster.best_score),
        "rounds_evaluated": len(evals_result[validation.name]["ndcg@10"]),
    }
    return best_booster, training


def _stage_gate(evaluation: dict[str, dict[str, object]]) -> dict[str, object]:
    primary = evaluation[PRIMARY_VALIDATION_SUBSET]["comparison"]
    seen = evaluation["seen_family_val"]["comparison"]
    checks = {
        "primary_ndcg10_point_improves": primary["ndcg@10"]["delta"] > 0.0,
        "primary_ndcg10_ci95_excludes_zero": (
            primary["ndcg@10"]["ci95_lower"] > 0.0
        ),
        "primary_selection_overlap_point_improves": (
            primary["selection_overlap"]["delta"] > 0.0
        ),
        "seen_family_ndcg10_does_not_regress": seen["ndcg@10"]["delta"] >= 0.0,
        "seen_family_selection_overlap_does_not_regress": (
            seen["selection_overlap"]["delta"] >= 0.0
        ),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "consequence": (
            "eligible for paired multi-seed online validation"
            if passed
            else "do not start active SCIP intervention from this imitation model"
        ),
    }


def _feature_importance(
    booster: xgb.Booster, feature_names: np.ndarray
) -> list[dict[str, object]]:
    gain = booster.get_score(importance_type="gain")
    rows = []
    for index, name in enumerate(feature_names):
        value = float(gain.get(f"f{index}", 0.0))
        rows.append({"feature": str(name), "gain": value})
    return sorted(rows, key=lambda row: (-row["gain"], row["feature"]))


def train_imitation_model(
    dataset_dir: Path,
    dataset_manifest_path: Path,
    output_dir: Path,
    manifest_path: Path,
    seed: int = DEFAULT_SEED,
    nthread: int = 0,
    num_boost_round: int = 1200,
    early_stopping_rounds: int = 80,
    bootstrap_samples: int = 5000,
) -> dict[str, object]:
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if not all(dataset_manifest["checks"].values()):
        raise ValueError("Source ranking dataset has failed checks")
    train = load_effective_matrix(dataset_dir / f"{TRAIN_SUBSET}.npz", TRAIN_SUBSET)
    validation = {
        subset: load_effective_matrix(dataset_dir / f"{subset}.npz", subset)
        for subset in VALIDATION_SUBSETS
    }
    for subset, matrix in validation.items():
        if not np.array_equal(matrix.feature_names, train.feature_names):
            raise ValueError(f"Feature contract mismatch in {subset}")

    best_booster, fit = fit_booster(
        train,
        validation[PRIMARY_VALIDATION_SUBSET],
        seed=seed,
        nthread=nthread,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.ubj"
    best_booster.save_model(model_path)
    evaluation = {}
    anchored_evaluation = {}
    for subset, matrix in validation.items():
        scores = best_booster.predict(_dmatrix(matrix, nthread))
        evaluation[subset] = compare_with_baseline(
            matrix, scores, seed=seed, bootstrap_samples=bootstrap_samples
        )
        anchored_evaluation[subset] = compare_with_baseline(
            matrix,
            anchor_scip_top_candidate(matrix, scores),
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
    raw_stage_gate = _stage_gate(evaluation)
    anchored_stage_gate = _stage_gate(anchored_evaluation)
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "purpose": (
            "SCIP application imitation baseline for at most one root intervention per run; "
            "not a solve-benefit model"
        ),
        "dataset_manifest": _manifest_path(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "dataset_checks": dataset_manifest["checks"],
        "test_policy": "no test matrix was loaded or scored",
        "model": {
            "path": _manifest_path(model_path),
            "sha256": _sha256_file(model_path),
            "best_iteration": fit["best_iteration"],
            "best_primary_ndcg10": fit["best_validation_ndcg10"],
        },
        "training": {
            "rows": train.n_rows,
            "queries": train.n_groups,
            "instances": len(set(train.group_instances.tolist())),
            "features": len(train.feature_names),
            "parameters": fit["parameters"],
            "num_boost_round_limit": num_boost_round,
            "early_stopping_rounds": early_stopping_rounds,
            "rounds_evaluated": fit["rounds_evaluated"],
        },
        "evaluation": evaluation,
        "anchored_evaluation": anchored_evaluation,
        "policy_contract": (
            "preserve SCIP rank 1 and use the model only to rerank remaining candidates"
        ),
        "raw_stage_gate": raw_stage_gate,
        "stage_gate": anchored_stage_gate,
        "feature_importance_gain": _feature_importance(best_booster, train.feature_names),
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
        },
    }
    if not math.isfinite(manifest["model"]["best_primary_ndcg10"]):
        raise ValueError("Training did not produce a finite primary metric")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nthread", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = train_imitation_model(
        dataset_dir=args.dataset_dir.resolve(),
        dataset_manifest_path=args.dataset_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        manifest_path=args.manifest.resolve(),
        seed=args.seed,
        nthread=args.nthread,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(manifest["stage_gate"], sort_keys=True))
    for subset, result in manifest["anchored_evaluation"].items():
        ndcg = result["comparison"]["ndcg@10"]
        overlap = result["comparison"]["selection_overlap"]
        print(
            f"{subset} anchored: ndcg@10_delta={ndcg['delta']:.6f} "
            f"ci95=[{ndcg['ci95_lower']:.6f}, {ndcg['ci95_upper']:.6f}] "
            f"selection_overlap_delta={overlap['delta']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
