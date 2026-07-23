"""Run fixed-configuration, official-Group-disjoint ranking cross-validation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

from .observational import PROJECT_ROOT
from .ranking_train import (
    DEFAULT_DATASET_DIR,
    DEFAULT_DATASET_MANIFEST,
    DEFAULT_SEED,
    RankingMatrix,
    _bootstrap_delta,
    _dmatrix,
    _manifest_path,
    _sha256_file,
    anchor_scip_top_candidate,
    compare_with_baseline,
    fit_booster,
    load_effective_matrix,
    subset_matrix,
    training_parameters,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "ranking_imitation_xgb_group_cv_v1.json"
)
CROSSVAL_SCHEMA_VERSION = 1
DEFAULT_FOLDS = 5


def assign_official_group_folds(
    matrix: RankingMatrix, n_folds: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if n_folds < 2:
        raise ValueError("At least two folds are required")
    group_instances: dict[str, set[str]] = defaultdict(set)
    group_rows = defaultdict(int)
    for official_group, instance, size in zip(
        matrix.group_official_groups, matrix.group_instances, matrix.group_sizes
    ):
        name = str(official_group)
        if not name:
            continue
        group_instances[name].add(str(instance))
        group_rows[name] += int(size)
    if len(group_instances) < n_folds:
        raise ValueError("Fewer official Groups than requested folds")

    fold_groups: list[list[str]] = [[] for _ in range(n_folds)]
    fold_instances = [0] * n_folds
    fold_rows = [0] * n_folds
    ordered_groups = sorted(
        group_instances,
        key=lambda group: (-len(group_instances[group]), -group_rows[group], group),
    )
    for group in ordered_groups:
        fold = min(
            range(n_folds),
            key=lambda index: (fold_instances[index], fold_rows[index], index),
        )
        fold_groups[fold].append(group)
        fold_instances[fold] += len(group_instances[group])
        fold_rows[fold] += group_rows[group]

    group_to_fold = {
        group: fold for fold, groups in enumerate(fold_groups) for group in groups
    }
    assignments = np.asarray(
        [group_to_fold.get(str(group), -1) for group in matrix.group_official_groups],
        dtype=np.int16,
    )
    summaries = [
        {
            "fold": fold,
            "official_groups": sorted(groups),
            "official_group_count": len(groups),
            "instance_count": fold_instances[fold],
            "row_count": fold_rows[fold],
        }
        for fold, groups in enumerate(fold_groups)
    ]
    return assignments, summaries


def _aggregate_out_of_fold(
    per_instance: dict[str, dict[str, dict[str, float]]],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, object]:
    baseline = {
        instance: values["baseline"] for instance, values in per_instance.items()
    }
    model = {instance: values["model"] for instance, values in per_instance.items()}
    metric_names = list(next(iter(baseline.values())))
    baseline_aggregate = {
        metric: float(np.mean([values[metric] for values in baseline.values()]))
        for metric in metric_names
    }
    model_aggregate = {
        metric: float(np.mean([values[metric] for values in model.values()]))
        for metric in metric_names
    }
    comparison = {
        metric: _bootstrap_delta(
            baseline, model, metric, seed + index, bootstrap_samples
        )
        for index, metric in enumerate(metric_names)
    }
    ndcg10_deltas = np.asarray(
        [values["delta"]["ndcg@10"] for values in per_instance.values()]
    )
    return {
        "instances": len(per_instance),
        "aggregation": "one out-of-fold score per instance, mean across instances",
        "baseline": baseline_aggregate,
        "model": model_aggregate,
        "comparison": comparison,
        "ndcg10_instance_win_fraction": float(np.mean(ndcg10_deltas > 0.0)),
        "ndcg10_instance_loss_fraction": float(np.mean(ndcg10_deltas < 0.0)),
        "ndcg10_instance_tie_fraction": float(np.mean(ndcg10_deltas == 0.0)),
        "per_instance": dict(sorted(per_instance.items())),
    }


def run_group_cross_validation(
    dataset_dir: Path,
    dataset_manifest_path: Path,
    manifest_path: Path,
    n_folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
    nthread: int = 0,
    num_boost_round: int = 1200,
    early_stopping_rounds: int = 80,
    bootstrap_samples: int = 5000,
) -> dict[str, object]:
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if not all(dataset_manifest["checks"].values()):
        raise ValueError("Source ranking dataset has failed checks")
    source = load_effective_matrix(dataset_dir / "train.npz", "train")
    fold_ids, fold_summaries = assign_official_group_folds(source, n_folds)
    grouped = fold_ids >= 0
    excluded_instances = sorted(set(source.group_instances[~grouped].tolist()))
    fold_results = []
    out_of_fold = {}
    anchored_out_of_fold = {}
    for fold in range(n_folds):
        validation_mask = fold_ids == fold
        training_mask = grouped & ~validation_mask
        train = subset_matrix(source, training_mask, f"fold_{fold}_train")
        validation = subset_matrix(source, validation_mask, f"fold_{fold}_validation")
        booster, fit = fit_booster(
            train,
            validation,
            seed=seed + fold,
            nthread=nthread,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        scores = booster.predict(_dmatrix(validation, nthread))
        evaluation = compare_with_baseline(
            validation,
            scores,
            seed=seed + 1000 + fold,
            bootstrap_samples=bootstrap_samples,
        )
        anchored_evaluation = compare_with_baseline(
            validation,
            anchor_scip_top_candidate(validation, scores),
            seed=seed + 2000 + fold,
            bootstrap_samples=bootstrap_samples,
        )
        overlap = set(out_of_fold) & set(evaluation["per_instance"])
        if overlap:
            raise ValueError(f"Instances appeared in multiple folds: {sorted(overlap)}")
        out_of_fold.update(evaluation["per_instance"])
        anchored_out_of_fold.update(anchored_evaluation["per_instance"])
        fold_results.append(
            {
                **fold_summaries[fold],
                "training_instances": len(set(train.group_instances.tolist())),
                "training_queries": train.n_groups,
                "validation_queries": validation.n_groups,
                "fit": fit,
                "evaluation": {
                    key: value
                    for key, value in evaluation.items()
                    if key != "per_instance"
                },
                "anchored_evaluation": {
                    key: value
                    for key, value in anchored_evaluation.items()
                    if key != "per_instance"
                },
            }
        )

    aggregate = _aggregate_out_of_fold(out_of_fold, seed, bootstrap_samples)
    anchored_aggregate = _aggregate_out_of_fold(
        anchored_out_of_fold, seed + 3000, bootstrap_samples
    )
    ndcg10 = anchored_aggregate["comparison"]["ndcg@10"]
    overlap = anchored_aggregate["comparison"]["selection_overlap"]
    diagnostic_checks = {
        "ndcg10_point_improves": ndcg10["delta"] > 0.0,
        "ndcg10_ci95_excludes_zero": ndcg10["ci95_lower"] > 0.0,
        "selection_overlap_point_does_not_regress": overlap["delta"] >= 0.0,
        "more_instances_win_than_lose_ndcg10": (
            anchored_aggregate["ndcg10_instance_win_fraction"]
            > anchored_aggregate["ndcg10_instance_loss_fraction"]
        ),
    }
    manifest = {
        "schema_version": CROSSVAL_SCHEMA_VERSION,
        "purpose": (
            "fixed-configuration diagnostic of unseen official-Group imitation "
            "generalization; not an online solve-benefit evaluation"
        ),
        "dataset_manifest": _manifest_path(dataset_manifest_path),
        "dataset_manifest_sha256": _sha256_file(dataset_manifest_path),
        "dataset_checks": dataset_manifest["checks"],
        "test_policy": "only train.npz was loaded; validation and test matrices were untouched",
        "fold_contract": {
            "folds": n_folds,
            "assignment": (
                "greedy balance by official-Group instance count then candidate rows"
            ),
            "nonempty_official_groups": len(
                set(source.group_official_groups[grouped].tolist())
            ),
            "grouped_instances": len(set(source.group_instances[grouped].tolist())),
            "excluded_ungrouped_instances": len(excluded_instances),
            "excluded_ungrouped_instance_names": excluded_instances,
            "early_stopping_note": (
                "each held-out fold selects its own boosting round; this diagnostic is mildly "
                "optimistic and cannot override the external validation stage gate"
            ),
        },
        "parameters": training_parameters(seed, nthread),
        "folds": fold_results,
        "aggregate": aggregate,
        "anchored_aggregate": anchored_aggregate,
        "policy_contract": (
            "preserve SCIP rank 1 and use the model only to rerank remaining candidates"
        ),
        "diagnostic_gate": {
            "passed": all(diagnostic_checks.values()),
            "checks": diagnostic_checks,
            "consequence": (
                "supports a broad imitation signal but cannot override external validation"
                if all(diagnostic_checks.values())
                else "fixed imitation model lacks broad group-disjoint support"
            ),
        },
        "runtime": {
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
        },
    }
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
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nthread", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_group_cross_validation(
        dataset_dir=args.dataset_dir.resolve(),
        dataset_manifest_path=args.dataset_manifest.resolve(),
        manifest_path=args.manifest.resolve(),
        n_folds=args.folds,
        seed=args.seed,
        nthread=args.nthread,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        bootstrap_samples=args.bootstrap_samples,
    )
    ndcg = manifest["anchored_aggregate"]["comparison"]["ndcg@10"]
    overlap = manifest["anchored_aggregate"]["comparison"]["selection_overlap"]
    print(json.dumps(manifest["diagnostic_gate"], sort_keys=True))
    print(
        f"anchored_group_cv: instances={manifest['anchored_aggregate']['instances']} "
        f"ndcg@10_delta={ndcg['delta']:.6f} "
        f"ci95=[{ndcg['ci95_lower']:.6f}, {ndcg['ci95_upper']:.6f}] "
        f"selection_overlap_delta={overlap['delta']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
