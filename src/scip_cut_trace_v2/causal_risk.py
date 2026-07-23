"""Diagnose whether pre-intervention context predicts unsafe root actions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xgboost as xgb


SCHEMA_VERSION = 1
DEFAULT_SEED = 20260716
DEFAULT_BOOST_ROUNDS = 120
ACTIONS = ("boundary-swap", "boundary-swap-2", "efficacy-promote")
PAIR_NUMERIC_FIELDS = (
    "efficacy",
    "obj_parallelism",
    "cutoff_distance",
    "nnz",
    "n_int_cols",
    "coeff_norm_l2",
    "coeff_max_abs",
    "coeff_std_abs",
)
AGGREGATE_GROUPS = ("selected", "unselected")
DEFAULT_DATASET = Path("data/processed/causal_first_run_train_v1.jsonl.gz")
DEFAULT_MANIFEST = Path("data/manifests/causal_risk_grouped_oof_v1.json")


@dataclass(frozen=True)
class RiskMatrix:
    features: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]
    instances: np.ndarray
    seeds: np.ndarray
    actions: np.ndarray
    candidate_counts: np.ndarray
    selected_counts: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.labels.size)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _finite(values: Iterable[Any]) -> list[float]:
    return [number for value in values if math.isfinite(number := _as_float(value))]


def _mean(values: Iterable[Any]) -> float:
    numbers = _finite(values)
    return sum(numbers) / len(numbers) if numbers else math.nan


def _minimum(values: Iterable[Any]) -> float:
    numbers = _finite(values)
    return min(numbers) if numbers else math.nan


def _maximum(values: Iterable[Any]) -> float:
    numbers = _finite(values)
    return max(numbers) if numbers else math.nan


def _fraction(rows: list[dict[str, Any]], key: str) -> float:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return sum(bool(value) for value in values) / len(values) if values else math.nan


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    numerator_value = _as_float(numerator)
    denominator_value = _as_float(denominator)
    if not math.isfinite(numerator_value) or not denominator_value:
        return math.nan
    return numerator_value / denominator_value


def _efficacy_score(candidate: dict[str, Any]) -> float:
    value = _as_float(candidate.get("efficacy"))
    return value if math.isfinite(value) else -math.inf


def action_candidate_indices(
    action: str, candidates: list[dict[str, Any]], nselected: int
) -> tuple[int, int] | None:
    """Recover the exact selected/unselected pair used by a declared action."""
    if not 0 < nselected < len(candidates):
        return None
    if action == "boundary-swap":
        return nselected - 1, nselected
    if action == "boundary-swap-2":
        if nselected + 1 >= len(candidates):
            return None
        return nselected - 1, nselected + 1
    if action != "efficacy-promote":
        raise ValueError(f"unsupported action: {action}")
    removed = min(
        range(nselected),
        key=lambda index: (_efficacy_score(candidates[index]), -index),
    )
    added = max(
        range(nselected, len(candidates)),
        key=lambda index: (_efficacy_score(candidates[index]), -index),
    )
    if _efficacy_score(candidates[added]) <= _efficacy_score(candidates[removed]):
        return None
    return removed, added


def _append_feature(
    names: list[str], values: list[float], name: str, value: Any
) -> None:
    names.append(name)
    values.append(_as_float(value))


def context_action_features(
    context: dict[str, Any], action: str
) -> tuple[tuple[str, ...], np.ndarray] | None:
    """Return context-only features for one action; outcome labels are never read."""
    solver = context["solver_state"]
    candidates = context["candidates"]
    nselected = int(solver["native_selected_cuts"])
    pair = action_candidate_indices(action, candidates, nselected)
    if pair is None:
        return None

    names: list[str] = []
    values: list[float] = []
    for field in (
        "lp_rows",
        "lp_cols",
        "lp_iterations_total",
        "lp_iterations_node",
        "lp_count",
        "separation_rounds_node",
        "gap",
        "processed_nodes",
        "total_nodes",
        "cuts_applied",
        "candidate_cuts",
        "forced_cuts",
        "native_selected_cuts",
    ):
        _append_feature(names, values, f"solver_{field}", solver.get(field))
    _append_feature(
        names,
        values,
        "solver_selected_fraction",
        _safe_ratio(nselected, len(candidates)),
    )
    _append_feature(
        names,
        values,
        "solver_candidates_per_lp_row",
        _safe_ratio(len(candidates), solver.get("lp_rows")),
    )
    _append_feature(
        names,
        values,
        "solver_lp_rows_per_col",
        _safe_ratio(solver.get("lp_rows"), solver.get("lp_cols")),
    )
    _append_feature(
        names,
        values,
        "solver_lp_iterations_per_row",
        _safe_ratio(solver.get("lp_iterations_total"), solver.get("lp_rows")),
    )

    grouped_candidates = {
        "selected": candidates[:nselected],
        "unselected": candidates[nselected:],
    }
    for group in AGGREGATE_GROUPS:
        rows = grouped_candidates[group]
        for field in ("efficacy", "obj_parallelism", "nnz"):
            _append_feature(
                names, values, f"{group}_{field}_mean", _mean(row.get(field) for row in rows)
            )
            _append_feature(
                names, values, f"{group}_{field}_max", _maximum(row.get(field) for row in rows)
            )
        _append_feature(
            names,
            values,
            f"{group}_efficacy_min",
            _minimum(row.get("efficacy") for row in rows),
        )
        _append_feature(names, values, f"{group}_integral_fraction", _fraction(rows, "is_integral"))
        _append_feature(
            names,
            values,
            f"{group}_global_pool_fraction",
            _fraction(rows, "in_global_cutpool"),
        )

    removed_index, added_index = pair
    removed = candidates[removed_index]
    added = candidates[added_index]
    for field in PAIR_NUMERIC_FIELDS:
        removed_value = _as_float(removed.get(field))
        added_value = _as_float(added.get(field))
        _append_feature(names, values, f"removed_{field}", removed_value)
        _append_feature(names, values, f"added_{field}", added_value)
        _append_feature(
            names,
            values,
            f"delta_added_minus_removed_{field}",
            added_value - removed_value,
        )
    for field in ("is_integral", "is_local", "in_global_cutpool"):
        _append_feature(
            names,
            values,
            f"delta_added_minus_removed_{field}",
            int(bool(added.get(field))) - int(bool(removed.get(field))),
        )
    _append_feature(
        names, values, "removed_rank_fraction", _safe_ratio(removed_index, len(candidates))
    )
    _append_feature(
        names, values, "added_rank_fraction", _safe_ratio(added_index, len(candidates))
    )
    for declared_action in ACTIONS:
        _append_feature(
            names,
            values,
            f"action_{declared_action}",
            int(action == declared_action),
        )
    return tuple(names), np.asarray(values, dtype=np.float32)


def load_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def build_risk_matrix(records: Iterable[dict[str, Any]]) -> RiskMatrix:
    feature_rows = []
    labels = []
    instances = []
    seeds = []
    actions = []
    candidate_counts = []
    selected_counts = []
    feature_names: tuple[str, ...] | None = None
    for record in records:
        context = record["context"]
        for action in ACTIONS:
            label = record["action_labels"][action]
            extracted = context_action_features(context, action)
            if bool(label["eligible"]) != (extracted is not None):
                raise ValueError(
                    f"action eligibility disagrees with context for "
                    f"{record['instance_id']} seed {record['seed']} {action}"
                )
            if extracted is None:
                continue
            current_names, feature_row = extracted
            if feature_names is None:
                feature_names = current_names
            elif current_names != feature_names:
                raise ValueError("inconsistent risk feature schema")
            feature_rows.append(feature_row)
            labels.append(not bool(label["safe"]))
            instances.append(str(record["instance_id"]))
            seeds.append(int(record["seed"]))
            actions.append(action)
            candidate_counts.append(len(context["candidates"]))
            selected_counts.append(int(context["solver_state"]["native_selected_cuts"]))
    if not feature_rows or feature_names is None:
        raise ValueError("no eligible action rows found")
    return RiskMatrix(
        features=np.vstack(feature_rows),
        labels=np.asarray(labels, dtype=np.int8),
        feature_names=feature_names,
        instances=np.asarray(instances, dtype=str),
        seeds=np.asarray(seeds, dtype=np.int32),
        actions=np.asarray(actions, dtype=str),
        candidate_counts=np.asarray(candidate_counts, dtype=np.float64),
        selected_counts=np.asarray(selected_counts, dtype=np.float64),
    )


def leave_one_instance_out_masks(matrix: RiskMatrix) -> list[tuple[str, np.ndarray, np.ndarray]]:
    folds = []
    for instance in sorted(set(matrix.instances.tolist())):
        test = matrix.instances == instance
        train = ~test
        if set(matrix.instances[train]) & set(matrix.instances[test]):
            raise AssertionError("instance leakage across risk fold")
        folds.append((instance, train, test))
    return folds


def training_parameters(seed: int, nthread: int) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "eta": 0.05,
        "max_depth": 2,
        "min_child_weight": 2.0,
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "lambda": 8.0,
        "alpha": 1.0,
        "tree_method": "hist",
        "seed": seed,
        "nthread": nthread,
    }


def grouped_oof_scores(
    matrix: RiskMatrix,
    seed: int = DEFAULT_SEED,
    nthread: int = 1,
    boost_rounds: int = DEFAULT_BOOST_ROUNDS,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, float]]:
    scores = np.full(matrix.rows, np.nan, dtype=np.float64)
    folds = []
    importance: defaultdict[str, float] = defaultdict(float)
    for fold_index, (instance, train, test) in enumerate(
        leave_one_instance_out_masks(matrix)
    ):
        train_labels = matrix.labels[train]
        positives = int(train_labels.sum())
        negatives = int(train_labels.size - positives)
        if not positives or not negatives:
            raise ValueError(f"fold {instance} has only one training class")
        parameters = training_parameters(seed + fold_index, nthread)
        parameters["scale_pos_weight"] = negatives / positives
        train_matrix = xgb.DMatrix(
            matrix.features[train],
            label=train_labels,
            feature_names=list(matrix.feature_names),
            nthread=nthread,
        )
        test_matrix = xgb.DMatrix(
            matrix.features[test],
            feature_names=list(matrix.feature_names),
            nthread=nthread,
        )
        booster = xgb.train(parameters, train_matrix, num_boost_round=boost_rounds)
        scores[test] = booster.predict(test_matrix)
        for feature, gain in booster.get_score(importance_type="total_gain").items():
            importance[feature] += float(gain)
        folds.append(
            {
                "fold": fold_index,
                "held_out_instance": instance,
                "training_instances": len(set(matrix.instances[train].tolist())),
                "training_rows": int(train.sum()),
                "training_unsafe": positives,
                "test_rows": int(test.sum()),
                "test_unsafe": int(matrix.labels[test].sum()),
            }
        )
    if not np.isfinite(scores).all():
        raise ValueError("not all action rows received an out-of-fold risk score")
    return scores, folds, dict(importance)


def action_prior_oof_scores(matrix: RiskMatrix) -> np.ndarray:
    """Laplace-smoothed action risk rate using other instances only."""
    scores = np.empty(matrix.rows, dtype=np.float64)
    for _, train, test in leave_one_instance_out_masks(matrix):
        for action in ACTIONS:
            action_train = train & (matrix.actions == action)
            positives = int(matrix.labels[action_train].sum())
            count = int(action_train.sum())
            scores[test & (matrix.actions == action)] = (positives + 1.0) / (
                count + 2.0
            )
    return scores


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if not positives:
        raise ValueError("average precision requires an unsafe row")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    true_positives = 0
    seen = 0
    weighted_precision = 0.0
    index = 0
    while index < labels.size:
        end = index + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[index]:
            end += 1
        block_positives = int(sorted_labels[index:end].sum())
        seen = end
        true_positives += block_positives
        weighted_precision += (true_positives / seen) * block_positives
        index = end
    return weighted_precision / positives


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    if not len(positive_scores) or not len(negative_scores):
        raise ValueError("ROC AUC requires both classes")
    comparisons = [
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positive_scores
        for negative in negative_scores
    ]
    return sum(comparisons) / len(comparisons)


def _threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    abstain = scores >= threshold
    positives = labels == 1
    negatives = ~positives
    return {
        "threshold": float(threshold),
        "abstained_rows": int(abstain.sum()),
        "unsafe_recall": float(np.mean(abstain[positives])),
        "safe_abstention_rate": float(np.mean(abstain[negatives])),
    }


def risk_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be matching one-dimensional arrays")
    unsafe_scores = scores[labels == 1]
    if not len(unsafe_scores):
        raise ValueError("risk metrics require an unsafe row")
    top_fraction = {}
    descending = np.sort(scores)[::-1]
    for fraction in (0.05, 0.10, 0.20):
        count = max(1, math.ceil(len(scores) * fraction))
        top_fraction[f"top_{int(fraction * 100)}_percent"] = _threshold_metrics(
            labels, scores, descending[count - 1]
        )
    return {
        "prevalence": float(np.mean(labels)),
        "average_precision": _average_precision(labels, scores),
        "roc_auc": _roc_auc(labels, scores),
        "top_risk_fraction": top_fraction,
        "full_unsafe_recall": _threshold_metrics(
            labels, scores, float(np.min(unsafe_scores))
        ),
    }


def _importance_summary(importance: dict[str, float]) -> list[dict[str, Any]]:
    total = sum(importance.values())
    return [
        {
            "feature": feature,
            "total_gain": gain,
            "gain_fraction": gain / total if total else 0.0,
        }
        for feature, gain in sorted(
            importance.items(), key=lambda item: (-item[1], item[0])
        )[:20]
    ]


def run_diagnostic(
    dataset_path: Path,
    manifest_path: Path,
    seed: int = DEFAULT_SEED,
    nthread: int = 1,
    boost_rounds: int = DEFAULT_BOOST_ROUNDS,
) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    matrix = build_risk_matrix(load_records(dataset_path))
    model_scores, folds, importance = grouped_oof_scores(
        matrix, seed=seed, nthread=nthread, boost_rounds=boost_rounds
    )
    baseline_scores = {
        "constant_prevalence": np.full(matrix.rows, float(np.mean(matrix.labels))),
        "candidate_count": matrix.candidate_counts,
        "native_selected_count": matrix.selected_counts,
        "action_prior_other_instances": action_prior_oof_scores(matrix),
    }
    model_metrics = risk_metrics(matrix.labels, model_scores)
    baseline_metrics = {
        name: risk_metrics(matrix.labels, scores)
        for name, scores in baseline_scores.items()
    }
    best_baseline_ap = max(
        metrics["average_precision"] for metrics in baseline_metrics.values()
    )
    best_baseline_full_recall_cost = min(
        metrics["full_unsafe_recall"]["safe_abstention_rate"]
        for metrics in baseline_metrics.values()
    )
    top_twenty = model_metrics["top_risk_fraction"]["top_20_percent"]
    full_recall = model_metrics["full_unsafe_recall"]
    gate_checks = {
        "average_precision_at_least_twice_prevalence": (
            model_metrics["average_precision"] >= 2.0 * model_metrics["prevalence"]
        ),
        "roc_auc_at_least_0_75": model_metrics["roc_auc"] >= 0.75,
        "top_20_percent_recalls_all_unsafe": top_twenty["unsafe_recall"] == 1.0,
        "full_recall_abstains_at_most_20_percent_safe": (
            full_recall["safe_abstention_rate"] <= 0.20
        ),
        "average_precision_exceeds_all_structural_baselines": (
            model_metrics["average_precision"] > best_baseline_ap
        ),
        "full_recall_cost_beats_all_structural_baselines": (
            full_recall["safe_abstention_rate"] < best_baseline_full_recall_cost
        ),
    }
    unsafe_instances = sorted(set(matrix.instances[matrix.labels == 1].tolist()))
    predictions = [
        {
            "instance_id": str(matrix.instances[index]),
            "seed": int(matrix.seeds[index]),
            "action": str(matrix.actions[index]),
            "unsafe": bool(matrix.labels[index]),
            "model_risk": float(model_scores[index]),
            "candidate_count": int(matrix.candidate_counts[index]),
            "native_selected_count": int(matrix.selected_counts[index]),
            "action_prior_risk": float(
                baseline_scores["action_prior_other_instances"][index]
            ),
        }
        for index in range(matrix.rows)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "preliminary instance-grouped diagnosis of whether leakage-safe "
            "pre-intervention context predicts unsafe declared root actions"
        ),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "feature_contract": {
            "features": list(matrix.feature_names),
            "feature_count": len(matrix.feature_names),
            "excluded": [
                "instance identity",
                "seed",
                "solve time",
                "native final outcome",
                "treatment outcome",
                "post-action metrics",
            ],
            "missing_values": "represented as NaN and handled natively by XGBoost",
        },
        "evaluation_contract": {
            "split": "leave one complete instance out; all seeds and actions stay together",
            "thresholds": "diagnostic only; no deployment threshold is selected",
            "positive_label": "eligible treatment was unsafe relative to a completed native arm",
            "unit": "eligible pre-intervention context-action row",
        },
        "data": {
            "rows": matrix.rows,
            "instances": len(set(matrix.instances.tolist())),
            "unsafe_rows": int(matrix.labels.sum()),
            "safe_rows": int(matrix.rows - matrix.labels.sum()),
            "unsafe_instances": len(unsafe_instances),
            "unsafe_instance_names": unsafe_instances,
            "rows_by_action": {
                action: int(np.sum(matrix.actions == action)) for action in ACTIONS
            },
            "unsafe_by_action": {
                action: int(np.sum(matrix.labels[matrix.actions == action]))
                for action in ACTIONS
            },
        },
        "model": {
            "algorithm": "XGBoost binary classifier",
            "parameters": training_parameters(seed, nthread),
            "boost_rounds": boost_rounds,
            "out_of_fold_metrics": model_metrics,
            "top_feature_gain": _importance_summary(importance),
        },
        "baselines": baseline_metrics,
        "folds": folds,
        "diagnostic_gate": {
            "passed": all(gate_checks.values()),
            "checks": gate_checks,
            "consequence": (
                "supports testing the frozen risk protocol on an independent active cohort; not deployment"
                if all(gate_checks.values())
                else "current active contexts do not support a useful generalized risk gate"
            ),
        },
        "out_of_fold_predictions": predictions,
        "runtime": {"numpy": np.__version__, "xgboost": xgb.__version__},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nthread", type=int, default=1)
    parser.add_argument("--boost-rounds", type=int, default=DEFAULT_BOOST_ROUNDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_diagnostic(
        args.dataset, args.manifest, args.seed, args.nthread, args.boost_rounds
    )
    metrics = manifest["model"]["out_of_fold_metrics"]
    print(json.dumps(manifest["diagnostic_gate"], sort_keys=True))
    print(
        f"causal_risk_oof: rows={manifest['data']['rows']} "
        f"unsafe={manifest['data']['unsafe_rows']} "
        f"ap={metrics['average_precision']:.6f} "
        f"auc={metrics['roc_auc']:.6f} "
        f"full_recall_safe_abstention="
        f"{metrics['full_unsafe_recall']['safe_abstention_rate']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
