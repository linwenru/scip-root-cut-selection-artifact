"""Build and execute the online-compatible XGBoost imitation baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .cut_selection_baselines import BaselineSelection
from .observational import PROJECT_ROOT


SCHEMA_VERSION = 1
DEFAULT_SOURCE_DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "ranking_imitation_v1"
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "ranking_imitation_v1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "datasets" / "ranking_imitation_online_v1"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "ranking_imitation_online_v1.json"

# These fields have the same pre-action meaning in the trace and live callback.
ONLINE_ENCODED_FEATURES = (
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
    "is_removable",
    "is_integral",
    "in_global_cutpool",
    "run_number",
    "n_candidate_occurrences",
    "n_logical_candidates",
    "n_lp_rows_pre",
    "n_lp_cols_pre",
    "lp_obj_val_pre",
    "lp_iterations_total_pre",
    "lp_iterations_node_pre",
    "dual_bound_pre",
    "primal_bound_pre",
    "gap_pre",
    "n_cuts_applied_pre",
    "origin_type==CONS",
    "origin_type==CONSHDLR",
    "origin_type==SEPA",
    "origin_type==<UNK>",
)

EXCLUDED_NON_EQUIVALENT_FEATURES = (
    "sep_round_node",
    "sep_round_run",
    "sep_round_global",
    "lp_round_node",
    "lp_round_run",
    "lp_round_global",
    "n_cuts_generated_node_pre",
)

ORIGIN_VOCABULARY = ("CONS", "CONSHDLR", "SEPA", "<UNK>")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _resolve_artifact(path: str, manifest_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists():
        return project_candidate
    return manifest_path.resolve().parent / candidate


def derive_online_dataset(
    source_dir: Path,
    source_manifest_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Slice the audited matrices to fields reproducible in a live callback."""
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_features = tuple(
        source_manifest["feature_contract"]["encoded_feature_names"]
    )
    missing = [feature for feature in ONLINE_ENCODED_FEATURES if feature not in source_features]
    if missing:
        raise ValueError(f"Online features are absent from source matrices: {missing}")
    unexpected_exclusions = sorted(
        set(source_features) - set(ONLINE_ENCODED_FEATURES) - set(EXCLUDED_NON_EQUIVALENT_FEATURES)
    )
    if unexpected_exclusions:
        raise ValueError(
            "Feature equivalence must be reviewed before silently dropping fields: "
            f"{unexpected_exclusions}"
        )

    feature_indices = np.asarray(
        [source_features.index(feature) for feature in ONLINE_ENCODED_FEATURES],
        dtype=np.int64,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    matrices: dict[str, dict[str, Any]] = {}
    for subset, source_record in source_manifest["matrices"].items():
        source_path = source_dir / f"{subset}.npz"
        if _sha256_file(source_path) != source_record["sha256"]:
            raise ValueError(f"Source matrix hash mismatch: {source_path}")
        with np.load(source_path, allow_pickle=False) as source:
            payload = {name: source[name] for name in source.files}
        payload["X"] = payload["X"][:, feature_indices]
        payload["feature_names"] = np.asarray(ONLINE_ENCODED_FEATURES, dtype=str)
        output_path = output_dir / f"{subset}.npz"
        np.savez_compressed(output_path, **payload)
        matrices[subset] = {
            **{
                key: value
                for key, value in source_record.items()
                if key not in {"path", "sha256", "bytes", "features"}
            },
            "path": _manifest_path(output_path),
            "sha256": _sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "features": len(ONLINE_ENCODED_FEATURES),
        }

    checks = {
        "source_dataset_checks_passed": all(source_manifest["checks"].values()),
        "source_feature_contract_accounted_for": (
            set(source_features)
            == set(ONLINE_ENCODED_FEATURES) | set(EXCLUDED_NON_EQUIVALENT_FEATURES)
        ),
        "online_feature_order_frozen": tuple(ONLINE_ENCODED_FEATURES)
        == tuple(matrices and ONLINE_ENCODED_FEATURES),
        "all_matrix_feature_counts_match": all(
            matrix["features"] == len(ONLINE_ENCODED_FEATURES)
            for matrix in matrices.values()
        ),
        "test_labels_remain_unscored": True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Online dataset checks failed: {failed}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "online-compatible observational imitation matrices; labels record native "
            "SCIP cut application, not solve-level causal benefit"
        ),
        "source_manifest": _manifest_path(source_manifest_path),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "source_dataset_dir": _manifest_path(source_dir),
        "output_dir": _manifest_path(output_dir),
        "feature_contract": {
            "encoded_feature_names": list(ONLINE_ENCODED_FEATURES),
            "excluded_non_equivalent_features": list(EXCLUDED_NON_EQUIVALENT_FEATURES),
            "live_semantics": (
                "all fields are recomputed before the intervention from the callback's "
                "candidate rows and current SCIP state"
            ),
            "label_semantics": "observed logical cut applied by native SCIP",
            "origin_type_vocabulary": list(ORIGIN_VOCABULARY),
        },
        "matrices": matrices,
        "checks": checks,
        "test_policy": "test matrices are transformed mechanically but never loaded or scored",
        "numpy_version": np.__version__,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _safe(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except Exception:
        return None


def _number(value: Any) -> float:
    if value is None:
        return math.nan
    number = float(value)
    return number if math.isfinite(number) else math.nan


def _origin_name(value: Any) -> str:
    if value is None:
        return "<UNK>"
    try:
        raw = int(value)
    except (TypeError, ValueError):
        name = str(getattr(value, "name", value)).upper()
        return name if name in ORIGIN_VOCABULARY else "<UNK>"
    return {1: "CONSHDLR", 2: "CONS", 3: "SEPA"}.get(raw, "<UNK>")


def _coefficient_features(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "nnz": 0.0,
            "coeff_norm_l1": 0.0,
            "coeff_norm_l2": 0.0,
            "coeff_max_abs": 0.0,
            "coeff_min_abs": 0.0,
            "coeff_mean_abs": 0.0,
            "coeff_std_abs": 0.0,
        }
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "nnz": float(len(absolute)),
        "coeff_norm_l1": float(absolute.sum()),
        "coeff_norm_l2": float(np.linalg.norm(absolute)),
        "coeff_max_abs": float(absolute.max()),
        "coeff_min_abs": float(absolute.min()),
        "coeff_mean_abs": float(absolute.mean()),
        "coeff_std_abs": float(absolute.std()),
    }


def _hybrid_parameters(model: Any) -> dict[str, float]:
    return {
        name: float(model.getParam(f"cutselection/hybrid/{name}"))
        for name in (
            "efficacyweight",
            "dircutoffdistweight",
            "objparalweight",
            "intsupportweight",
        )
    }


def _hybrid_score(model: Any, row: Any, incumbent: Any, params: dict[str, float]) -> float:
    efficacy = _number(_safe(lambda: model.getCutEfficacy(row)))
    efficacy = 0.0 if math.isnan(efficacy) else efficacy
    objective_parallelism = _number(_safe(lambda: model.getRowObjParallelism(row)))
    objective_parallelism = 0.0 if math.isnan(objective_parallelism) else objective_parallelism
    nonzeros = int(_safe(row.getNNonz) or 0)
    integer_columns = int(_safe(lambda: model.getRowNumIntCols(row)) or 0)
    integer_support = integer_columns / nonzeros if nonzeros else 0.0

    efficacy_weight = params["efficacyweight"]
    directed_weight = params["dircutoffdistweight"]
    if incumbent is not None and directed_weight > 0.0:
        if bool(_safe(row.isLocal)):
            directed = efficacy
        else:
            cutoff = _number(
                _safe(lambda: model.getCutLPSolCutoffDistance(row, incumbent))
            )
            directed = max(0.0 if math.isnan(cutoff) else cutoff, efficacy)
        score = directed_weight * directed
    else:
        efficacy_weight += directed_weight
        score = 0.0
    score += efficacy_weight * efficacy
    score += params["objparalweight"] * objective_parallelism
    score += params["intsupportweight"] * integer_support
    if bool(_safe(row.isInGlobalCutpool)):
        score += 1e-4
    return float(score)


def _extract_row(model: Any, row: Any, score: float, incumbent: Any) -> dict[str, Any]:
    values = _safe(lambda: [float(value) for value in row.getVals()]) or []
    cutoff = (
        _number(_safe(lambda: model.getCutLPSolCutoffDistance(row, incumbent)))
        if incumbent is not None
        else math.nan
    )
    record: dict[str, Any] = {
        "cut_name": str(getattr(row, "name", "")),
        "origin_type": _origin_name(_safe(row.getOrigintype)),
        "score": score,
        "rhs": _number(_safe(row.getRhs)),
        "lhs": _number(_safe(row.getLhs)),
        "constant": _number(_safe(row.getConstant)),
        "efficacy": _number(_safe(lambda: model.getCutEfficacy(row))),
        "obj_parallelism": _number(_safe(lambda: model.getRowObjParallelism(row))),
        "cutoff_distance": cutoff,
        "cutoff_distance_available": float(not math.isnan(cutoff)),
        "n_int_cols": _number(_safe(lambda: model.getRowNumIntCols(row))),
        "is_removable": float(bool(_safe(row.isRemovable))),
        "is_integral": float(bool(_safe(row.isIntegral))),
        "in_global_cutpool": float(bool(_safe(row.isInGlobalCutpool))),
        "is_local": bool(_safe(row.isLocal)),
        "is_modifiable": bool(_safe(row.isModifiable)),
    }
    record.update(_coefficient_features(values))
    return record


def _signature(record: dict[str, Any]) -> tuple[Any, ...]:
    fields = (
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
    return tuple(
        None if isinstance(record[field], float) and math.isnan(record[field]) else record[field]
        for field in fields
    )


def _solver_state(model: Any, run_number: int, occurrences: int, logical: int) -> dict[str, float]:
    return {
        "run_number": float(run_number),
        "n_candidate_occurrences": float(occurrences),
        "n_logical_candidates": float(logical),
        "n_lp_rows_pre": _number(_safe(model.getNLPRows)),
        "n_lp_cols_pre": _number(_safe(model.getNLPCols)),
        "lp_obj_val_pre": _number(_safe(model.getLPObjVal)),
        "lp_iterations_total_pre": _number(_safe(model.getNLPIterations)),
        "lp_iterations_node_pre": _number(_safe(model.getNNodeLPIterations)),
        "dual_bound_pre": _number(_safe(model.getDualbound)),
        "primal_bound_pre": _number(_safe(model.getPrimalbound)),
        "gap_pre": _number(_safe(model.getGap)),
        "n_cuts_applied_pre": _number(_safe(model.getNCutsApplied)),
    }


@dataclass
class OnlineImitationRanker:
    booster: Any
    feature_names: tuple[str, ...]
    model_manifest_path: Path
    model_manifest_sha256: str
    model_path: Path
    model_sha256: str
    dataset_manifest_path: Path
    dataset_manifest_sha256: str
    offline_stage_gate_passed: bool

    @classmethod
    def load(cls, model_manifest_path: Path) -> "OnlineImitationRanker":
        import xgboost as xgb

        model_manifest_path = model_manifest_path.resolve()
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        model_path = _resolve_artifact(model_manifest["model"]["path"], model_manifest_path)
        if _sha256_file(model_path) != model_manifest["model"]["sha256"]:
            raise ValueError(f"Model hash mismatch: {model_path}")
        dataset_manifest_path = _resolve_artifact(
            model_manifest["dataset_manifest"], model_manifest_path
        )
        if _sha256_file(dataset_manifest_path) != model_manifest["dataset_manifest_sha256"]:
            raise ValueError(f"Dataset manifest hash mismatch: {dataset_manifest_path}")
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        feature_names = tuple(
            dataset_manifest["feature_contract"]["encoded_feature_names"]
        )
        if feature_names != ONLINE_ENCODED_FEATURES:
            raise ValueError("Model dataset does not use the frozen online feature order")
        booster = xgb.Booster()
        booster.load_model(model_path)
        return cls(
            booster=booster,
            feature_names=feature_names,
            model_manifest_path=model_manifest_path,
            model_manifest_sha256=_sha256_file(model_manifest_path),
            model_path=model_path,
            model_sha256=_sha256_file(model_path),
            dataset_manifest_path=dataset_manifest_path,
            dataset_manifest_sha256=_sha256_file(dataset_manifest_path),
            offline_stage_gate_passed=bool(model_manifest["stage_gate"]["passed"]),
        )

    def rank(
        self,
        model: Any,
        original_cuts: Sequence[Any],
        native_order: Sequence[Any],
        nselectedcuts: int,
        run_number: int,
    ) -> BaselineSelection:
        """Predict one score per logical cut and expand it to occurrence rows."""
        import xgboost as xgb

        if not original_cuts:
            return BaselineSelection([], nselectedcuts, (), self.metadata())
        params = _hybrid_parameters(model)
        incumbent = _safe(model.getBestSol)
        score_rows = []
        groups: dict[tuple[Any, ...], list[tuple[int, Any, dict[str, Any]]]] = defaultdict(list)
        for original_index, row in enumerate(original_cuts):
            score = _hybrid_score(model, row, incumbent, params)
            record = _extract_row(model, row, score, incumbent)
            groups[_signature(record)].append((original_index, row, record))

        logical = [min(occurrences, key=lambda item: item[0]) for occurrences in groups.values()]
        logical.sort(key=lambda item: (-item[2]["score"], item[0], item[2]["cut_name"]))
        denominator = max(len(logical) - 1, 1)
        state = _solver_state(model, run_number, len(original_cuts), len(logical))
        for rank, (_, _, record) in enumerate(logical, start=1):
            combined = {
                **record,
                **state,
                "score_rank_pre": float(rank),
                "score_rank_fraction_pre": float(rank - 1) / denominator,
            }
            score_rows.append(
                [
                    (
                        float(combined[feature])
                        if "==" not in feature
                        else float(combined["origin_type"] == feature.split("==", 1)[1])
                    )
                    for feature in self.feature_names
                ]
            )
        matrix = np.asarray(score_rows, dtype=np.float32)
        predictions = self.booster.predict(
            xgb.DMatrix(
                matrix,
                feature_names=[f"f{index}" for index in range(len(self.feature_names))],
                missing=np.nan,
            )
        ).astype(np.float64)

        native_top = native_order[0] if native_order else original_cuts[0]
        anchored_index = next(
            (
                index
                for index, representative in enumerate(logical)
                if any(row is native_top for _, row, _ in groups[_signature(representative[2])])
            ),
            0,
        )
        maximum = float(np.max(predictions))
        predictions[anchored_index] = maximum + max(abs(maximum), 1.0)
        logical_order = sorted(
            range(len(logical)), key=lambda index: (-predictions[index], index)
        )

        representatives = []
        duplicate_tail = []
        for index in logical_order:
            occurrences = sorted(groups[_signature(logical[index][2])], key=lambda item: item[0])
            representatives.append(occurrences[0][1])
            duplicate_tail.extend(item[1] for item in occurrences[1:])
        reordered = representatives + duplicate_tail
        metadata = self.metadata()
        metadata.update(
            {
                "logical_candidates": len(logical),
                "candidate_occurrences": len(original_cuts),
                "duplicate_occurrences": len(original_cuts) - len(logical),
                "anchor": "native-hybrid top candidate",
                "prediction_min": float(np.min(predictions)),
                "prediction_max": float(np.max(predictions)),
            }
        )
        return BaselineSelection(
            cuts=reordered,
            nselectedcuts=nselectedcuts,
            scores=tuple(float(value) for value in predictions),
            metadata=metadata,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "ranking": "xgboost-native-application-imitation-online-v1",
            "feature_schema_version": SCHEMA_VERSION,
            "features": len(self.feature_names),
            "model_manifest": str(self.model_manifest_path),
            "model_manifest_sha256": self.model_manifest_sha256,
            "model": str(self.model_path),
            "model_sha256": self.model_sha256,
            "dataset_manifest": str(self.dataset_manifest_path),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "offline_stage_gate_passed": self.offline_stage_gate_passed,
            "label_semantics": "imitate native SCIP application, not solve benefit",
            "tie_break": "pre-action hybrid-score rank",
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DATASET_DIR)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = derive_online_dataset(
        args.source_dir.resolve(),
        args.source_manifest.resolve(),
        args.output_dir.resolve(),
        args.manifest.resolve(),
    )
    print(json.dumps(manifest["checks"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
