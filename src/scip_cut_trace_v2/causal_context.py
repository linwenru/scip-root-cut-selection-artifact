"""Capture leakage-safe pre-intervention context for root cut actions."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable


CONTEXT_SCHEMA_VERSION = 1


def _safe(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except Exception:
        return None


def _number(value: Any) -> float | str | None:
    if value is None:
        return None
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number


def _integer(value: Any) -> int | None:
    return None if value is None else int(value)


def _boolean(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _origin_value(value: Any) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(getattr(value, "name", value))


def _coefficient_features(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "nnz": 0,
            "coeff_norm_l1": 0.0,
            "coeff_norm_l2": 0.0,
            "coeff_max_abs": 0.0,
            "coeff_min_abs": 0.0,
            "coeff_mean_abs": 0.0,
            "coeff_std_abs": 0.0,
        }
    absolute = [abs(value) for value in values]
    mean = sum(absolute) / len(absolute)
    return {
        "nnz": len(values),
        "coeff_norm_l1": sum(absolute),
        "coeff_norm_l2": math.sqrt(sum(value * value for value in values)),
        "coeff_max_abs": max(absolute),
        "coeff_min_abs": min(absolute),
        "coeff_mean_abs": mean,
        "coeff_std_abs": math.sqrt(
            sum((value - mean) ** 2 for value in absolute) / len(absolute)
        ),
    }


def extract_row_context(
    model: Any,
    row: Any,
    native_rank: int,
    native_selected: bool,
) -> dict[str, Any]:
    values = _safe(lambda: [float(value) for value in row.getVals()]) or []
    best_solution = _safe(model.getBestSol)
    context = {
        "cut_name": str(getattr(row, "name", "")),
        "native_rank": native_rank,
        "native_selected": native_selected,
        "rhs": _number(_safe(row.getRhs)),
        "lhs": _number(_safe(row.getLhs)),
        "constant": _number(_safe(row.getConstant)),
        "origin_type_raw": _origin_value(_safe(row.getOrigintype)),
        "efficacy": _number(_safe(lambda: model.getCutEfficacy(row))),
        "obj_parallelism": _number(
            _safe(lambda: model.getRowObjParallelism(row))
        ),
        "cutoff_distance": _number(
            _safe(
                lambda: model.getCutLPSolCutoffDistance(row, best_solution)
                if best_solution is not None
                else None
            )
        ),
        "n_int_cols": _integer(_safe(lambda: model.getRowNumIntCols(row))),
        "is_local": _boolean(_safe(row.isLocal)),
        "is_modifiable": _boolean(_safe(row.isModifiable)),
        "is_removable": _boolean(_safe(row.isRemovable)),
        "is_integral": _boolean(_safe(row.isIntegral)),
        "in_global_cutpool": _boolean(_safe(row.isInGlobalCutpool)),
    }
    context.update(_coefficient_features(values))
    return context


def capture_decision_context(
    model: Any,
    cuts: list[Any],
    forcedcuts: list[Any],
    nselectedcuts: int,
    run_number: int,
    selector_call: int,
) -> dict[str, Any]:
    node = _safe(model.getCurrentNode)
    solver_state = {
        "run_number": run_number,
        "selector_call": selector_call,
        "node_number": _integer(
            _safe(node.getNumber) if node is not None else None
        ),
        "node_depth": _integer(_safe(node.getDepth) if node is not None else None),
        "lp_status": str(_safe(model.getLPSolstat)),
        "objective_sense": str(_safe(model.getObjectiveSense)),
        "lp_objective": _number(_safe(model.getLPObjVal)),
        "lp_rows": _integer(_safe(model.getNLPRows)),
        "lp_cols": _integer(_safe(model.getNLPCols)),
        "lp_iterations_total": _integer(_safe(model.getNLPIterations)),
        "lp_iterations_node": _integer(_safe(model.getNNodeLPIterations)),
        "lp_count": _integer(_safe(model.getNLPs)),
        "separation_rounds_node": _integer(_safe(model.getNSepaRounds)),
        "dual_bound": _number(_safe(model.getDualbound)),
        "primal_bound": _number(_safe(model.getPrimalbound)),
        "gap": _number(_safe(model.getGap)),
        "processed_nodes": _integer(_safe(model.getNNodes)),
        "total_nodes": _integer(_safe(model.getNTotalNodes)),
        "cuts_applied": _integer(_safe(model.getNCutsApplied)),
        "candidate_cuts": len(cuts),
        "forced_cuts": len(forcedcuts),
        "native_selected_cuts": nselectedcuts,
    }
    candidates = [
        extract_row_context(model, row, rank, rank < nselectedcuts)
        for rank, row in enumerate(cuts)
    ]
    forced = [
        extract_row_context(model, row, rank, True)
        for rank, row in enumerate(forcedcuts)
    ]
    return {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "solver_state": solver_state,
        "candidates": candidates,
        "forced_candidates": forced,
    }


def context_sha256(context: dict[str, Any]) -> str:
    payload = json.dumps(
        context, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
