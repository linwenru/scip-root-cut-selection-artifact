"""Run paired, process-isolated SCIP cut-selection causal experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 3
LEGACY_TREATMENT_ARMS = ("boundary-swap", "boundary-swap-2", "efficacy-promote")
FIXED_BASELINE_ARMS = ("random-rank", "efficacy-rank", "adaptive-score")
LEARNED_POLICY_ARMS = ("xgb-imitation-rank",)
LEARNED_SHADOW_ARMS = ("xgb-imitation-shadow",)
LEARNED_TREATMENT_ARMS = (*LEARNED_POLICY_ARMS, *LEARNED_SHADOW_ARMS)
POLICY_RANKING_ARMS = (*FIXED_BASELINE_ARMS, *LEARNED_POLICY_ARMS)
TREATMENT_ARMS = (
    *LEGACY_TREATMENT_ARMS,
    *POLICY_RANKING_ARMS,
    *LEARNED_SHADOW_ARMS,
)
ARMS = ("native", "noop", "direct-hybrid", *TREATMENT_ARMS)
PARITY_CANDIDATE_ARMS = ("noop", "direct-hybrid")
INTERVENTION_SCOPES = ("per-run", "first-run-only")
PRIMARY_ORACLE_METRIC = "lp_iterations"
COMPLETE_STATUSES = frozenset(("optimal", "infeasible", "unbounded", "inforunbd"))
TREATMENT_CONTRACTS = {
    "boundary-swap": (
        "exchange the last native-selected cut with the first native-unselected cut"
    ),
    "boundary-swap-2": (
        "exchange the last native-selected cut with the second native-unselected cut"
    ),
    "efficacy-promote": (
        "exchange the least efficacious native-selected cut with the most efficacious "
        "native-unselected cut only when efficacy strictly improves"
    ),
    "random-rank": (
        "apply a deterministic hash permutation and keep the native-selected prefix size"
    ),
    "efficacy-rank": (
        "rank every candidate by descending SCIP cut efficacy and keep the "
        "native-selected prefix size"
    ),
    "adaptive-score": (
        "apply the independently ported Turner et al. normalized score and "
        "parallelism filter at the native-selected prefix size"
    ),
    "xgb-imitation-rank": (
        "apply the frozen online-compatible XGBoost native-application imitation "
        "ranker, preserve the native top candidate, and keep the native-selected "
        "prefix size"
    ),
    "xgb-imitation-shadow": (
        "execute the same native-hybrid, context, feature, and XGBoost ranking path "
        "as xgb-imitation-rank, but return the unchanged native selection"
    ),
}
SEED_PARAMETERS = (
    "randomization/randomseedshift",
    "randomization/permutationseed",
    "randomization/lpseed",
)
STRUCTURAL_INTEGER_FIELDS = (
    "nodes",
    "total_nodes",
    "lp_iterations",
    "lp_count",
    "cuts_applied",
)
STRUCTURAL_FLOAT_FIELDS = ("primal_bound", "dual_bound", "gap")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_text(value: float) -> float | str:
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number


def _version(model: Any) -> str:
    return ".".join(
        str(part)
        for part in (
            model.getMajorVersion(),
            model.getMinorVersion(),
            model.getTechVersion(),
        )
    )


def _set_param(model: Any, name: str, value: Any) -> None:
    if name not in model.getParams():
        raise ValueError(f"SCIP parameter is unavailable: {name}")
    model.setParam(name, value)


@dataclass
class CutSelectorStats:
    calls: int = 0
    root_calls: int = 0
    candidate_cuts: int = 0
    forced_cuts: int = 0
    selected_cuts: int = 0
    eligible_root_calls: int = 0
    ineligible_root_calls: int = 0
    interventions: int = 0
    shadow_evaluations: int = 0
    intervention_records: list[dict[str, Any]] = field(default_factory=list)
    shadow_records: list[dict[str, Any]] = field(default_factory=list)
    context_records: list[dict[str, Any]] = field(default_factory=list)
    callback_time_ns: int = 0
    native_hybrid_time_ns: int = 0
    context_capture_time_ns: int = 0
    policy_compute_time_ns: int = 0
    model_load_time_ns: int = 0


@dataclass
class RunInterventionState:
    run_number: int = 0
    decided_runs: set[int] = field(default_factory=set)
    intervened_runs: set[int] = field(default_factory=set)

    def mark_root_focused(self) -> None:
        self.run_number += 1

    def ensure_run(self) -> int:
        if self.run_number == 0:
            self.run_number = 1
        return self.run_number

    def can_intervene(self, scope: str = "per-run") -> bool:
        if scope not in INTERVENTION_SCOPES:
            raise ValueError(f"Unsupported intervention scope: {scope}")
        run_number = self.ensure_run()
        if scope == "first-run-only" and run_number != 1:
            return False
        return run_number not in self.decided_runs

    def mark_decided(self) -> None:
        self.decided_runs.add(self.ensure_run())

    def mark_intervened(self) -> None:
        run_number = self.ensure_run()
        self.decided_runs.add(run_number)
        self.intervened_runs.add(run_number)


def boundary_swap(
    cuts: list[Any], nselectedcuts: int, unselected_offset: int = 0
) -> tuple[list[Any], Any, Any] | None:
    """Exchange one selected and one unselected cut without changing the budget."""
    if unselected_offset < 0:
        raise ValueError("unselected_offset must be nonnegative")
    if not 0 < nselectedcuts or nselectedcuts + unselected_offset >= len(cuts):
        return None
    swapped = list(cuts)
    selected_index = nselectedcuts - 1
    unselected_index = nselectedcuts + unselected_offset
    removed = swapped[selected_index]
    added = swapped[unselected_index]
    swapped[selected_index], swapped[unselected_index] = added, removed
    return swapped, removed, added


def efficacy_promote(
    cuts: list[Any], nselectedcuts: int, efficacies: list[float]
) -> tuple[list[Any], Any, Any] | None:
    """Promote a strictly more efficacious unselected cut at fixed budget."""
    if len(efficacies) != len(cuts):
        raise ValueError("one efficacy value is required per cut")
    if not 0 < nselectedcuts < len(cuts):
        return None

    def score(value: float) -> float:
        value = float(value)
        return -math.inf if math.isnan(value) else value

    selected_index = min(
        range(nselectedcuts), key=lambda index: (score(efficacies[index]), -index)
    )
    unselected_index = max(
        range(nselectedcuts, len(cuts)),
        key=lambda index: (score(efficacies[index]), -index),
    )
    if score(efficacies[unselected_index]) <= score(efficacies[selected_index]):
        return None

    promoted = list(cuts)
    removed = promoted[selected_index]
    added = promoted[unselected_index]
    promoted[selected_index], promoted[unselected_index] = added, removed
    return promoted, removed, added


def _row_name(row: Any) -> str:
    return str(getattr(row, "name", ""))


def _row_identity_index(rows: list[Any], target: Any) -> int:
    return next(index for index, row in enumerate(rows) if row is target)


def _selected_set_changes(
    native_order: list[Any], treatment_order: list[Any], nselectedcuts: int
) -> tuple[list[Any], list[Any]]:
    """Return native-selected removals and treatment-selected additions by identity."""
    native_selected = native_order[:nselectedcuts]
    treatment_selected = treatment_order[:nselectedcuts]
    native_ids = {id(row) for row in native_selected}
    treatment_ids = {id(row) for row in treatment_selected}
    removed = [row for row in native_selected if id(row) not in treatment_ids]
    added = [row for row in treatment_selected if id(row) not in native_ids]
    return removed, added


def _build_noop_cut_selector() -> tuple[Any, CutSelectorStats]:
    from pyscipopt import SCIP_RESULT
    from pyscipopt.scip import Cutsel

    stats = CutSelectorStats()

    class DelegatingNoopCutSelector(Cutsel):
        """Observe the callback and ask SCIP to try the next selector."""

        def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
            started_ns = time.perf_counter_ns()
            try:
                return self._select(cuts, forcedcuts, root, maxnselectedcuts)
            finally:
                stats.callback_time_ns += time.perf_counter_ns() - started_ns

        def _select(self, cuts, forcedcuts, root, maxnselectedcuts):
            del maxnselectedcuts
            stats.calls += 1
            stats.root_calls += int(bool(root))
            stats.candidate_cuts += len(cuts)
            stats.forced_cuts += len(forcedcuts)
            return {"nselectedcuts": 0, "result": SCIP_RESULT.DIDNOTFIND}

    return DelegatingNoopCutSelector(), stats


def _build_direct_hybrid_cut_selector() -> tuple[Any, CutSelectorStats]:
    from pyscipopt.scip import Cutsel

    from .native_hybrid import select_cuts_hybrid

    stats = CutSelectorStats()

    class DirectHybridCutSelector(Cutsel):
        """Call SCIP's included hybrid callback directly and return its decision."""

        def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
            started_ns = time.perf_counter_ns()
            try:
                return self._select(cuts, forcedcuts, root, maxnselectedcuts)
            finally:
                stats.callback_time_ns += time.perf_counter_ns() - started_ns

        def _select(self, cuts, forcedcuts, root, maxnselectedcuts):
            stats.calls += 1
            stats.root_calls += int(bool(root))
            stats.candidate_cuts += len(cuts)
            stats.forced_cuts += len(forcedcuts)
            hybrid_started_ns = time.perf_counter_ns()
            try:
                selection = select_cuts_hybrid(
                    self.model, cuts, forcedcuts, root, maxnselectedcuts
                )
            finally:
                stats.native_hybrid_time_ns += (
                    time.perf_counter_ns() - hybrid_started_ns
                )
            stats.selected_cuts += selection.nselectedcuts
            return {
                "cuts": selection.cuts,
                "nselectedcuts": selection.nselectedcuts,
                "result": selection.result,
            }

    return DirectHybridCutSelector(), stats


def _build_root_treatment_cut_selector(
    state: RunInterventionState,
    treatment_arm: str,
    intervention_scope: str,
    experiment_seed: int = 0,
    learned_model_manifest: Path | None = None,
) -> tuple[Any, CutSelectorStats]:
    from pyscipopt import SCIP_RESULT
    from pyscipopt.scip import Cutsel

    from .causal_context import capture_decision_context, context_sha256
    from .cut_selection_baselines import (
        adaptive_score_rank,
        deterministic_random_rank,
        efficacy_rank,
    )
    from .native_hybrid import select_cuts_hybrid
    from .online_imitation import OnlineImitationRanker

    if treatment_arm not in TREATMENT_ARMS:
        raise ValueError(f"Unsupported treatment arm: {treatment_arm}")
    if intervention_scope not in INTERVENTION_SCOPES:
        raise ValueError(f"Unsupported intervention scope: {intervention_scope}")
    if treatment_arm in LEARNED_TREATMENT_ARMS and learned_model_manifest is None:
        raise ValueError(f"{treatment_arm} requires a learned model manifest")
    stats = CutSelectorStats()
    learned_ranker = None
    if treatment_arm in LEARNED_TREATMENT_ARMS:
        model_load_started_ns = time.perf_counter_ns()
        try:
            learned_ranker = OnlineImitationRanker.load(learned_model_manifest)
        finally:
            stats.model_load_time_ns += (
                time.perf_counter_ns() - model_load_started_ns
            )

    class RootTreatmentCutSelector(Cutsel):
        """Apply one native-budget-preserving root action per SCIP run."""

        def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
            started_ns = time.perf_counter_ns()
            try:
                return self._select(cuts, forcedcuts, root, maxnselectedcuts)
            finally:
                stats.callback_time_ns += time.perf_counter_ns() - started_ns

        def _select(self, cuts, forcedcuts, root, maxnselectedcuts):
            stats.calls += 1
            stats.root_calls += int(bool(root))
            stats.candidate_cuts += len(cuts)
            stats.forced_cuts += len(forcedcuts)
            if not root or not state.can_intervene(intervention_scope):
                return {"nselectedcuts": 0, "result": SCIP_RESULT.DIDNOTFIND}

            hybrid_started_ns = time.perf_counter_ns()
            try:
                selection = select_cuts_hybrid(
                    self.model, cuts, forcedcuts, root, maxnselectedcuts
                )
            finally:
                stats.native_hybrid_time_ns += (
                    time.perf_counter_ns() - hybrid_started_ns
                )
            stats.selected_cuts += selection.nselectedcuts
            if not 0 < selection.nselectedcuts < len(selection.cuts):
                stats.ineligible_root_calls += 1
                return {
                    "cuts": selection.cuts,
                    "nselectedcuts": selection.nselectedcuts,
                    "result": selection.result,
                }

            run_number = state.ensure_run()
            context_started_ns = time.perf_counter_ns()
            try:
                context = capture_decision_context(
                    self.model,
                    selection.cuts,
                    forcedcuts,
                    selection.nselectedcuts,
                    run_number,
                    stats.calls,
                )
                context_digest = context_sha256(context)
            finally:
                stats.context_capture_time_ns += (
                    time.perf_counter_ns() - context_started_ns
                )
            context_record = {
                "run": run_number,
                "selector_call": stats.calls,
                "context_sha256": context_digest,
                "requested_action": treatment_arm,
                "decision_context": context,
            }
            stats.context_records.append(context_record)
            state.mark_decided()
            policy_started_ns = time.perf_counter_ns()
            efficacies = None
            action_metadata: dict[str, Any] = {}
            if treatment_arm == "boundary-swap":
                action_result = boundary_swap(selection.cuts, selection.nselectedcuts)
            elif treatment_arm == "boundary-swap-2":
                action_result = boundary_swap(
                    selection.cuts, selection.nselectedcuts, unselected_offset=1
                )
            elif treatment_arm == "efficacy-promote":
                efficacies = [
                    float(self.model.getCutEfficacy(row)) for row in selection.cuts
                ]
                action_result = efficacy_promote(
                    selection.cuts, selection.nselectedcuts, efficacies
                )
            else:
                if treatment_arm == "random-rank":
                    baseline = deterministic_random_rank(
                        selection.cuts,
                        selection.nselectedcuts,
                        key=(
                            f"seed={experiment_seed};run={run_number};"
                            f"context={context_digest}"
                        ),
                    )
                elif treatment_arm == "efficacy-rank":
                    baseline = efficacy_rank(
                        self.model, selection.cuts, selection.nselectedcuts
                    )
                    efficacies = list(baseline.scores or ())
                elif treatment_arm == "adaptive-score":
                    baseline = adaptive_score_rank(
                        self.model,
                        selection.cuts,
                        forcedcuts,
                        selection.nselectedcuts,
                        root=bool(root),
                    )
                else:
                    if learned_ranker is None:
                        raise RuntimeError("learned ranker was not initialized")
                    baseline = learned_ranker.rank(
                        self.model,
                        cuts,
                        selection.cuts,
                        selection.nselectedcuts,
                        run_number,
                    )
                removed_rows, added_rows = _selected_set_changes(
                    selection.cuts, baseline.cuts, selection.nselectedcuts
                )
                action_result = (
                    (baseline.cuts, removed_rows, added_rows)
                    if removed_rows or added_rows
                    else None
                )
                action_metadata = baseline.metadata

            stats.policy_compute_time_ns += time.perf_counter_ns() - policy_started_ns
            if treatment_arm in LEARNED_SHADOW_ARMS:
                proposed_removed, proposed_added = _selected_set_changes(
                    selection.cuts, baseline.cuts, selection.nselectedcuts
                )
                proposed_removed_indices = [
                    _row_identity_index(selection.cuts, row)
                    for row in proposed_removed
                ]
                proposed_added_indices = [
                    _row_identity_index(selection.cuts, row) for row in proposed_added
                ]
                context_record.update(
                    {
                        "shadow_only": True,
                        "selected_set_changed": False,
                        "proposed_selected_set_changed": bool(
                            proposed_removed or proposed_added
                        ),
                        "proposed_changed_selected_cuts": len(proposed_removed),
                        "action_metadata": action_metadata,
                    }
                )
                stats.eligible_root_calls += 1
                stats.shadow_evaluations += 1
                stats.shadow_records.append(
                    {
                        "action": treatment_arm,
                        "run": run_number,
                        "selector_call": stats.calls,
                        "context_sha256": context_digest,
                        "candidate_cuts": len(cuts),
                        "forced_cuts": len(forcedcuts),
                        "selected_cuts": selection.nselectedcuts,
                        "proposed_changed_selected_cuts": len(proposed_removed),
                        "proposed_removed_native_indices": (
                            proposed_removed_indices
                        ),
                        "proposed_added_native_indices": proposed_added_indices,
                        "proposed_removed_cut_names": [
                            _row_name(row) for row in proposed_removed
                        ],
                        "proposed_added_cut_names": [
                            _row_name(row) for row in proposed_added
                        ],
                        "action_metadata": action_metadata,
                    }
                )
                return {
                    "cuts": selection.cuts,
                    "nselectedcuts": selection.nselectedcuts,
                    "result": selection.result,
                }

            if action_result is None:
                context_record.update(
                    {
                        "selected_set_changed": False,
                        "changed_selected_cuts": 0,
                        "action_metadata": action_metadata,
                    }
                )
                stats.ineligible_root_calls += 1
                return {
                    "cuts": selection.cuts,
                    "nselectedcuts": selection.nselectedcuts,
                    "result": selection.result,
                }

            reordered, removed_value, added_value = action_result
            removed_rows = (
                removed_value if isinstance(removed_value, list) else [removed_value]
            )
            added_rows = added_value if isinstance(added_value, list) else [added_value]
            removed_indices = [
                _row_identity_index(selection.cuts, row) for row in removed_rows
            ]
            added_indices = [
                _row_identity_index(selection.cuts, row) for row in added_rows
            ]
            context_record.update(
                {
                    "selected_set_changed": True,
                    "changed_selected_cuts": len(removed_rows),
                    "action_metadata": action_metadata,
                }
            )
            state.mark_intervened()
            stats.eligible_root_calls += 1
            stats.interventions += 1
            stats.intervention_records.append(
                {
                    "action": treatment_arm,
                    "run": run_number,
                    "selector_call": stats.calls,
                    "context_sha256": context_digest,
                    "candidate_cuts": len(cuts),
                    "forced_cuts": len(forcedcuts),
                    "selected_cuts": selection.nselectedcuts,
                    "changed_selected_cuts": len(removed_rows),
                    "removed_native_indices": removed_indices,
                    "added_native_indices": added_indices,
                    "removed_cut_names": [_row_name(row) for row in removed_rows],
                    "added_cut_names": [_row_name(row) for row in added_rows],
                    "removed_native_index": (
                        removed_indices[0] if len(removed_indices) == 1 else None
                    ),
                    "added_native_index": (
                        added_indices[0] if len(added_indices) == 1 else None
                    ),
                    "removed_cut_name": (
                        _row_name(removed_rows[0]) if len(removed_rows) == 1 else None
                    ),
                    "added_cut_name": (
                        _row_name(added_rows[0]) if len(added_rows) == 1 else None
                    ),
                    "removed_efficacy": (
                        _finite_or_text(efficacies[removed_indices[0]])
                        if efficacies is not None and len(removed_indices) == 1
                        else None
                    ),
                    "added_efficacy": (
                        _finite_or_text(efficacies[added_indices[0]])
                        if efficacies is not None and len(added_indices) == 1
                        else None
                    ),
                    "action_metadata": action_metadata,
                }
            )
            return {
                "cuts": reordered,
                "nselectedcuts": selection.nselectedcuts,
                "result": selection.result,
            }

    return RootTreatmentCutSelector(), stats


def _build_run_event_handler(state: RunInterventionState) -> Any:
    from pyscipopt import Eventhdlr, SCIP_EVENTTYPE

    class RunEventHandler(Eventhdlr):
        def eventinitsol(self):
            self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)

        def eventexitsol(self):
            self.model.dropEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)

        def eventexec(self, event):
            node = event.getNode()
            if node is not None and int(node.getDepth()) == 0:
                state.mark_root_focused()
            return {}

    return RunEventHandler()


def solve_arm(
    instance: Path,
    arm: str,
    seed: int,
    time_limit: float,
    node_limit: int | None,
    intervention_scope: str = "per-run",
    learned_model_manifest: Path | None = None,
) -> dict[str, Any]:
    import pyscipopt
    from pyscipopt import Model

    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if intervention_scope not in INTERVENTION_SCOPES:
        raise ValueError(f"Unsupported intervention scope: {intervention_scope}")
    if arm in LEARNED_TREATMENT_ARMS and learned_model_manifest is None:
        raise ValueError(f"{arm} requires a learned model manifest")
    if seed < 0:
        raise ValueError("SCIP seeds must be nonnegative")
    instance = instance.resolve()
    if not instance.is_file():
        raise FileNotFoundError(instance)

    arm_started_ns = time.perf_counter_ns()
    model = Model()
    model.hideOutput()
    parameter_values: dict[str, int | float] = {}
    for name in SEED_PARAMETERS:
        _set_param(model, name, seed)
        parameter_values[name] = seed
    for name, value in (("parallel/maxnthreads", 1), ("lp/threads", 1)):
        _set_param(model, name, value)
        parameter_values[name] = value
    _set_param(model, "limits/time", time_limit)
    parameter_values["limits/time"] = time_limit
    if node_limit is not None:
        _set_param(model, "limits/nodes", node_limit)
        parameter_values["limits/nodes"] = node_limit

    selector_stats = CutSelectorStats()
    run_state = RunInterventionState()
    if arm in PARITY_CANDIDATE_ARMS:
        selector, selector_stats = (
            _build_noop_cut_selector()
            if arm == "noop"
            else _build_direct_hybrid_cut_selector()
        )
        model.includeCutsel(
            selector,
            f"v2_{arm.replace('-', '_')}",
            f"V2 structural parity probe for {arm}",
            priority=100_000_000,
        )
    elif arm in TREATMENT_ARMS:
        selector, selector_stats = _build_root_treatment_cut_selector(
            run_state, arm, intervention_scope, seed, learned_model_manifest
        )
        model.includeCutsel(
            selector,
            f"v2_{arm.replace('-', '_')}",
            f"One native-budget-preserving root {arm} action per SCIP run",
            priority=100_000_000,
        )
        model.includeEventhdlr(
            _build_run_event_handler(run_state),
            "v2_run_tracker",
            "Track SCIP root focus events and restart runs",
        )

    model.readProblem(str(instance))
    model.optimize()
    arm_wall_time_seconds = (time.perf_counter_ns() - arm_started_ns) / 1e9
    selector_timing_ns = {
        "model_load": selector_stats.model_load_time_ns,
        "callback_total": selector_stats.callback_time_ns,
        "native_hybrid": selector_stats.native_hybrid_time_ns,
        "context_capture": selector_stats.context_capture_time_ns,
        "policy_compute": selector_stats.policy_compute_time_ns,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "arm": arm,
        "seed": seed,
        "instance": str(instance),
        "instance_sha256": _sha256_file(instance),
        "runtime": {
            "python": sys.version.split()[0],
            "pyscipopt": pyscipopt.__version__,
            "scip": _version(model),
        },
        "parameters": parameter_values,
        "outcome": {
            "status": str(model.getStatus()),
            "objective_sense": str(model.getObjectiveSense()),
            "primal_bound": _finite_or_text(model.getPrimalbound()),
            "dual_bound": _finite_or_text(model.getDualbound()),
            "gap": _finite_or_text(model.getGap()),
            "nodes": int(model.getNNodes()),
            "total_nodes": int(model.getNTotalNodes()),
            "lp_iterations": int(model.getNLPIterations()),
            "lp_count": int(model.getNLPs()),
            "cuts_applied": int(model.getNCutsApplied()),
            "solving_time": float(model.getSolvingTime()),
            "arm_wall_time_seconds": arm_wall_time_seconds,
            "primal_dual_integral": float(model.getPrimalDualIntegral()),
        },
        "selector": {
            "mode": arm,
            "intervention_scope": intervention_scope,
            "calls": selector_stats.calls,
            "root_calls": selector_stats.root_calls,
            "candidate_cuts": selector_stats.candidate_cuts,
            "forced_cuts": selector_stats.forced_cuts,
            "selected_cuts": selector_stats.selected_cuts,
            "run_count": run_state.run_number,
            "decisions": len(run_state.decided_runs),
            "eligible_root_calls": selector_stats.eligible_root_calls,
            "ineligible_root_calls": selector_stats.ineligible_root_calls,
            "interventions": selector_stats.interventions,
            "shadow_evaluations": selector_stats.shadow_evaluations,
            "intervention_records": selector_stats.intervention_records,
            "shadow_records": selector_stats.shadow_records,
            "context_records": selector_stats.context_records,
            "timing_ns": selector_timing_ns,
            "timing_seconds": {
                name: value / 1e9 for name, value in selector_timing_ns.items()
            },
            "learned_model_manifest": (
                str(learned_model_manifest.resolve())
                if learned_model_manifest is not None
                else None
            ),
            "learned_model_manifest_sha256": (
                _sha256_file(learned_model_manifest.resolve())
                if learned_model_manifest is not None
                else None
            ),
        },
    }
    model.freeProb()
    return result


def _numbers_close(left: float | str, right: float | str) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def compare_pair(
    native: dict[str, Any], candidate: dict[str, Any], candidate_arm: str = "noop"
) -> dict[str, Any]:
    if candidate_arm not in PARITY_CANDIDATE_ARMS:
        raise ValueError(f"Unsupported parity candidate arm: {candidate_arm}")
    native_outcome = native["outcome"]
    candidate_outcome = candidate["outcome"]
    checks: dict[str, bool] = {
        "arm_order": (
            native["arm"] == "native" and candidate["arm"] == candidate_arm
        ),
        "same_instance_sha256": (
            native["instance_sha256"] == candidate["instance_sha256"]
        ),
        "same_seed": native["seed"] == candidate["seed"],
        "same_parameters": native["parameters"] == candidate["parameters"],
        "same_runtime_versions": native["runtime"] == candidate["runtime"],
        "same_status": native_outcome["status"] == candidate_outcome["status"],
        "same_objective_sense": (
            native_outcome["objective_sense"]
            == candidate_outcome["objective_sense"]
        ),
        "candidate_callback_exercised": candidate["selector"]["calls"] > 0,
    }
    checks.update(
        {
            f"same_{field}": native_outcome[field] == candidate_outcome[field]
            for field in STRUCTURAL_INTEGER_FIELDS
        }
    )
    checks.update(
        {
            f"same_{field}": _numbers_close(
                native_outcome[field], candidate_outcome[field]
            )
            for field in STRUCTURAL_FLOAT_FIELDS
        }
    )
    native_time = float(native_outcome["solving_time"])
    candidate_time = float(candidate_outcome["solving_time"])
    native_integral = float(native_outcome["primal_dual_integral"])
    candidate_integral = float(candidate_outcome["primal_dual_integral"])
    return {
        "candidate_arm": candidate_arm,
        "passed": all(checks.values()),
        "checks": checks,
        "non_gating_measurements": {
            "solving_time_native": native_time,
            "solving_time_candidate": candidate_time,
            "solving_time_ratio_candidate_over_native": (
                candidate_time / native_time if native_time > 0.0 else None
            ),
            "primal_dual_integral_native": native_integral,
            "primal_dual_integral_candidate": candidate_integral,
            "primal_dual_integral_delta_candidate_minus_native": (
                candidate_integral - native_integral
            ),
        },
    }


def classify_parity_pair(
    native: dict[str, Any], candidate: dict[str, Any], comparison: dict[str, Any]
) -> str:
    """Classify parity evidence without changing the pre-registered all-pair gate."""
    if comparison["passed"]:
        return "passed"
    if not comparison["checks"]["candidate_callback_exercised"]:
        return "callback_not_exercised"
    native_status = native["outcome"]["status"]
    candidate_status = candidate["outcome"]["status"]
    if native_status not in COMPLETE_STATUSES and candidate_status == native_status:
        return "both_arms_same_incomplete_limit_status"
    return "structural_mismatch"


def compare_causal_pair(
    native: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, Any]:
    treatment_arm = treatment["arm"]
    if treatment_arm not in TREATMENT_ARMS:
        raise ValueError(f"Unsupported treatment arm: {treatment_arm}")
    native_outcome = native["outcome"]
    treatment_outcome = treatment["outcome"]
    selector = treatment["selector"]
    intervention_scope = selector.get("intervention_scope", "per-run")
    context_records = selector.get("context_records", [])
    safety_checks = {
        "arm_order": (
            native["arm"] == "native" and treatment["arm"] == treatment_arm
        ),
        "same_instance_sha256": (
            native["instance_sha256"] == treatment["instance_sha256"]
        ),
        "same_seed": native["seed"] == treatment["seed"],
        "same_parameters": native["parameters"] == treatment["parameters"],
        "same_runtime_versions": native["runtime"] == treatment["runtime"],
        "native_complete": native_outcome["status"] in COMPLETE_STATUSES,
        "treatment_complete": treatment_outcome["status"] in COMPLETE_STATUSES,
        "same_status": native_outcome["status"] == treatment_outcome["status"],
        "same_objective_sense": (
            native_outcome["objective_sense"]
            == treatment_outcome["objective_sense"]
        ),
        "same_primal_bound": _numbers_close(
            native_outcome["primal_bound"], treatment_outcome["primal_bound"]
        ),
        "same_dual_bound": _numbers_close(
            native_outcome["dual_bound"], treatment_outcome["dual_bound"]
        ),
        "known_intervention_scope": intervention_scope in INTERVENTION_SCOPES,
        "run_budget_respected": (
            0
            <= selector["interventions"]
            <= (
                1
                if intervention_scope == "first-run-only"
                else selector["run_count"]
            )
        ),
        "one_record_per_intervention": (
            len(selector["intervention_records"]) == selector["interventions"]
        ),
        "context_budget_respected": len(context_records)
        <= (1 if intervention_scope == "first-run-only" else selector["run_count"]),
        "interventions_have_context": (
            "context_records" not in selector
            or all(
                record.get("context_sha256")
                in {
                    context_record.get("context_sha256")
                    for context_record in context_records
                }
                for record in selector["intervention_records"]
            )
        ),
    }
    neutral_shadow = treatment_arm in LEARNED_SHADOW_ARMS
    if neutral_shadow:
        safety_checks.update(
            {
                "shadow_has_no_interventions": selector["interventions"] == 0,
                "one_record_per_shadow_evaluation": len(
                    selector.get("shadow_records", [])
                )
                == selector.get("shadow_evaluations", 0),
                **{
                    f"shadow_same_{field}": (
                        native_outcome[field] == treatment_outcome[field]
                    )
                    for field in STRUCTURAL_INTEGER_FIELDS
                },
                **{
                    f"shadow_same_{field}": _numbers_close(
                        native_outcome[field], treatment_outcome[field]
                    )
                    for field in STRUCTURAL_FLOAT_FIELDS
                },
            }
        )

    metrics = {}
    for field in (
        "nodes",
        "total_nodes",
        "lp_iterations",
        "lp_count",
        "cuts_applied",
        "solving_time",
        "arm_wall_time_seconds",
        "primal_dual_integral",
    ):
        if field not in native_outcome or field not in treatment_outcome:
            continue
        baseline_value = float(native_outcome[field])
        treatment_value = float(treatment_outcome[field])
        metrics[field] = {
            "native": baseline_value,
            "treatment": treatment_value,
            "delta_treatment_minus_native": treatment_value - baseline_value,
            "relative_saving": (
                (baseline_value - treatment_value) / baseline_value
                if baseline_value != 0.0
                else None
            ),
        }
    safe = all(safety_checks.values())
    eligible = selector["interventions"] > 0
    policy_fallback = (
        treatment_arm in POLICY_RANKING_ARMS
        and not eligible
    )
    policy_evaluable = neutral_shadow or eligible or policy_fallback
    return {
        "safe": safe,
        "eligible": eligible,
        "neutral_shadow": neutral_shadow,
        "policy_fallback": policy_fallback,
        "policy_evaluable": policy_evaluable,
        "valid": safe and policy_evaluable,
        "safety_checks": safety_checks,
        "metrics": metrics,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_child(
    instance: Path,
    arm: str,
    seed: int,
    time_limit: float,
    node_limit: int | None,
    output: Path,
    intervention_scope: str = "per-run",
    learned_model_manifest: Path | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "scip_cut_trace_v2.causal_harness",
        "solve",
        str(instance),
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--time-limit",
        str(time_limit),
        "--intervention-scope",
        intervention_scope,
        "--output",
        str(output),
    ]
    if node_limit is not None:
        command.extend(("--node-limit", str(node_limit)))
    if learned_model_manifest is not None:
        command.extend(("--learned-model-manifest", str(learned_model_manifest)))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Child arm {arm!r}, seed {seed} failed with exit code "
            f"{completed.returncode}:\n{completed.stderr.strip()}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _load_or_run_child(
    instance: Path,
    arm: str,
    seed: int,
    time_limit: float,
    node_limit: int | None,
    output: Path,
    reuse_existing: bool,
    expected_instance_sha256: str | None = None,
    intervention_scope: str = "per-run",
    learned_model_manifest: Path | None = None,
) -> dict[str, Any]:
    if not reuse_existing or not output.is_file():
        return _run_child(
            instance,
            arm,
            seed,
            time_limit,
            node_limit,
            output,
            intervention_scope,
            learned_model_manifest,
        )
    result = json.loads(output.read_text(encoding="utf-8"))
    checks = {
        "arm": result.get("arm") == arm,
        "seed": result.get("seed") == seed,
        "instance_sha256": result.get("instance_sha256")
        == (expected_instance_sha256 or _sha256_file(instance.resolve())),
        "time_limit": result.get("parameters", {}).get("limits/time") == time_limit,
        "node_limit": (
            result.get("parameters", {}).get("limits/nodes") == node_limit
            if node_limit is not None
            else "limits/nodes" not in result.get("parameters", {})
        ),
        "intervention_scope": result.get("selector", {}).get(
            "intervention_scope", "per-run"
        )
        == intervention_scope,
        "learned_model_manifest": (
            result.get("selector", {}).get("learned_model_manifest_sha256")
            == _sha256_file(learned_model_manifest.resolve())
            if learned_model_manifest is not None
            else result.get("selector", {}).get("learned_model_manifest") is None
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(
            f"Existing arm result does not match this experiment ({output}): {failed}"
        )
    return result


def _execute_arm_jobs(
    jobs: Iterable[dict[str, Any]], max_workers: int
) -> None:
    """Materialize resumable raw arm JSONs before deterministic aggregation."""
    jobs = tuple(jobs)
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    def execute(job: dict[str, Any]) -> None:
        _load_or_run_child(**job)

    if max_workers == 1:
        for job in jobs:
            execute(job)
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        for future in futures:
            future.result()


def _execution_arm_order(
    arms: Iterable[str], instance_id: str, seed: int, key: str | None
) -> tuple[str, ...]:
    arms = tuple(arms)
    if key is None:
        return arms
    return tuple(
        sorted(
            arms,
            key=lambda arm: hashlib.sha256(
                f"{key}\0{instance_id}\0{seed}\0{arm}".encode("utf-8")
            ).digest(),
        )
    )


def run_structural_parity(
    instance: Path,
    candidate_arm: str,
    seeds: Iterable[int],
    time_limit: float,
    node_limit: int | None,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if candidate_arm not in PARITY_CANDIDATE_ARMS:
        raise ValueError(f"Unsupported parity candidate arm: {candidate_arm}")
    seeds = tuple(seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for seed in seeds:
        arm_results = {}
        for arm in ("native", candidate_arm):
            output = output_dir / f"seed_{seed}_{arm}.json"
            arm_results[arm] = _run_child(
                instance, arm, seed, time_limit, node_limit, output
            )
        pairs.append(
            {
                "seed": seed,
                "native_result": str((output_dir / f"seed_{seed}_native.json").resolve()),
                "candidate_result": str(
                    (output_dir / f"seed_{seed}_{candidate_arm}.json").resolve()
                ),
                "native_outcome": arm_results["native"]["outcome"],
                "candidate_outcome": arm_results[candidate_arm]["outcome"],
                "candidate_selector": arm_results[candidate_arm]["selector"],
                "comparison": compare_pair(
                    arm_results["native"], arm_results[candidate_arm], candidate_arm
                ),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": f"native-vs-{candidate_arm}-cut-selector",
        "candidate_arm": candidate_arm,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "instance": str(instance.resolve()),
        "instance_sha256": _sha256_file(instance.resolve()),
        "seeds": list(seeds),
        "time_limit": time_limit,
        "node_limit": node_limit,
        "process_isolation": "one fresh Python/SCIP process per arm and seed",
        "gate_contract": {
            "gating": (
                "status, bounds, gap, nodes, total nodes, LP iterations, LP count, "
                "cuts applied, versions, parameters, and an exercised candidate callback"
            ),
            "non_gating": (
                "wall-clock solving time and primal-dual integral include Python callback "
                "overhead and are reported but excluded from structural parity"
            ),
        },
        "passed": bool(pairs) and all(pair["comparison"]["passed"] for pair in pairs),
        "pairs": pairs,
    }
    _write_json(manifest_path, manifest)
    return manifest


def run_noop_parity(
    instance: Path,
    seeds: Iterable[int],
    time_limit: float,
    node_limit: int | None,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    return run_structural_parity(
        instance,
        "noop",
        seeds,
        time_limit,
        node_limit,
        output_dir,
        manifest_path,
    )


def run_structural_parity_suite(
    instances: Iterable[Path],
    candidate_arms: Iterable[str],
    seeds: Iterable[int],
    time_limit: float,
    node_limit: int | None,
    output_dir: Path,
    manifest_path: Path,
    reuse_existing: bool = False,
    max_workers: int = 1,
) -> dict[str, Any]:
    """Run no-op/direct-hybrid parity across independent instances and seeds."""
    instances = tuple(Path(instance).resolve() for instance in instances)
    candidate_arms = tuple(dict.fromkeys(candidate_arms))
    seeds = tuple(seeds)
    if not instances:
        raise ValueError("at least one parity instance is required")
    if len({_instance_id(instance) for instance in instances}) != len(instances):
        raise ValueError("Parity suite instance IDs must be unique")
    unsupported = [arm for arm in candidate_arms if arm not in PARITY_CANDIDATE_ARMS]
    if unsupported or not candidate_arms:
        raise ValueError(f"Unsupported parity candidate arms: {unsupported}")

    instance_hashes = {instance: _sha256_file(instance) for instance in instances}
    jobs = []
    for instance in instances:
        instance_dir = output_dir / _instance_id(instance)
        for seed in seeds:
            for arm in ("native", *candidate_arms):
                jobs.append(
                    {
                        "instance": instance,
                        "arm": arm,
                        "seed": seed,
                        "time_limit": time_limit,
                        "node_limit": node_limit,
                        "output": instance_dir / f"seed_{seed}_{arm}.json",
                        "reuse_existing": reuse_existing,
                        "expected_instance_sha256": instance_hashes[instance],
                        "intervention_scope": "per-run",
                    }
                )
    _execute_arm_jobs(jobs, max_workers)

    per_instance = []
    for instance in instances:
        instance_id = _instance_id(instance)
        instance_dir = output_dir / instance_id
        pairs = []
        for seed in seeds:
            native_path = instance_dir / f"seed_{seed}_native.json"
            native = _load_or_run_child(
                instance,
                "native",
                seed,
                time_limit,
                node_limit,
                native_path,
                True,
                instance_hashes[instance],
            )
            for candidate_arm in candidate_arms:
                candidate_path = instance_dir / f"seed_{seed}_{candidate_arm}.json"
                candidate = _load_or_run_child(
                    instance,
                    candidate_arm,
                    seed,
                    time_limit,
                    node_limit,
                    candidate_path,
                    True,
                    instance_hashes[instance],
                )
                comparison = compare_pair(native, candidate, candidate_arm)
                pairs.append(
                    {
                        "seed": seed,
                        "candidate_arm": candidate_arm,
                        "native_result": str(native_path.resolve()),
                        "candidate_result": str(candidate_path.resolve()),
                        "native_outcome": native["outcome"],
                        "candidate_outcome": candidate["outcome"],
                        "candidate_selector": candidate["selector"],
                        "comparison": comparison,
                        "classification": classify_parity_pair(
                            native, candidate, comparison
                        ),
                    }
                )
        per_instance.append(
            {
                "instance_id": instance_id,
                "instance": str(instance),
                "instance_sha256": instance_hashes[instance],
                "passed": all(pair["comparison"]["passed"] for pair in pairs),
                "pairs": pairs,
            }
        )

    arm_summary = {
        arm: {
            "pairs": sum(
                pair["candidate_arm"] == arm
                for instance in per_instance
                for pair in instance["pairs"]
            ),
            "passed_pairs": sum(
                pair["candidate_arm"] == arm and pair["comparison"]["passed"]
                for instance in per_instance
                for pair in instance["pairs"]
            ),
            "passed_instances": sum(
                all(
                    pair["comparison"]["passed"]
                    for pair in instance["pairs"]
                    if pair["candidate_arm"] == arm
                )
                for instance in per_instance
            ),
            "classification_counts": {
                classification: sum(
                    pair["candidate_arm"] == arm
                    and pair["classification"] == classification
                    for instance in per_instance
                    for pair in instance["pairs"]
                )
                for classification in (
                    "passed",
                    "callback_not_exercised",
                    "both_arms_same_incomplete_limit_status",
                    "structural_mismatch",
                )
            },
            "complete_callback_exercised_pairs": sum(
                pair["candidate_arm"] == arm
                and pair["candidate_selector"]["calls"] > 0
                and pair["native_outcome"]["status"] in COMPLETE_STATUSES
                and pair["candidate_outcome"]["status"] in COMPLETE_STATUSES
                for instance in per_instance
                for pair in instance["pairs"]
            ),
            "complete_callback_exercised_passed_pairs": sum(
                pair["candidate_arm"] == arm
                and pair["candidate_selector"]["calls"] > 0
                and pair["native_outcome"]["status"] in COMPLETE_STATUSES
                and pair["candidate_outcome"]["status"] in COMPLETE_STATUSES
                and pair["comparison"]["passed"]
                for instance in per_instance
                for pair in instance["pairs"]
            ),
        }
        for arm in candidate_arms
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "multi-instance-structural-cut-selector-parity-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "instances": len(instances),
        "candidate_arms": list(candidate_arms),
        "seeds": list(seeds),
        "time_limit": time_limit,
        "node_limit": node_limit,
        "max_workers": max_workers,
        "process_isolation": "one fresh Python/SCIP process per arm and seed",
        "gate_contract": {
            "gating": (
                "status, bounds, gap, nodes, total nodes, LP iterations, LP count, "
                "cuts applied, versions, parameters, and exercised callbacks"
            ),
            "non_gating": "wall-clock solving time and primal-dual integral",
        },
        "passed": bool(per_instance) and all(item["passed"] for item in per_instance),
        "diagnostic_complete_evidence_passed": all(
            summary["complete_callback_exercised_pairs"] > 0
            and summary["complete_callback_exercised_pairs"]
            == summary["complete_callback_exercised_passed_pairs"]
            for summary in arm_summary.values()
        ),
        "arm_summary": arm_summary,
        "per_instance": per_instance,
    }
    _write_json(manifest_path, manifest)
    return manifest


def run_treatment_experiment(
    instance: Path,
    treatment_arm: str,
    seeds: Iterable[int],
    time_limit: float,
    node_limit: int | None,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if treatment_arm not in TREATMENT_ARMS:
        raise ValueError(f"Unsupported treatment arm: {treatment_arm}")
    seeds = tuple(seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for seed in seeds:
        arm_results = {}
        for arm in ("native", treatment_arm):
            output = output_dir / f"seed_{seed}_{arm}.json"
            arm_results[arm] = _run_child(
                instance, arm, seed, time_limit, node_limit, output
            )
        pairs.append(
            {
                "seed": seed,
                "native_outcome": arm_results["native"]["outcome"],
                "treatment_arm": treatment_arm,
                "treatment_outcome": arm_results[treatment_arm]["outcome"],
                "treatment_selector": arm_results[treatment_arm]["selector"],
                "comparison": compare_causal_pair(
                    arm_results["native"], arm_results[treatment_arm]
                ),
            }
        )

    safe_pairs = [pair for pair in pairs if pair["comparison"]["safe"]]
    valid_pairs = [pair for pair in pairs if pair["comparison"]["valid"]]
    aggregate = {}
    if valid_pairs:
        for field in valid_pairs[0]["comparison"]["metrics"]:
            values = [
                pair["comparison"]["metrics"][field] for pair in valid_pairs
            ]
            aggregate[field] = {
                "mean_delta_treatment_minus_native": sum(
                    value["delta_treatment_minus_native"] for value in values
                )
                / len(values),
                "wins": sum(
                    value["delta_treatment_minus_native"] < 0.0 for value in values
                ),
                "ties": sum(
                    value["delta_treatment_minus_native"] == 0.0 for value in values
                ),
                "losses": sum(
                    value["delta_treatment_minus_native"] > 0.0 for value in values
                ),
            }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": f"native-vs-root-{treatment_arm}-once-per-run",
        "treatment_arm": treatment_arm,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "instance": str(instance.resolve()),
        "instance_sha256": _sha256_file(instance.resolve()),
        "seeds": list(seeds),
        "time_limit": time_limit,
        "node_limit": node_limit,
        "treatment_contract": (
            "at the first eligible root callback of each SCIP run, call native hybrid, "
            f"keep its selected count, and {TREATMENT_CONTRACTS[treatment_arm]}; "
            "delegate all later callbacks in that run"
        ),
        "safe": bool(pairs) and len(safe_pairs) == len(pairs),
        "valid": bool(pairs) and len(valid_pairs) == len(pairs),
        "safe_pairs": len(safe_pairs),
        "eligible_pairs": len(valid_pairs),
        "valid_pairs": len(valid_pairs),
        "pairs": pairs,
        "aggregate": aggregate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def run_boundary_swap_experiment(
    instance: Path,
    seeds: Iterable[int],
    time_limit: float,
    node_limit: int | None,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    return run_treatment_experiment(
        instance,
        "boundary-swap",
        seeds,
        time_limit,
        node_limit,
        output_dir,
        manifest_path,
    )


def _instance_id(path: Path) -> str:
    name = path.name
    for suffix in (".mps.gz", ".mps", ".lp.gz", ".lp"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _compact_selector(selector: dict[str, Any]) -> dict[str, Any]:
    compact = dict(selector)
    compact["context_records"] = [
        {
            key: value
            for key, value in record.items()
            if key != "decision_context"
        }
        for record in selector.get("context_records", [])
    ]
    return compact


def _aggregate_comparisons(
    comparisons: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    valid = [comparison for comparison in comparisons if comparison["valid"]]
    aggregate: dict[str, Any] = {}
    if not valid:
        return aggregate
    for field in valid[0]["metrics"]:
        deltas = [
            comparison["metrics"][field]["delta_treatment_minus_native"]
            for comparison in valid
        ]
        relative_savings = [
            comparison["metrics"][field]["relative_saving"]
            for comparison in valid
            if comparison["metrics"][field]["relative_saving"] is not None
        ]
        aggregate[field] = {
            "valid_pairs": len(deltas),
            "mean_delta_treatment_minus_native": sum(deltas) / len(deltas),
            "mean_relative_saving": (
                sum(relative_savings) / len(relative_savings)
                if relative_savings
                else None
            ),
            "wins": sum(delta < 0.0 for delta in deltas),
            "ties": sum(delta == 0.0 for delta in deltas),
            "losses": sum(delta > 0.0 for delta in deltas),
        }
    return aggregate


def select_oracle_action(
    comparisons: dict[str, dict[str, Any]],
    primary_metric: str = PRIMARY_ORACLE_METRIC,
) -> dict[str, Any]:
    """Choose the post-hoc best valid action, with native winning every tie."""
    if not comparisons:
        raise ValueError("at least one action comparison is required")
    native_values = {
        float(comparison["metrics"][primary_metric]["native"])
        for comparison in comparisons.values()
    }
    if len(native_values) != 1:
        raise ValueError("all action comparisons must share one native baseline")
    native_value = native_values.pop()
    candidates = [(native_value, 0, "native")]
    for action_index, action in enumerate(TREATMENT_ARMS, start=1):
        comparison = comparisons.get(action)
        if comparison is not None and comparison["valid"]:
            candidates.append(
                (
                    float(comparison["metrics"][primary_metric]["treatment"]),
                    action_index,
                    action,
                )
            )
    selected_value, _, selected_action = min(candidates)
    return {
        "primary_metric": primary_metric,
        "selected_action": selected_action,
        "native_value": native_value,
        "selected_value": selected_value,
        "delta_selected_minus_native": selected_value - native_value,
        "relative_saving": (
            (native_value - selected_value) / native_value
            if native_value != 0.0
            else None
        ),
        "valid_treatment_actions": [
            action
            for action in TREATMENT_ARMS
            if action in comparisons and comparisons[action]["valid"]
        ],
    }


def evaluate_leave_one_seed_out(
    pairs: Iterable[dict[str, Any]],
    actions: Iterable[str],
    primary_metric: str = PRIMARY_ORACLE_METRIC,
) -> dict[str, Any]:
    """Choose on all other seeds and evaluate the action on the held-out seed."""
    pairs = tuple(pairs)
    actions = tuple(actions)

    def policy_relative_saving(comparison: dict[str, Any]) -> float | None:
        if not comparison["safe"]:
            return None
        if not comparison["eligible"]:
            return 0.0
        return comparison["metrics"][primary_metric]["relative_saving"]

    evaluations = []
    for held_pair in pairs:
        training_pairs = [pair for pair in pairs if pair is not held_pair]
        training_scores = {}
        for action in actions:
            scores = [
                policy_relative_saving(pair["actions"][action]["comparison"])
                for pair in training_pairs
            ]
            if scores and all(score is not None for score in scores):
                training_scores[action] = sum(scores) / len(scores)

        selected_action = "native"
        selected_training_score = 0.0
        for action in actions:
            score = training_scores.get(action)
            if score is not None and score > selected_training_score:
                selected_action = action
                selected_training_score = score

        if selected_action == "native":
            held_safe = held_pair["native_outcome"]["status"] in COMPLETE_STATUSES
            held_eligible = False
            effective_action = "native"
            delta = 0.0 if held_safe else None
            relative_saving = 0.0 if held_safe else None
        else:
            comparison = held_pair["actions"][selected_action]["comparison"]
            held_safe = comparison["safe"]
            held_eligible = comparison["eligible"]
            if held_safe and not held_eligible:
                effective_action = "native-ineligible-fallback"
                delta = 0.0
                relative_saving = 0.0
            elif comparison["valid"]:
                effective_action = selected_action
                metric = comparison["metrics"][primary_metric]
                delta = metric["delta_treatment_minus_native"]
                relative_saving = metric["relative_saving"]
            else:
                effective_action = selected_action
                delta = None
                relative_saving = None

        evaluations.append(
            {
                "held_out_seed": held_pair["seed"],
                "training_seeds": [pair["seed"] for pair in training_pairs],
                "training_action_mean_relative_saving": training_scores,
                "selected_action": selected_action,
                "selected_training_mean_relative_saving": selected_training_score,
                "effective_action": effective_action,
                "held_out_safe": held_safe,
                "held_out_eligible": held_eligible,
                "held_out_delta_selected_minus_native": delta,
                "held_out_relative_saving": relative_saving,
            }
        )

    evaluable = [
        evaluation
        for evaluation in evaluations
        if evaluation["held_out_relative_saving"] is not None
    ]
    relative_savings = [
        evaluation["held_out_relative_saving"] for evaluation in evaluable
    ]
    return {
        "safe": bool(evaluations)
        and all(evaluation["held_out_safe"] for evaluation in evaluations),
        "evaluations": evaluations,
        "evaluable_seeds": len(evaluable),
        "unsafe_seeds": sum(
            not evaluation["held_out_safe"] for evaluation in evaluations
        ),
        "mean_relative_saving": (
            sum(relative_savings) / len(relative_savings)
            if relative_savings
            else None
        ),
        "wins": sum(value > 0.0 for value in relative_savings),
        "ties": sum(value == 0.0 for value in relative_savings),
        "losses": sum(value < 0.0 for value in relative_savings),
        "selected_action_counts": {
            action: sum(
                evaluation["selected_action"] == action
                for evaluation in evaluations
            )
            for action in ("native", *actions)
        },
    }
def run_action_oracle_suite(
    instances: Iterable[Path],
    actions: Iterable[str],
    seeds: Iterable[int],
    time_limit: float,
    node_limit: int | None,
    output_dir: Path,
    manifest_path: Path,
    reuse_existing: bool = False,
    intervention_scope: str = "per-run",
    learned_model_manifest: Path | None = None,
    max_workers: int = 1,
    execution_order_key: str | None = None,
    execution_schedule: dict[tuple[str, int], tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    instances = tuple(Path(instance).resolve() for instance in instances)
    actions = tuple(dict.fromkeys(actions))
    seeds = tuple(seeds)
    unsupported = [action for action in actions if action not in TREATMENT_ARMS]
    if unsupported:
        raise ValueError(f"Unsupported treatment arms: {unsupported}")
    if not actions:
        raise ValueError("at least one treatment action is required")
    if any(action in LEARNED_TREATMENT_ARMS for action in actions):
        if learned_model_manifest is None:
            raise ValueError("A learned model manifest is required for learned actions")
        learned_model_manifest = learned_model_manifest.resolve()
    if intervention_scope not in INTERVENTION_SCOPES:
        raise ValueError(f"Unsupported intervention scope: {intervention_scope}")
    if len({_instance_id(instance) for instance in instances}) != len(instances):
        raise ValueError("Suite instance IDs must be unique")

    instance_hashes = {instance: _sha256_file(instance) for instance in instances}
    jobs = []
    for instance in instances:
        instance_output = output_dir / _instance_id(instance)
        for seed in seeds:
            block_key = (_instance_id(instance), seed)
            arms = (
                execution_schedule[block_key]
                if execution_schedule is not None
                else _execution_arm_order(
                    ("native", *actions),
                    block_key[0],
                    seed,
                    execution_order_key,
                )
            )
            if set(arms) != {"native", *actions} or len(arms) != len(actions) + 1:
                raise ValueError(f"Invalid arm schedule for {block_key}: {arms}")
            for arm in arms:
                jobs.append(
                    {
                        "instance": instance,
                        "arm": arm,
                        "seed": seed,
                        "time_limit": time_limit,
                        "node_limit": node_limit,
                        "output": instance_output / f"seed_{seed}_{arm}.json",
                        "reuse_existing": reuse_existing,
                        "expected_instance_sha256": instance_hashes[instance],
                        "intervention_scope": intervention_scope,
                        "learned_model_manifest": (
                            learned_model_manifest
                            if arm in LEARNED_TREATMENT_ARMS
                            else None
                        ),
                    }
                )
    _execute_arm_jobs(jobs, max_workers)
    reuse_existing = True

    instance_results = []
    for instance in instances:
        instance_id = _instance_id(instance)
        instance_sha256 = instance_hashes[instance]
        instance_output = output_dir / instance_id
        instance_output.mkdir(parents=True, exist_ok=True)
        pairs = []
        for seed in seeds:
            native_path = instance_output / f"seed_{seed}_native.json"
            native = _load_or_run_child(
                instance,
                "native",
                seed,
                time_limit,
                node_limit,
                native_path,
                reuse_existing,
                instance_sha256,
                intervention_scope,
            )
            action_results = {}
            comparisons = {}
            for action in actions:
                action_path = instance_output / f"seed_{seed}_{action}.json"
                treatment = _load_or_run_child(
                    instance,
                    action,
                    seed,
                    time_limit,
                    node_limit,
                    action_path,
                    reuse_existing,
                    instance_sha256,
                    intervention_scope,
                    (
                        learned_model_manifest
                        if action in LEARNED_TREATMENT_ARMS
                        else None
                    ),
                )
                comparison = compare_causal_pair(native, treatment)
                comparisons[action] = comparison
                context_records = treatment["selector"].get("context_records", [])
                action_results[action] = {
                    "result_path": str(action_path.resolve()),
                    "outcome": treatment["outcome"],
                    "selector": _compact_selector(treatment["selector"]),
                    "comparison": comparison,
                    "initial_context_sha256": (
                        context_records[0]["context_sha256"]
                        if context_records
                        else None
                    ),
                }
            initial_context_hashes = {
                action: result["initial_context_sha256"]
                for action, result in action_results.items()
            }
            observed_context_hashes = [
                value for value in initial_context_hashes.values() if value is not None
            ]
            pairs.append(
                {
                    "seed": seed,
                    "native_result_path": str(native_path.resolve()),
                    "native_outcome": native["outcome"],
                    "actions": action_results,
                    "initial_context": {
                        "sha256_by_action": initial_context_hashes,
                        "all_actions_observed": len(observed_context_hashes)
                        == len(actions),
                        "no_action_observed": not observed_context_hashes,
                        "partial_actions_observed": 0
                        < len(observed_context_hashes)
                        < len(actions),
                        "matching_across_actions": len(observed_context_hashes)
                        == len(actions)
                        and len(set(observed_context_hashes)) == 1,
                    },
                    "oracle": select_oracle_action(comparisons),
                }
            )

        per_action = {}
        for action in actions:
            comparisons = [pair["actions"][action]["comparison"] for pair in pairs]
            per_action[action] = {
                "safe": bool(comparisons)
                and all(comparison["safe"] for comparison in comparisons),
                "safe_pairs": sum(comparison["safe"] for comparison in comparisons),
                "eligible_pairs": sum(
                    comparison["eligible"] for comparison in comparisons
                ),
                "neutral_shadow_pairs": sum(
                    comparison.get("neutral_shadow", False)
                    for comparison in comparisons
                ),
                "valid_pairs": sum(comparison["valid"] for comparison in comparisons),
                "unsafe_eligible_pairs": sum(
                    comparison["eligible"] and not comparison["safe"]
                    for comparison in comparisons
                ),
                "policy_fallback_pairs": sum(
                    comparison.get("policy_fallback", False)
                    for comparison in comparisons
                ),
                "unsafe_policy_pairs": sum(
                    comparison.get("policy_evaluable", comparison["eligible"])
                    and not comparison["safe"]
                    for comparison in comparisons
                ),
                "aggregate": _aggregate_comparisons(comparisons),
            }
        oracle_deltas = [pair["oracle"]["delta_selected_minus_native"] for pair in pairs]
        oracle_relative_savings = [
            pair["oracle"]["relative_saving"]
            for pair in pairs
            if pair["oracle"]["relative_saving"] is not None
        ]
        leave_one_seed_out = evaluate_leave_one_seed_out(pairs, actions)
        instance_results.append(
            {
                "instance_id": instance_id,
                "instance": str(instance),
                "instance_sha256": instance_sha256,
                "pairs": pairs,
                "per_action": per_action,
                "oracle": {
                    "mean_delta_selected_minus_native": (
                        sum(oracle_deltas) / len(oracle_deltas)
                        if oracle_deltas
                        else None
                    ),
                    "mean_relative_saving": (
                        sum(oracle_relative_savings) / len(oracle_relative_savings)
                        if oracle_relative_savings
                        else None
                    ),
                    "wins": sum(delta < 0.0 for delta in oracle_deltas),
                    "ties": sum(delta == 0.0 for delta in oracle_deltas),
                    "selected_action_counts": {
                        action: sum(
                            pair["oracle"]["selected_action"] == action
                            for pair in pairs
                        )
                        for action in ("native", *actions)
                    },
                },
                "leave_one_seed_out": leave_one_seed_out,
            }
        )

    action_summary = {}
    for action in actions:
        results = [result["per_action"][action] for result in instance_results]
        instance_deltas = [
            result["aggregate"][PRIMARY_ORACLE_METRIC][
                "mean_delta_treatment_minus_native"
            ]
            for result in results
            if PRIMARY_ORACLE_METRIC in result["aggregate"]
        ]
        instance_relative_savings = [
            result["aggregate"][PRIMARY_ORACLE_METRIC]["mean_relative_saving"]
            for result in results
            if PRIMARY_ORACLE_METRIC in result["aggregate"]
            and result["aggregate"][PRIMARY_ORACLE_METRIC]["mean_relative_saving"]
            is not None
        ]
        action_summary[action] = {
            "safe": bool(results) and all(result["safe"] for result in results),
            "unsafe_eligible_pairs": sum(
                result["unsafe_eligible_pairs"] for result in results
            ),
            "policy_fallback_pairs": sum(
                result["policy_fallback_pairs"] for result in results
            ),
            "unsafe_policy_pairs": sum(
                result["unsafe_policy_pairs"] for result in results
            ),
            "eligible_pairs": sum(result["eligible_pairs"] for result in results),
            "neutral_shadow_pairs": sum(
                result["neutral_shadow_pairs"] for result in results
            ),
            "valid_pairs": sum(result["valid_pairs"] for result in results),
            "eligible_instances": len(instance_deltas),
            "instance_equal_mean_primary_delta": (
                sum(instance_deltas) / len(instance_deltas)
                if instance_deltas
                else None
            ),
            "instance_equal_mean_relative_saving": (
                sum(instance_relative_savings) / len(instance_relative_savings)
                if instance_relative_savings
                else None
            ),
            "instance_wins": sum(delta < 0.0 for delta in instance_deltas),
            "instance_ties": sum(delta == 0.0 for delta in instance_deltas),
            "instance_losses": sum(delta > 0.0 for delta in instance_deltas),
        }

    oracle_instance_deltas = [
        result["oracle"]["mean_delta_selected_minus_native"]
        for result in instance_results
        if result["oracle"]["mean_delta_selected_minus_native"] is not None
    ]
    oracle_instance_relative_savings = [
        result["oracle"]["mean_relative_saving"]
        for result in instance_results
        if result["oracle"]["mean_relative_saving"] is not None
    ]
    stability_instance_savings = [
        result["leave_one_seed_out"]["mean_relative_saving"]
        for result in instance_results
        if result["leave_one_seed_out"]["mean_relative_saving"] is not None
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "predeclared-root-action-library-oracle-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "instances": len(instance_results),
        "seeds": list(seeds),
        "actions": list(actions),
        "time_limit": time_limit,
        "node_limit": node_limit,
        "intervention_scope": intervention_scope,
        "learned_model_manifest": (
            str(learned_model_manifest) if learned_model_manifest is not None else None
        ),
        "learned_model_manifest_sha256": (
            _sha256_file(learned_model_manifest)
            if learned_model_manifest is not None
            else None
        ),
        "max_workers": max_workers,
        "execution_order": (
            {
                "method": "explicit pre-registered balanced block schedule",
                "blocks": len(execution_schedule),
                "schedule_sha256": hashlib.sha256(
                    json.dumps(
                        [
                            [instance_id, seed, list(arms)]
                            for (instance_id, seed), arms in sorted(
                                execution_schedule.items()
                            )
                        ],
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            if execution_schedule is not None
            else
            {
                "method": "sha256 permutation within each instance-seed arm block",
                "key": execution_order_key,
                "key_sha256": hashlib.sha256(execution_order_key.encode("utf-8")).hexdigest(),
            }
            if execution_order_key is not None
            else {"method": "native then actions in manifest order"}
        ),
        "process_isolation": "one fresh Python/SCIP process per arm and seed",
        "pre_registered_contract": {
            "intervention_budget": (
                "at most one action at an eligible root callback per SCIP run"
                if intervention_scope == "per-run"
                else "at most one action in the first SCIP run of the complete solve"
            ),
            "selected_cut_budget": "preserve native hybrid selected-cut count",
            "primary_metric": PRIMARY_ORACLE_METRIC,
            "tie_rule": "native wins every primary-metric tie",
            "safety": (
                "both arms complete with matching status, objective sense, final "
                "primal and dual bounds, runtime, parameters, and run budget"
            ),
            "oracle_scope": (
                "post-hoc per instance and seed; an upper bound only, never a deployable policy"
            ),
        },
        "all_actions_safe": bool(action_summary)
        and all(summary["safe"] for summary in action_summary.values()),
        "initial_context_match_pairs": sum(
            pair["initial_context"]["matching_across_actions"]
            for result in instance_results
            for pair in result["pairs"]
        ),
        "initial_context_total_pairs": sum(
            len(result["pairs"]) for result in instance_results
        ),
        "initial_context_observed_pairs": sum(
            pair["initial_context"]["all_actions_observed"]
            for result in instance_results
            for pair in result["pairs"]
        ),
        "initial_context_missing_pairs": sum(
            pair["initial_context"]["no_action_observed"]
            for result in instance_results
            for pair in result["pairs"]
        ),
        "initial_context_partial_pairs": sum(
            pair["initial_context"]["partial_actions_observed"]
            for result in instance_results
            for pair in result["pairs"]
        ),
        "all_observed_contexts_match": all(
            pair["initial_context"]["matching_across_actions"]
            for result in instance_results
            for pair in result["pairs"]
            if pair["initial_context"]["all_actions_observed"]
        ),
        "action_summary": action_summary,
        "oracle_summary": {
            "primary_metric": PRIMARY_ORACLE_METRIC,
            "eligible_instances": len(oracle_instance_deltas),
            "instance_equal_mean_delta_selected_minus_native": (
                sum(oracle_instance_deltas) / len(oracle_instance_deltas)
                if oracle_instance_deltas
                else None
            ),
            "instance_equal_mean_relative_saving": (
                sum(oracle_instance_relative_savings)
                / len(oracle_instance_relative_savings)
                if oracle_instance_relative_savings
                else None
            ),
            "instance_wins": sum(delta < 0.0 for delta in oracle_instance_deltas),
            "instance_ties": sum(delta == 0.0 for delta in oracle_instance_deltas),
            "instance_losses": sum(delta > 0.0 for delta in oracle_instance_deltas),
            "selected_action_counts": {
                action: sum(
                    result["oracle"]["selected_action_counts"].get(action, 0)
                    for result in instance_results
                )
                for action in ("native", *actions)
            },
        },
        "leave_one_seed_out_summary": {
            "selection_rule": (
                "choose the action with the largest positive mean relative LP-iteration "
                "saving on all other seeds; exclude actions with any unsafe training "
                "seed; native wins ties"
            ),
            "safe": bool(instance_results)
            and all(
                result["leave_one_seed_out"]["safe"]
                for result in instance_results
            ),
            "unsafe_held_out_seeds": sum(
                result["leave_one_seed_out"]["unsafe_seeds"]
                for result in instance_results
            ),
            "instance_equal_mean_relative_saving": (
                sum(stability_instance_savings) / len(stability_instance_savings)
                if stability_instance_savings
                else None
            ),
            "instance_wins": sum(value > 0.0 for value in stability_instance_savings),
            "instance_ties": sum(value == 0.0 for value in stability_instance_savings),
            "instance_losses": sum(value < 0.0 for value in stability_instance_savings),
            "seed_wins": sum(
                result["leave_one_seed_out"]["wins"] for result in instance_results
            ),
            "seed_ties": sum(
                result["leave_one_seed_out"]["ties"] for result in instance_results
            ),
            "seed_losses": sum(
                result["leave_one_seed_out"]["losses"] for result in instance_results
            ),
            "selected_action_counts": {
                action: sum(
                    result["leave_one_seed_out"]["selected_action_counts"].get(
                        action, 0
                    )
                    for result in instance_results
                )
                for action in ("native", *actions)
            },
        },
        "per_instance": instance_results,
    }
    _write_json(manifest_path, manifest)
    return manifest


def run_boundary_swap_suite(
    instances: Iterable[Path],
    seeds: Iterable[int],
    time_limit: float,
    node_limit: int | None,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    instances = tuple(Path(instance).resolve() for instance in instances)
    seeds = tuple(seeds)
    if len({_instance_id(instance) for instance in instances}) != len(instances):
        raise ValueError("Suite instance IDs must be unique")

    instance_results = []
    for instance in instances:
        instance_id = _instance_id(instance)
        instance_output = output_dir / instance_id
        result = run_boundary_swap_experiment(
            instance,
            seeds,
            time_limit,
            node_limit,
            instance_output,
            instance_output / "manifest.json",
        )
        records = [
            record
            for pair in result["pairs"]
            for record in pair["treatment_selector"]["intervention_records"]
        ]
        instance_results.append(
            {
                "instance_id": instance_id,
                "instance": str(instance),
                "instance_sha256": result["instance_sha256"],
                "safe": result["safe"],
                "eligible_pairs": result["eligible_pairs"],
                "noncomparable_eligible_pairs": sum(
                    pair["comparison"]["eligible"]
                    and not pair["comparison"]["safe"]
                    for pair in result["pairs"]
                ),
                "total_pairs": len(result["pairs"]),
                "aggregate": result["aggregate"],
                "intervention_records": records,
            }
        )

    metric_names = sorted(
        {
            metric
            for result in instance_results
            for metric in result["aggregate"]
        }
    )
    aggregate = {}
    for metric in metric_names:
        deltas = [
            result["aggregate"][metric]["mean_delta_treatment_minus_native"]
            for result in instance_results
            if metric in result["aggregate"]
        ]
        aggregate[metric] = {
            "eligible_instances": len(deltas),
            "instance_equal_mean_delta_treatment_minus_native": (
                sum(deltas) / len(deltas) if deltas else None
            ),
            "instance_wins": sum(delta < 0.0 for delta in deltas),
            "instance_ties": sum(delta == 0.0 for delta in deltas),
            "instance_losses": sum(delta > 0.0 for delta in deltas),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "root-boundary-swap-train-suite-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds),
        "time_limit": time_limit,
        "node_limit": node_limit,
        "aggregation": (
            "first average paired seed deltas within each instance, then weight "
            "eligible instances equally; metric aggregates exclude noncomparable "
            "pairs, which are reported separately and still fail suite safety"
        ),
        "safe": bool(instance_results)
        and all(result["safe"] for result in instance_results),
        "instances": len(instance_results),
        "eligible_instances": sum(
            result["eligible_pairs"] > 0 for result in instance_results
        ),
        "noncomparable_eligible_pairs": sum(
            result["noncomparable_eligible_pairs"] for result in instance_results
        ),
        "unsafe_instances": [
            result["instance_id"] for result in instance_results if not result["safe"]
        ],
        "per_instance": instance_results,
        "aggregate": aggregate,
    }
    _write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve = subparsers.add_parser("solve", help="run one process-isolated arm")
    solve.add_argument("instance", type=Path)
    solve.add_argument("--arm", choices=ARMS, required=True)
    solve.add_argument("--seed", type=int, required=True)
    solve.add_argument("--time-limit", type=float, default=300.0)
    solve.add_argument("--node-limit", type=int)
    solve.add_argument(
        "--intervention-scope",
        choices=INTERVENTION_SCOPES,
        default="per-run",
    )
    solve.add_argument("--learned-model-manifest", type=Path)
    solve.add_argument("--output", type=Path, required=True)

    parity = subparsers.add_parser(
        "noop-parity", help="compare native SCIP against an exercised no-op selector"
    )
    parity.add_argument("instance", type=Path)
    parity.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parity.add_argument("--time-limit", type=float, default=300.0)
    parity.add_argument("--node-limit", type=int)
    parity.add_argument(
        "--output-dir", type=Path, default=Path("experiments/noop_parity_v1")
    )
    parity.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/causal_noop_parity_v1.json"),
    )

    direct = subparsers.add_parser(
        "direct-hybrid-parity",
        help="compare native SCIP against a direct call to its hybrid selector",
    )
    direct.add_argument("instance", type=Path)
    direct.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    direct.add_argument("--time-limit", type=float, default=300.0)
    direct.add_argument("--node-limit", type=int)
    direct.add_argument(
        "--output-dir", type=Path, default=Path("experiments/direct_hybrid_parity_v1")
    )
    direct.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/causal_direct_hybrid_parity_v1.json"),
    )

    parity_suite = subparsers.add_parser(
        "parity-suite",
        help="run no-op and direct-hybrid structural parity across instances",
    )
    parity_suite.add_argument("instances", type=Path, nargs="+")
    parity_suite.add_argument(
        "--candidate-arms",
        choices=PARITY_CANDIDATE_ARMS,
        nargs="+",
        default=list(PARITY_CANDIDATE_ARMS),
    )
    parity_suite.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parity_suite.add_argument("--time-limit", type=float, default=300.0)
    parity_suite.add_argument("--node-limit", type=int)
    parity_suite.add_argument("--reuse-existing", action="store_true")
    parity_suite.add_argument("--jobs", type=int, default=1)
    parity_suite.add_argument(
        "--output-dir", type=Path, default=Path("experiments/parity_suite_v1")
    )
    parity_suite.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/causal_parity_suite_v1.json"),
    )

    treatment = subparsers.add_parser(
        "boundary-swap-pair",
        help="run one root boundary-swap treatment per SCIP run",
    )
    treatment.add_argument("instance", type=Path)
    treatment.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    treatment.add_argument("--time-limit", type=float, default=300.0)
    treatment.add_argument("--node-limit", type=int)
    treatment.add_argument(
        "--output-dir", type=Path, default=Path("experiments/boundary_swap_pilot_v1")
    )
    treatment.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/causal_boundary_swap_pilot_v1.json"),
    )

    suite = subparsers.add_parser(
        "boundary-swap-suite",
        help="run an instance-equal multi-instance boundary-swap experiment",
    )
    suite.add_argument("instances", type=Path, nargs="+")
    suite.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    suite.add_argument("--time-limit", type=float, default=300.0)
    suite.add_argument("--node-limit", type=int)
    suite.add_argument(
        "--output-dir", type=Path, default=Path("experiments/boundary_swap_suite_v1")
    )
    suite.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/causal_boundary_swap_train_suite_v1.json"),
    )

    oracle = subparsers.add_parser(
        "action-oracle-suite",
        help="run the predeclared root-action library and its post-hoc oracle",
    )
    oracle.add_argument("instances", type=Path, nargs="+")
    oracle.add_argument(
        "--actions",
        choices=TREATMENT_ARMS,
        nargs="+",
        default=list(LEGACY_TREATMENT_ARMS),
    )
    oracle.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    oracle.add_argument("--time-limit", type=float, default=300.0)
    oracle.add_argument("--node-limit", type=int)
    oracle.add_argument(
        "--intervention-scope",
        choices=INTERVENTION_SCOPES,
        default="per-run",
    )
    oracle.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse matching raw arm JSON files instead of rerunning them",
    )
    oracle.add_argument("--learned-model-manifest", type=Path)
    oracle.add_argument("--jobs", type=int, default=1)
    oracle.add_argument("--execution-order-key")
    oracle.add_argument(
        "--output-dir", type=Path, default=Path("experiments/action_oracle_v1")
    )
    oracle.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/causal_action_oracle_v1.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.time_limit <= 0:
        raise SystemExit("--time-limit must be positive")
    if args.node_limit is not None and args.node_limit <= 0:
        raise SystemExit("--node-limit must be positive")
    if hasattr(args, "jobs") and args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    invalid_seed = (
        args.seed < 0
        if args.command == "solve"
        else any(seed < 0 for seed in args.seeds)
    )
    if invalid_seed:
        raise SystemExit("seeds must be nonnegative")

    if args.command == "solve":
        result = solve_arm(
            args.instance,
            args.arm,
            args.seed,
            args.time_limit,
            args.node_limit,
            args.intervention_scope,
            args.learned_model_manifest,
        )
        _write_json(args.output, result)
        return 0

    if args.command == "boundary-swap-pair":
        manifest = run_boundary_swap_experiment(
            args.instance,
            tuple(dict.fromkeys(args.seeds)),
            args.time_limit,
            args.node_limit,
            args.output_dir,
            args.manifest,
        )
        print(
            json.dumps(
                {"valid": manifest["valid"], "manifest": str(args.manifest)}
            )
        )
        return 0 if manifest["safe"] else 2

    if args.command == "boundary-swap-suite":
        manifest = run_boundary_swap_suite(
            args.instances,
            tuple(dict.fromkeys(args.seeds)),
            args.time_limit,
            args.node_limit,
            args.output_dir,
            args.manifest,
        )
        print(
            json.dumps(
                {"safe": manifest["safe"], "manifest": str(args.manifest)}
            )
        )
        return 0 if manifest["safe"] else 2

    if args.command == "parity-suite":
        manifest = run_structural_parity_suite(
            args.instances,
            args.candidate_arms,
            tuple(dict.fromkeys(args.seeds)),
            args.time_limit,
            args.node_limit,
            args.output_dir,
            args.manifest,
            args.reuse_existing,
            args.jobs,
        )
        print(
            json.dumps(
                {
                    "passed": manifest["passed"],
                    "arm_summary": manifest["arm_summary"],
                    "manifest": str(args.manifest),
                }
            )
        )
        return 0 if manifest["passed"] else 2

    if args.command == "action-oracle-suite":
        manifest = run_action_oracle_suite(
            args.instances,
            args.actions,
            tuple(dict.fromkeys(args.seeds)),
            args.time_limit,
            args.node_limit,
            args.output_dir,
            args.manifest,
            args.reuse_existing,
            args.intervention_scope,
            args.learned_model_manifest,
            args.jobs,
            args.execution_order_key,
            None,
        )
        print(
            json.dumps(
                {
                    "all_actions_safe": manifest["all_actions_safe"],
                    "oracle_summary": manifest["oracle_summary"],
                    "manifest": str(args.manifest),
                }
            )
        )
        return 0

    candidate_arm = "noop" if args.command == "noop-parity" else "direct-hybrid"
    manifest = run_structural_parity(
        args.instance,
        candidate_arm,
        tuple(dict.fromkeys(args.seeds)),
        args.time_limit,
        args.node_limit,
        args.output_dir,
        args.manifest,
    )
    print(json.dumps({"passed": manifest["passed"], "manifest": str(args.manifest)}))
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
