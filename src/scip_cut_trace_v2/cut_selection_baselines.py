"""Deterministic fixed baselines for root cut-ranking experiments."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class AdaptiveScoreWeights:
    """Weights used by the Turner et al. cut-scoring family."""

    directed_cutoff_distance: float = 0.0
    efficacy: float = 1.0
    integer_support: float = 0.1
    objective_parallelism: float = 0.1


@dataclass(frozen=True)
class BaselineSelection:
    """A complete candidate ordering and its selected prefix."""

    cuts: list[Any]
    nselectedcuts: int
    scores: tuple[float, ...] | None
    metadata: dict[str, Any]


def _score(value: float) -> float:
    value = float(value)
    return -math.inf if math.isnan(value) else value


def _stable_descending_indices(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-_score(scores[index]), index))


def _validate_budget(cuts: Sequence[Any], nselectedcuts: int) -> None:
    if not 0 <= nselectedcuts <= len(cuts):
        raise ValueError(
            f"selected-cut budget {nselectedcuts} is outside [0, {len(cuts)}]"
        )


def deterministic_random_rank(
    cuts: Sequence[Any], nselectedcuts: int, key: str
) -> BaselineSelection:
    """Hash-permute candidates without process-global random state."""
    _validate_budget(cuts, nselectedcuts)

    def rank_key(index: int) -> tuple[bytes, int]:
        row_name = str(getattr(cuts[index], "name", ""))
        payload = f"{key}\0{index}\0{row_name}".encode("utf-8")
        return hashlib.sha256(payload).digest(), index

    order = sorted(range(len(cuts)), key=rank_key)
    return BaselineSelection(
        cuts=[cuts[index] for index in order],
        nselectedcuts=nselectedcuts,
        scores=None,
        metadata={"ranking": "sha256-permutation", "random_key_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest()},
    )


def efficacy_rank(model: Any, cuts: Sequence[Any], nselectedcuts: int) -> BaselineSelection:
    """Rank all candidates by SCIP efficacy with stable native-order ties."""
    _validate_budget(cuts, nselectedcuts)
    scores = tuple(float(model.getCutEfficacy(row)) for row in cuts)
    order = _stable_descending_indices(scores)
    return BaselineSelection(
        cuts=[cuts[index] for index in order],
        nselectedcuts=nselectedcuts,
        scores=scores,
        metadata={"ranking": "descending-cut-efficacy", "tie_break": "native-order"},
    )


def _normalized_log_square(value: float, maximum: float) -> float:
    value = float(value)
    maximum = float(maximum)
    if not math.isfinite(value) or not math.isfinite(maximum):
        return 0.0
    if value <= 0.0 or maximum <= 0.0:
        return 0.0
    return (math.log1p(value) / math.log1p(maximum)) ** 2


def adaptive_scores(
    model: Any,
    cuts: Sequence[Any],
    weights: AdaptiveScoreWeights = AdaptiveScoreWeights(),
) -> tuple[tuple[float, ...], tuple[dict[str, float], ...], bool]:
    """Compute the independently ported Turner et al. normalized score."""
    if not cuts:
        return (), (), False

    raw_efficacies = tuple(float(model.getCutEfficacy(row)) for row in cuts)
    maximum_efficacy = max(raw_efficacies)
    has_incumbent = int(model.getNSols()) > 0
    incumbent = model.getBestSol() if has_incumbent else None
    raw_cutoff_distances = (
        tuple(
            float(model.getCutLPSolCutoffDistance(row, incumbent)) for row in cuts
        )
        if has_incumbent
        else tuple(0.0 for _ in cuts)
    )
    maximum_cutoff_distance = (
        max(raw_cutoff_distances) if has_incumbent else maximum_efficacy
    )

    scores = []
    components = []
    for index, row in enumerate(cuts):
        nonzeros = int(row.getNNonz())
        integer_support = (
            float(model.getRowNumIntCols(row)) / nonzeros if nonzeros > 0 else 0.0
        )
        objective_parallelism = float(model.getRowObjParallelism(row))
        normalized_efficacy = _normalized_log_square(
            raw_efficacies[index], maximum_efficacy
        )
        normalized_cutoff_distance = (
            _normalized_log_square(
                raw_cutoff_distances[index], maximum_cutoff_distance
            )
            if has_incumbent
            else 0.0
        )
        efficacy_weight = weights.efficacy + (
            0.0 if has_incumbent else weights.directed_cutoff_distance
        )
        component = {
            "efficacy": efficacy_weight * normalized_efficacy,
            "directed_cutoff_distance": (
                weights.directed_cutoff_distance * normalized_cutoff_distance
            ),
            "integer_support": weights.integer_support * integer_support,
            "objective_parallelism": (
                weights.objective_parallelism * objective_parallelism
            ),
            "global_cutpool_bonus": (
                1e-4
                if hasattr(row, "isInGlobalCutpool") and row.isInGlobalCutpool()
                else 0.0
            ),
        }
        scores.append(sum(component.values()))
        components.append(component)
    return tuple(scores), tuple(components), has_incumbent


def adaptive_score_rank(
    model: Any,
    cuts: Sequence[Any],
    forcedcuts: Sequence[Any],
    nselectedcuts: int,
    root: bool,
    weights: AdaptiveScoreWeights = AdaptiveScoreWeights(),
    min_orthogonality_root: float = 0.9,
    min_orthogonality: float = 0.9,
) -> BaselineSelection:
    """Apply the Turner scoring and parallelism filter at a native cut budget.

    This intentionally preserves SCIP native hybrid's selected count. It is a
    causal scoring-family baseline, not the paper's full fixed-ten-cuts GCNN
    protocol.
    """
    _validate_budget(cuts, nselectedcuts)
    for name, value in (
        ("min_orthogonality_root", min_orthogonality_root),
        ("min_orthogonality", min_orthogonality),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    scores, _, has_incumbent = adaptive_scores(model, cuts, weights)
    if not cuts or nselectedcuts == 0:
        return BaselineSelection(
            cuts=list(cuts),
            nselectedcuts=nselectedcuts,
            scores=scores,
            metadata={
                "ranking": "turner-2023-adaptive-score-port",
                "has_incumbent": has_incumbent,
                "weights": weights.__dict__,
                "parallelism_filtered": 0,
            },
        )

    max_parallelism = 1.0 - (
        min_orthogonality_root if root else min_orthogonality
    )
    good_max_parallelism = max(0.5, max_parallelism)
    good_score = max(scores)
    viable = list(range(len(cuts)))
    filtered: set[int] = set()

    def filter_against(reference: Any, indices: Sequence[int]) -> list[int]:
        retained = []
        for index in indices:
            allowed = (
                good_max_parallelism
                if _score(scores[index]) >= _score(good_score)
                else max_parallelism
            )
            if float(model.getRowParallelism(reference, cuts[index])) > allowed:
                filtered.add(index)
            else:
                retained.append(index)
        return retained

    for forcedcut in forcedcuts:
        viable = filter_against(forcedcut, viable)

    selected = []
    while viable and len(selected) < nselectedcuts:
        best = min(viable, key=lambda index: (-_score(scores[index]), index))
        viable.remove(best)
        selected.append(best)
        viable = filter_against(cuts[best], viable)

    if len(selected) < nselectedcuts:
        remaining = [index for index in range(len(cuts)) if index not in selected]
        remaining.sort(key=lambda index: (-_score(scores[index]), index))
        selected.extend(remaining[: nselectedcuts - len(selected)])

    selected_set = set(selected)
    unselected = [index for index in range(len(cuts)) if index not in selected_set]
    order = selected + unselected
    return BaselineSelection(
        cuts=[cuts[index] for index in order],
        nselectedcuts=nselectedcuts,
        scores=scores,
        metadata={
            "ranking": "turner-2023-adaptive-score-port",
            "has_incumbent": has_incumbent,
            "weights": weights.__dict__,
            "min_orthogonality_root": min_orthogonality_root,
            "min_orthogonality": min_orthogonality,
            "parallelism_filtered": len(filtered),
            "tie_break": "native-order",
        },
    )
