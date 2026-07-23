"""Narrow bridge to SCIP 10.0.2's built-in hybrid cut selector."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Sequence

from pyscipopt import SCIP_RESULT

from ._scip_pointer import row_pointer, select_hybrid_pointers


SUPPORTED_SCIP_VERSION = (10, 0, 2)


@dataclass(frozen=True)
class HybridSelection:
    cuts: list[Any]
    nselectedcuts: int
    result: int


def _validate_version(model: Any) -> None:
    version = (
        int(model.getMajorVersion()),
        int(model.getMinorVersion()),
        int(model.getTechVersion()),
    )
    if version != SUPPORTED_SCIP_VERSION:
        raise RuntimeError(
            "The native-hybrid bridge is ABI-guarded for SCIP 10.0.2; "
            f"found {version[0]}.{version[1]}.{version[2]}"
        )


def select_cuts_hybrid(
    model: Any,
    cuts: Sequence[Any],
    forcedcuts: Sequence[Any],
    root: bool,
    maxnselectedcuts: int,
) -> HybridSelection:
    """Call SCIP's public hybrid algorithm with its initialized plugin data."""
    _validate_version(model)
    cut_pointers = [int(row_pointer(row)) for row in cuts]

    sorted_pointers, nselectedcuts = select_hybrid_pointers(
        model, list(cuts), list(forcedcuts), bool(root), int(maxnselectedcuts)
    )
    rows_by_pointer = defaultdict(deque)
    for pointer, row in zip(cut_pointers, cuts):
        rows_by_pointer[pointer].append(row)
    try:
        sorted_cuts = [rows_by_pointer[int(pointer)].popleft() for pointer in sorted_pointers]
    except (KeyError, IndexError) as error:
        raise RuntimeError("Native hybrid changed the candidate row set") from error
    if any(rows for rows in rows_by_pointer.values()):
        raise RuntimeError("Native hybrid changed candidate pointer multiplicities")
    if len(sorted_cuts) != len(cuts):
        raise RuntimeError("Native hybrid changed the candidate row count")
    if not 0 <= nselectedcuts <= min(len(cuts), maxnselectedcuts):
        raise RuntimeError(f"Native hybrid selected invalid cut count {nselectedcuts}")
    return HybridSelection(sorted_cuts, nselectedcuts, int(SCIP_RESULT.SUCCESS))
