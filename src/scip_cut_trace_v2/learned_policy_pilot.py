"""Freeze, run, and evaluate the one-shot learned-policy pilot."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .causal_harness import COMPLETE_STATUSES, run_action_oracle_suite
from .observational import PROJECT_ROOT, _instance_stem
from .paper_statistics import cluster_bootstrap_geometric_ratio


SCHEMA_VERSION = 1
PILOT_INSTANCE_COUNT = 18
SEEDS = (0, 1, 2)
SHADOW_ARM = "xgb-imitation-shadow"
ACTIVE_ARM = "xgb-imitation-rank"
ARMS = ("native", SHADOW_ARM, ACTIVE_ARM)
PLAN_KEY = "learned-policy-pilot-v1-20260721"

DEFAULT_COLLECTION_METADATA = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "miplib2017_collection_metadata_2026-07-21.csv"
)
DEFAULT_DEVELOPMENT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "miplib2017_benchmark_metadata_2026-07-16.csv"
)
DEFAULT_SPLITS = tuple(
    PROJECT_ROOT / "vendor" / "tracer_snapshot" / "split" / f"{name}.test"
    for name in ("train", "val", "test")
)
DEFAULT_INSTANCES_DIR = PROJECT_ROOT / "data" / "raw" / "miplib2017_collection"
DEFAULT_MODEL_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "ranking_imitation_online_xgb_revision_v1.json"
)
DEFAULT_PLAN = (
    PROJECT_ROOT / "data" / "manifests" / "learned_policy_pilot_plan_v1.json"
)
DEFAULT_DOWNLOAD_PLAN = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "learned_policy_pilot_download_plan_v1.json"
)
DEFAULT_DOWNLOAD_RESULT = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "learned_policy_pilot_downloads_v1.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "learned_policy_pilot_v1"
DEFAULT_RESULT = (
    PROJECT_ROOT / "data" / "manifests" / "causal_learned_policy_pilot_v1.json"
)
DEFAULT_STATISTICS = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "learned_policy_pilot_statistics_v1.json"
)


@dataclass(frozen=True)
class PilotCandidate:
    instance_id: str
    instance_name: str
    instance: Path
    status: str
    official_group: str
    tags: tuple[str, ...]

    @property
    def group_key(self) -> str:
        return self.official_group or f"officially-ungrouped:{self.instance_id}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _resolve_destination(reference: str) -> Path:
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _gzip_is_valid(path: Path) -> bool:
    if path.suffix != ".gz":
        return True
    try:
        with gzip.open(path, "rb") as handle:
            for _ in iter(lambda: handle.read(1024 * 1024), b""):
                pass
    except (EOFError, OSError):
        return False
    return True


_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def _parse_content_range(value: str) -> tuple[int, int, int]:
    match = _CONTENT_RANGE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Invalid Content-Range header: {value!r}")
    return tuple(int(part) for part in match.groups())


def _open_http(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout)


def _remote_file_identity(
    url: str,
    timeout: float,
    open_http: Callable[..., Any] = _open_http,
) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD")
    with open_http(request, timeout) as response:
        size = int(response.headers["Content-Length"])
        if size <= 0:
            raise ValueError(f"Remote file has invalid Content-Length: {url}")
        return {
            "url": url,
            "content_length": size,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }


def _download_one_instance(
    url: str,
    destination: Path,
    chunk_bytes: int,
    retries: int,
    timeout: float,
    open_http: Callable[..., Any] = _open_http,
) -> dict[str, Any]:
    """Download one instance through verified byte ranges and atomically install it."""
    if chunk_bytes <= 0 or retries <= 0:
        raise ValueError("chunk_bytes and retries must be positive")
    identity = _remote_file_identity(url, timeout, open_http)
    expected_size = int(identity["content_length"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if (
        destination.is_file()
        and destination.stat().st_size == expected_size
        and _gzip_is_valid(destination)
    ):
        return {
            **identity,
            "destination": _manifest_path(destination),
            "sha256": _sha256_file(destination),
            "reused": True,
        }

    partial = destination.with_name(f".{destination.name}.range-part")
    state_path = destination.with_name(f".{destination.name}.range-state.json")
    expected_state = {
        "schema_version": 1,
        **identity,
        "chunk_bytes": chunk_bytes,
    }
    completed = 0
    if partial.is_file() and state_path.is_file():
        state = _load(state_path)
        state_identity = {
            key: state.get(key)
            for key in ("schema_version", "url", "content_length", "etag", "last_modified", "chunk_bytes")
        }
        if state_identity == expected_state:
            completed = int(state.get("completed_bytes", 0))
            if not 0 <= completed <= expected_size:
                completed = 0
    mode = "r+b" if partial.is_file() else "w+b"
    with partial.open(mode) as handle:
        handle.truncate(completed)
    _write_json_atomic(state_path, {**expected_state, "completed_bytes": completed})

    while completed < expected_size:
        end = min(completed + chunk_bytes, expected_size) - 1
        headers = {"Range": f"bytes={completed}-{end}", "Accept-Encoding": "identity"}
        if identity["etag"]:
            headers["If-Range"] = identity["etag"]
        error: Exception | None = None
        for attempt in range(retries):
            try:
                request = urllib.request.Request(url, headers=headers)
                with open_http(request, timeout) as response:
                    status = getattr(response, "status", response.getcode())
                    content_range = _parse_content_range(
                        response.headers.get("Content-Range", "")
                    )
                    data = response.read()
                if status != 206:
                    raise ValueError(f"Range request returned HTTP {status}")
                if content_range != (completed, end, expected_size):
                    raise ValueError(
                        f"Unexpected Content-Range {content_range}; "
                        f"expected {(completed, end, expected_size)}"
                    )
                if len(data) != end - completed + 1:
                    raise ValueError(
                        f"Range returned {len(data)} bytes; expected {end - completed + 1}"
                    )
                with partial.open("r+b") as handle:
                    handle.seek(completed)
                    handle.write(data)
                    handle.truncate(end + 1)
                    handle.flush()
                    os.fsync(handle.fileno())
                completed = end + 1
                _write_json_atomic(
                    state_path, {**expected_state, "completed_bytes": completed}
                )
                error = None
                break
            except (OSError, ValueError, urllib.error.URLError) as caught:
                error = caught
                if attempt + 1 < retries:
                    time.sleep(min(2**attempt, 8))
        if error is not None:
            raise RuntimeError(
                f"Unable to download byte range {completed}-{end} from {url}"
            ) from error

    if partial.stat().st_size != expected_size:
        raise ValueError(f"Downloaded size does not match Content-Length: {url}")
    if not _gzip_is_valid(partial):
        raise ValueError(f"Downloaded gzip stream is invalid: {url}")
    digest = _sha256_file(partial)
    os.replace(partial, destination)
    state_path.unlink(missing_ok=True)
    return {
        **identity,
        "destination": _manifest_path(destination),
        "sha256": digest,
        "reused": False,
    }


def download_plan_instances(
    plan_path: Path,
    chunk_bytes: int = 128 * 1024,
    retries: int = 5,
    timeout: float = 60.0,
) -> dict[str, Any]:
    plan = _load(plan_path)
    downloads = []
    for index, record in enumerate(plan["instances"], start=1):
        print(
            f"[{index}/{len(plan['instances'])}] {record['instance_id']}",
            flush=True,
        )
        downloaded = _download_one_instance(
            record["url"],
            _resolve_destination(record["destination"]),
            chunk_bytes,
            retries,
            timeout,
        )
        downloads.append({"instance_id": record["instance_id"], **downloaded})
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "all frozen pilot instances downloaded and verified",
        "source_plan": _manifest_path(plan_path),
        "source_plan_sha256": _sha256_file(plan_path),
        "range_chunk_bytes": chunk_bytes,
        "downloads": downloads,
        "checks": {
            "all_planned_instances_present": len(downloads) == len(plan["instances"]),
            "all_sizes_match": all(
                _resolve_destination(item["destination"]).stat().st_size
                == item["content_length"]
                for item in downloads
            ),
            "all_gzip_streams_valid": all(
                _gzip_is_valid(_resolve_destination(item["destination"]))
                for item in downloads
            ),
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"instance_name", "status", "group", "tags"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    return rows


def _read_split_names(paths: Iterable[Path]) -> set[str]:
    names = set()
    for path in paths:
        names.update(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
    return names


def _metadata_by_id(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    result = {_instance_stem(row["instance_name"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Metadata contains duplicate instance IDs: {path}")
    return result


def _resolve_instance(instances_dir: Path, raw_name: str) -> Path | None:
    candidates = [instances_dir / raw_name]
    instance_id = _instance_stem(raw_name)
    candidates.extend(
        (instances_dir / f"{instance_id}.mps.gz", instances_dir / f"{instance_id}.mps")
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def discover_candidates(
    collection_metadata_path: Path,
    development_metadata_path: Path,
    split_paths: Iterable[Path],
    instances_dir: Path,
    require_local: bool = True,
) -> tuple[list[PilotCandidate], set[str]]:
    """Return easy, local Collection instances disjoint from all development Groups."""
    split_paths = tuple(split_paths)
    development_names = _read_split_names(split_paths)
    development_by_id = _metadata_by_id(development_metadata_path)
    missing = sorted(
        name
        for name in development_names
        if _instance_stem(name) not in development_by_id
    )
    if missing:
        raise ValueError(f"Development instances lack Group metadata: {missing[:5]}")

    development_group_keys = {
        row["group"].strip()
        or f"officially-ungrouped:{_instance_stem(name)}"
        for name in development_names
        for row in (development_by_id[_instance_stem(name)],)
    }
    development_ids = {_instance_stem(name) for name in development_names}

    candidates = []
    for row in _read_csv(collection_metadata_path):
        instance_id = _instance_stem(row["instance_name"])
        official_group = row["group"].strip()
        group_key = official_group or f"officially-ungrouped:{instance_id}"
        instance_path = _resolve_instance(instances_dir, row["instance_name"])
        if (
            row["status"].strip().lower() != "easy"
            or "infeasible" in row["tags"].split()
            or instance_id in development_ids
            or group_key in development_group_keys
            or (require_local and instance_path is None)
        ):
            continue
        if instance_path is None:
            instance_path = (instances_dir / row["instance_name"]).resolve()
        candidates.append(
            PilotCandidate(
                instance_id=instance_id,
                instance_name=instance_path.name,
                instance=instance_path,
                status="easy",
                official_group=official_group,
                tags=tuple(sorted(set(row["tags"].split()))),
            )
        )
    return candidates, development_group_keys


def build_download_plan(
    collection_metadata_path: Path,
    development_metadata_path: Path,
    split_paths: Iterable[Path],
    instances_dir: Path,
) -> dict[str, Any]:
    split_paths = tuple(path.resolve() for path in split_paths)
    candidates, development_groups = discover_candidates(
        collection_metadata_path,
        development_metadata_path,
        split_paths,
        instances_dir,
        require_local=False,
    )
    selected = select_pilot_candidates(candidates)
    records = [
        {
            "instance_id": item.instance_id,
            "instance_name": item.instance_name,
            "group_key": item.group_key,
            "official_group": item.official_group or None,
            "url": f"https://miplib.zib.de/WebData/instances/{item.instance_name}",
            "destination": _manifest_path(item.instance),
        }
        for item in selected
    ]
    checks = {
        "eighteen_instances": len(records) == PILOT_INSTANCE_COUNT,
        "eighteen_distinct_group_keys": len({item["group_key"] for item in records})
        == PILOT_INSTANCE_COUNT,
        "all_groups_disjoint_from_existing_233": not (
            {item["group_key"] for item in records} & development_groups
        ),
        "existing_validation_and_test_not_selected": True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Learned-policy download plan checks failed: {failed}")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen before downloading new pilot instances",
        "purpose": "content-blind acquisition list for the one-shot learned-policy pilot",
        "selection_key": PLAN_KEY,
        "selection_rule": (
            "same frozen SHA-256 Group-disjoint rule used by the online pilot plan; "
            "no solver outcome or local file property participates"
        ),
        "source_artifacts": {
            "collection_metadata": {
                "path": _manifest_path(collection_metadata_path),
                "sha256": _sha256_file(collection_metadata_path),
            },
            "development_metadata": {
                "path": _manifest_path(development_metadata_path),
                "sha256": _sha256_file(development_metadata_path),
            },
            "development_splits": [
                {"path": _manifest_path(path), "sha256": _sha256_file(path)}
                for path in split_paths
            ],
        },
        "instances_dir": _manifest_path(instances_dir),
        "eligible_candidate_instances": len(candidates),
        "instances": records,
        "checks": checks,
    }


def select_pilot_candidates(
    candidates: Iterable[PilotCandidate],
    count: int = PILOT_INSTANCE_COUNT,
    key: str = PLAN_KEY,
) -> list[PilotCandidate]:
    """Hash-order candidates and retain at most one instance per Group key."""
    ordered = sorted(
        candidates,
        key=lambda item: hashlib.sha256(
            f"{key}\0{item.group_key}\0{item.instance_id}".encode("utf-8")
        ).digest(),
    )
    selected = []
    used_groups: set[str] = set()
    for candidate in ordered:
        if candidate.group_key in used_groups:
            continue
        selected.append(candidate)
        used_groups.add(candidate.group_key)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} distinct unseen Group keys are available; {count} required")


def build_balanced_schedule(
    instances: Iterable[dict[str, Any]], seeds: Iterable[int], key: str = PLAN_KEY
) -> list[dict[str, Any]]:
    """Assign all six three-arm orders equally across hash-ordered blocks."""
    sequences = tuple(itertools.permutations(ARMS))
    blocks = [
        (instance["instance_id"], int(seed))
        for instance in instances
        for seed in seeds
    ]
    if len(blocks) % len(sequences):
        raise ValueError("Instance-seed blocks must be divisible by the six arm orders")
    blocks.sort(
        key=lambda block: hashlib.sha256(
            f"{key}\0{block[0]}\0{block[1]}".encode("utf-8")
        ).digest()
    )
    return [
        {
            "instance_id": instance_id,
            "seed": seed,
            "design_row": index % len(sequences),
            "arm_order": list(sequences[index % len(sequences)]),
        }
        for index, (instance_id, seed) in enumerate(blocks)
    ]


def _schedule_position_counts(
    schedule: Iterable[dict[str, Any]],
) -> dict[str, dict[int, int]]:
    counts: dict[str, dict[int, int]] = {}
    for block in schedule:
        for position, arm in enumerate(block["arm_order"]):
            counts.setdefault(arm, {}).setdefault(position, 0)
            counts[arm][position] += 1
    return counts


def _resolve_model_path(model_manifest_path: Path, reference: str) -> Path:
    raw = Path(reference)
    candidates = (
        (raw,) if raw.is_absolute() else (PROJECT_ROOT / raw, model_manifest_path.parent / raw)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Frozen model file is missing: {reference}")


def build_plan(
    collection_metadata_path: Path,
    development_metadata_path: Path,
    split_paths: Iterable[Path],
    instances_dir: Path,
    model_manifest_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    result_path: Path = DEFAULT_RESULT,
    statistics_path: Path = DEFAULT_STATISTICS,
    download_plan_path: Path | None = None,
    download_result_path: Path | None = None,
) -> dict[str, Any]:
    split_paths = tuple(path.resolve() for path in split_paths)
    candidates, development_groups = discover_candidates(
        collection_metadata_path,
        development_metadata_path,
        split_paths,
        instances_dir,
        require_local=False,
    )
    selected = select_pilot_candidates(candidates)
    selected_ids = [item.instance_id for item in selected]
    download_plan = _load(download_plan_path) if download_plan_path else None
    download_plan_ids = (
        [item["instance_id"] for item in download_plan["instances"]]
        if download_plan
        else selected_ids
    )
    if download_plan_ids != selected_ids:
        raise ValueError("Frozen download-plan selection does not match the pilot rule")
    download_result = _load(download_result_path) if download_result_path else None
    download_records = (
        {item["instance_id"]: item for item in download_result["downloads"]}
        if download_result
        else {}
    )
    if download_result and list(download_records) != selected_ids:
        raise ValueError("Verified download records do not match the frozen pilot selection")
    model_manifest = _load(model_manifest_path)
    model_path = _resolve_model_path(model_manifest_path, model_manifest["model"]["path"])
    if _sha256_file(model_path) != model_manifest["model"]["sha256"]:
        raise ValueError("Frozen model hash does not match its manifest")

    instances = [
        {
            "instance_id": item.instance_id,
            "instance_name": item.instance_name,
            "instance": _manifest_path(item.instance),
            "instance_sha256": _sha256_file(item.instance),
            "status": item.status,
            "official_group": item.official_group or None,
            "group_key": item.group_key,
            "tags": list(item.tags),
        }
        for item in selected
    ]
    downloads_match_files = all(
        not download_result
        or (
            item["instance_id"] in download_records
            and _resolve_destination(item["instance"]).stat().st_size
            == download_records[item["instance_id"]]["content_length"]
            and item["instance_sha256"]
            == download_records[item["instance_id"]]["sha256"]
        )
        for item in instances
    )
    schedule = build_balanced_schedule(instances, SEEDS)
    position_counts = _schedule_position_counts(schedule)
    checks = {
        "eighteen_instances": len(instances) == PILOT_INSTANCE_COUNT,
        "eighteen_distinct_group_keys": len({item["group_key"] for item in instances})
        == PILOT_INSTANCE_COUNT,
        "all_groups_disjoint_from_existing_233": not (
            {item["group_key"] for item in instances} & development_groups
        ),
        "all_instances_easy_and_local": all(
            item["status"] == "easy"
            and "infeasible" not in item["tags"]
            and _resolve_destination(item["instance"]).is_file()
            and _gzip_is_valid(_resolve_destination(item["instance"]))
            for item in instances
        ),
        "three_paired_seeds": list(SEEDS) == [0, 1, 2],
        "fifty_four_blocks": len(schedule) == PILOT_INSTANCE_COUNT * len(SEEDS),
        "arm_positions_balanced": all(
            count == PILOT_INSTANCE_COUNT
            for arm_counts in position_counts.values()
            for count in arm_counts.values()
        ),
        "existing_validation_and_test_not_selected": True,
        "selection_matches_frozen_download_plan": download_plan_ids == selected_ids,
        "verified_downloads_match_frozen_instances": downloads_match_files,
        "frozen_model_offline_gate_recorded_as_failed": not model_manifest[
            "stage_gate"
        ]["passed"],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Learned-policy pilot plan checks failed: {failed}")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pre_registered_before_any_new_group_online_outcomes",
        "purpose": (
            "one-shot exploratory Go/No-Go test of the already frozen learned ranker; "
            "failure stops learned-policy development rather than starting another model iteration"
        ),
        "data_boundary": {
            "existing_233_instance_splits": "untouched; existing validation and test remain sealed",
            "pilot_source": "MIPLIB 2017 Collection instances outside all existing split Group keys",
            "independent_unit": "official MIPLIB Group key; each officially ungrouped instance is its own key",
            "outcome_blinding": (
                "selection uses only easy status, official Group, instance identity, and "
                "a frozen SHA-256 order"
            ),
        },
        "source_artifacts": {
            "collection_metadata": {
                "path": _manifest_path(collection_metadata_path),
                "sha256": _sha256_file(collection_metadata_path),
            },
            "development_metadata": {
                "path": _manifest_path(development_metadata_path),
                "sha256": _sha256_file(development_metadata_path),
            },
            "development_splits": [
                {"path": _manifest_path(path), "sha256": _sha256_file(path)}
                for path in split_paths
            ],
            "learned_model_manifest": {
                "path": _manifest_path(model_manifest_path),
                "sha256": _sha256_file(model_manifest_path),
                "offline_stage_gate_passed": model_manifest["stage_gate"]["passed"],
                "model_path": _manifest_path(model_path),
                "model_sha256": _sha256_file(model_path),
            },
            "download_plan": (
                {
                    "path": _manifest_path(download_plan_path),
                    "sha256": _sha256_file(download_plan_path),
                }
                if download_plan_path
                else None
            ),
            "download_result": (
                {
                    "path": _manifest_path(download_result_path),
                    "sha256": _sha256_file(download_result_path),
                }
                if download_result_path
                else None
            ),
        },
        "selection_contract": {
            "candidate_pool_size": len(candidates),
            "selection_key": PLAN_KEY,
            "rule": (
                "SHA-256 order by frozen key, Group key, and instance ID; retain the first "
                "18 distinct Group keys without examining solver outcomes"
            ),
        },
        "experiment": {
            "instances": instances,
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "time_limit_seconds": 300.0,
            "node_limit": None,
            "intervention_scope": "first-run-only",
            "max_workers": 1,
            "resource_isolation": "one fresh single-threaded SCIP process at a time",
            "execution_order_key": PLAN_KEY,
            "execution_schedule": schedule,
            "output_dir": _manifest_path(output_dir),
            "result_manifest": _manifest_path(result_path),
            "statistics_manifest": _manifest_path(statistics_path),
        },
        "analysis_contract": {
            "primary_population": "all 54 fallback-inclusive instance-seed blocks",
            "primary_estimand": (
                "Group-equal geometric mean active/native full-process wall-time PAR-2 ratio; "
                "seeds are averaged on the log scale within Group"
            ),
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260721,
            "go_no_go_thresholds": {
                "minimum_group_keys_with_active_intervention": 12,
                "maximum_active_correctness_failure_group_keys": 0,
                "maximum_native_complete_active_incomplete_group_keys": 0,
                "required_shadow_structural_matches": 54,
                "maximum_primary_point_ratio": 0.95,
                "maximum_one_sided_80_percent_upper_ratio": 1.0,
            },
            "interpretation": (
                "a pass authorizes one separately frozen 78-Group confirmation study; "
                "it is not confirmatory evidence by itself"
            ),
            "failure_action": (
                "stop learned-policy development, do not tune on pilot outcomes, keep the "
                "existing test sealed, and submit the audited study to the planned journal"
            ),
        },
        "checks": checks,
    }


def _verify_plan(plan_path: Path) -> dict[str, Any]:
    plan = _load(plan_path)
    if not all(plan["checks"].values()):
        raise ValueError("Frozen learned-policy plan contains failed checks")
    for name in ("collection_metadata", "development_metadata", "learned_model_manifest"):
        record = plan["source_artifacts"][name]
        if _sha256_file(_resolve_destination(record["path"])) != record["sha256"]:
            raise ValueError(f"Frozen source changed after plan creation: {name}")
    model = plan["source_artifacts"]["learned_model_manifest"]
    if _sha256_file(_resolve_destination(model["model_path"])) != model["model_sha256"]:
        raise ValueError("Frozen learned model changed after plan creation")
    for record in plan["source_artifacts"]["development_splits"]:
        if _sha256_file(_resolve_destination(record["path"])) != record["sha256"]:
            raise ValueError("A frozen development split changed after plan creation")
    for name in ("download_plan", "download_result"):
        record = plan["source_artifacts"].get(name)
        if record and _sha256_file(_resolve_destination(record["path"])) != record["sha256"]:
            raise ValueError(f"Frozen source changed after plan creation: {name}")
    for instance in plan["experiment"]["instances"]:
        if (
            _sha256_file(_resolve_destination(instance["instance"]))
            != instance["instance_sha256"]
        ):
            raise ValueError(f"Frozen instance changed: {instance['instance_id']}")
    return plan


def run_plan(plan_path: Path, reuse_existing: bool = False) -> dict[str, Any]:
    plan = _verify_plan(plan_path)
    experiment = plan["experiment"]
    schedule = {
        (block["instance_id"], int(block["seed"])): tuple(block["arm_order"])
        for block in experiment["execution_schedule"]
    }
    result = run_action_oracle_suite(
        [_resolve_destination(record["instance"]) for record in experiment["instances"]],
        [SHADOW_ARM, ACTIVE_ARM],
        experiment["seeds"],
        experiment["time_limit_seconds"],
        experiment["node_limit"],
        _resolve_destination(experiment["output_dir"]),
        _resolve_destination(experiment["result_manifest"]),
        reuse_existing,
        experiment["intervention_scope"],
        _resolve_destination(
            plan["source_artifacts"]["learned_model_manifest"]["path"]
        ),
        experiment["max_workers"],
        None,
        schedule,
    )
    result["learned_policy_pilot_plan"] = _manifest_path(plan_path)
    result["learned_policy_pilot_plan_sha256"] = _sha256_file(plan_path)
    _resolve_destination(experiment["result_manifest"]).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _penalized(outcome: dict[str, Any], field: str, time_limit: float) -> float:
    if outcome["status"] in COMPLETE_STATUSES:
        return max(float(outcome[field]), 1e-12)
    return 2.0 * time_limit


def _one_sided_bootstrap_upper(
    instance_log_ratios: Iterable[float], replicates: int, seed: int, confidence: float
) -> float | None:
    values = np.asarray(tuple(instance_log_ratios), dtype=np.float64)
    if values.size == 0:
        return None
    generator = np.random.default_rng(seed)
    sampled = generator.choice(values, size=(replicates, values.size), replace=True)
    return float(np.quantile(np.exp(sampled.mean(axis=1)), confidence))


def _mechanical_checks_pass(comparison: dict[str, Any]) -> bool:
    checks = comparison["safety_checks"]
    names = (
        "arm_order",
        "same_instance_sha256",
        "same_seed",
        "same_parameters",
        "same_runtime_versions",
        "same_objective_sense",
        "known_intervention_scope",
        "run_budget_respected",
        "one_record_per_intervention",
        "context_budget_respected",
        "interventions_have_context",
    )
    return all(checks.get(name, False) for name in names)


def analyze_result(
    result_path: Path,
    plan_path: Path,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    plan = _load(plan_path)
    result = _load(result_path)
    experiment = plan["experiment"]
    contract = plan["analysis_contract"]
    replicates = bootstrap_replicates or int(contract["bootstrap_replicates"])
    seed = int(contract["bootstrap_seed"])
    planned_ids = [record["instance_id"] for record in experiment["instances"]]
    observed_ids = [record["instance_id"] for record in result["per_instance"]]
    artifact_checks = {
        "plan_pre_registered": plan["status"]
        == "pre_registered_before_any_new_group_online_outcomes",
        "instances_match": observed_ids == planned_ids,
        "actions_match": result["actions"] == [SHADOW_ARM, ACTIVE_ARM],
        "seeds_match": result["seeds"] == experiment["seeds"],
        "time_limit_matches": float(result["time_limit"])
        == float(experiment["time_limit_seconds"]),
        "intervention_scope_matches": result["intervention_scope"]
        == experiment["intervention_scope"],
        "model_manifest_matches": result["learned_model_manifest_sha256"]
        == plan["source_artifacts"]["learned_model_manifest"]["sha256"],
    }
    if not all(artifact_checks.values()):
        failed = [name for name, passed in artifact_checks.items() if not passed]
        raise ValueError(f"Pilot result violates the frozen plan: {failed}")

    time_limit = float(experiment["time_limit_seconds"])
    primary_logs = []
    solving_time_logs = []
    shadow_matches = 0
    total_pairs = 0
    active_fallback_pairs = 0
    observed_context_pairs = 0
    context_mismatch_pairs = 0
    exposed_groups = 0
    correctness_failure_groups = 0
    catastrophic_loss_groups = 0
    per_group = []

    group_by_id = {
        record["instance_id"]: record["group_key"]
        for record in experiment["instances"]
    }
    for instance in result["per_instance"]:
        wall_logs = []
        solver_logs = []
        group_exposed = False
        group_correctness_failure = False
        group_catastrophic_loss = False
        group_shadow_matches = 0
        for pair in instance["pairs"]:
            total_pairs += 1
            native = pair["native_outcome"]
            shadow = pair["actions"][SHADOW_ARM]
            active = pair["actions"][ACTIVE_ARM]
            shadow_comparison = shadow["comparison"]
            active_comparison = active["comparison"]
            if shadow_comparison["safe"]:
                shadow_matches += 1
                group_shadow_matches += 1

            if pair["initial_context"]["all_actions_observed"]:
                observed_context_pairs += 1
                if not pair["initial_context"]["matching_across_actions"]:
                    context_mismatch_pairs += 1

            interventions = int(active["selector"]["interventions"])
            group_exposed = group_exposed or interventions > 0
            active_fallback_pairs += interventions == 0
            if not _mechanical_checks_pass(active_comparison):
                group_correctness_failure = True

            active_outcome = active["outcome"]
            native_complete = native["status"] in COMPLETE_STATUSES
            active_complete = active_outcome["status"] in COMPLETE_STATUSES
            if native_complete and active_complete:
                terminal_checks = active_comparison["safety_checks"]
                if not all(
                    terminal_checks[name]
                    for name in ("same_status", "same_primal_bound", "same_dual_bound")
                ):
                    group_correctness_failure = True
            if native_complete and not active_complete:
                group_catastrophic_loss = True

            wall_logs.append(
                math.log(
                    _penalized(active_outcome, "arm_wall_time_seconds", time_limit)
                    / _penalized(native, "arm_wall_time_seconds", time_limit)
                )
            )
            solver_logs.append(
                math.log(
                    _penalized(active_outcome, "solving_time", time_limit)
                    / _penalized(native, "solving_time", time_limit)
                )
            )

        group_wall_log = sum(wall_logs) / len(wall_logs)
        group_solver_log = sum(solver_logs) / len(solver_logs)
        primary_logs.append(group_wall_log)
        solving_time_logs.append(group_solver_log)
        exposed_groups += group_exposed
        correctness_failure_groups += group_correctness_failure
        catastrophic_loss_groups += group_catastrophic_loss
        per_group.append(
            {
                "instance_id": instance["instance_id"],
                "group_key": group_by_id[instance["instance_id"]],
                "active_intervention_observed": group_exposed,
                "active_correctness_failure": group_correctness_failure,
                "native_complete_active_incomplete": group_catastrophic_loss,
                "shadow_structural_matches": group_shadow_matches,
                "active_native_arm_wall_time_ratio": math.exp(group_wall_log),
                "active_native_solving_time_ratio": math.exp(group_solver_log),
            }
        )

    primary_ratio = math.exp(sum(primary_logs) / len(primary_logs))
    primary_interval = cluster_bootstrap_geometric_ratio(
        primary_logs, replicates, seed, confidence=0.95
    )
    one_sided_upper = _one_sided_bootstrap_upper(
        primary_logs, replicates, seed + 1, 0.80
    )
    solver_ratio = math.exp(sum(solving_time_logs) / len(solving_time_logs))
    thresholds = contract["go_no_go_thresholds"]
    gate_checks = {
        "all_planned_pairs_present": total_pairs
        == PILOT_INSTANCE_COUNT * len(SEEDS),
        "shadow_structural_matches_all_pairs": shadow_matches
        == thresholds["required_shadow_structural_matches"],
        "zero_observed_pre_action_context_mismatches": context_mismatch_pairs == 0,
        "enough_group_keys_receive_active_intervention": exposed_groups
        >= thresholds["minimum_group_keys_with_active_intervention"],
        "zero_active_correctness_failure_group_keys": correctness_failure_groups
        <= thresholds["maximum_active_correctness_failure_group_keys"],
        "zero_native_complete_active_incomplete_group_keys": catastrophic_loss_groups
        <= thresholds["maximum_native_complete_active_incomplete_group_keys"],
        "primary_point_ratio_at_most_0_95": primary_ratio
        <= thresholds["maximum_primary_point_ratio"],
        "one_sided_80_percent_upper_ratio_at_most_one": one_sided_upper is not None
        and one_sided_upper
        <= thresholds["maximum_one_sided_80_percent_upper_ratio"],
    }
    passed = all(gate_checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "one-shot learned-policy pilot decision",
        "source_plan": _manifest_path(plan_path),
        "source_plan_sha256": _sha256_file(plan_path),
        "source_result": _manifest_path(result_path),
        "source_result_sha256": _sha256_file(result_path),
        "artifact_contract_checks": artifact_checks,
        "population": {
            "group_keys": len(per_group),
            "pairs": total_pairs,
            "active_intervention_group_keys": exposed_groups,
            "active_fallback_pairs": active_fallback_pairs,
            "shadow_structural_matches": shadow_matches,
            "observed_context_pairs": observed_context_pairs,
            "context_mismatch_pairs": context_mismatch_pairs,
            "active_correctness_failure_group_keys": correctness_failure_groups,
            "native_complete_active_incomplete_group_keys": catastrophic_loss_groups,
        },
        "primary_full_process_wall_time": {
            "estimand": contract["primary_estimand"],
            "active_native_par2_ratio": primary_ratio,
            "cluster_bootstrap_interval_95": primary_interval,
            "one_sided_percentile_upper_80": {
                "replicates": replicates,
                "seed": seed + 1,
                "upper": one_sided_upper,
            },
        },
        "secondary_scip_solving_time": {
            "active_native_par2_ratio": solver_ratio,
            "cluster_bootstrap_interval_95": cluster_bootstrap_geometric_ratio(
                solving_time_logs, replicates, seed + 2, confidence=0.95
            ),
        },
        "per_group": per_group,
        "gate_checks": gate_checks,
        "passed": passed,
        "decision": (
            "freeze a separate 78-Group confirmatory learned-policy study using the unchanged model"
            if passed
            else "stop learned-policy development; do not tune on pilot outcomes or open the existing sealed test"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--collection-metadata", type=Path, default=DEFAULT_COLLECTION_METADATA)
    freeze.add_argument("--development-metadata", type=Path, default=DEFAULT_DEVELOPMENT_METADATA)
    freeze.add_argument("--splits", type=Path, nargs="+", default=list(DEFAULT_SPLITS))
    freeze.add_argument("--instances-dir", type=Path, default=DEFAULT_INSTANCES_DIR)
    freeze.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    freeze.add_argument("--download-plan", type=Path, default=DEFAULT_DOWNLOAD_PLAN)
    freeze.add_argument("--download-result", type=Path, default=DEFAULT_DOWNLOAD_RESULT)
    freeze.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    downloads = subparsers.add_parser("plan-downloads")
    downloads.add_argument("--collection-metadata", type=Path, default=DEFAULT_COLLECTION_METADATA)
    downloads.add_argument("--development-metadata", type=Path, default=DEFAULT_DEVELOPMENT_METADATA)
    downloads.add_argument("--splits", type=Path, nargs="+", default=list(DEFAULT_SPLITS))
    downloads.add_argument("--instances-dir", type=Path, default=DEFAULT_INSTANCES_DIR)
    downloads.add_argument("--output", type=Path, default=DEFAULT_DOWNLOAD_PLAN)
    download = subparsers.add_parser("download")
    download.add_argument("--plan", type=Path, default=DEFAULT_DOWNLOAD_PLAN)
    download.add_argument("--output", type=Path, default=DEFAULT_DOWNLOAD_RESULT)
    download.add_argument("--chunk-bytes", type=int, default=128 * 1024)
    download.add_argument("--retries", type=int, default=5)
    download.add_argument("--timeout", type=float, default=60.0)
    run = subparsers.add_parser("run")
    run.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    run.add_argument("--reuse-existing", action="store_true")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    analyze.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    analyze.add_argument("--output", type=Path, default=DEFAULT_STATISTICS)
    analyze.add_argument("--bootstrap-replicates", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan-downloads":
        if args.output.exists():
            raise SystemExit("Refusing to overwrite an existing download plan")
        plan = build_download_plan(
            args.collection_metadata.resolve(),
            args.development_metadata.resolve(),
            [path.resolve() for path in args.splits],
            args.instances_dir.resolve(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(plan["checks"], sort_keys=True))
        return 0
    if args.command == "download":
        if args.output.exists():
            raise SystemExit("Refusing to overwrite an existing download result")
        result = download_plan_instances(
            args.plan.resolve(), args.chunk_bytes, args.retries, args.timeout
        )
        if not all(result["checks"].values()):
            raise SystemExit("Downloaded instances did not pass all integrity checks")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.output, result)
        print(json.dumps(result["checks"], sort_keys=True))
        return 0
    if args.command == "freeze":
        result_path = DEFAULT_RESULT.resolve()
        if args.output.exists() or result_path.exists():
            raise SystemExit("Refusing to overwrite an existing pilot plan or result")
        plan = build_plan(
            args.collection_metadata.resolve(),
            args.development_metadata.resolve(),
            [path.resolve() for path in args.splits],
            args.instances_dir.resolve(),
            args.model_manifest.resolve(),
            download_plan_path=args.download_plan.resolve(),
            download_result_path=args.download_result.resolve(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(plan["checks"], sort_keys=True))
        return 0
    if args.command == "run":
        result = run_plan(args.plan.resolve(), args.reuse_existing)
        print(json.dumps({"instances": result["instances"], "manifest": str(DEFAULT_RESULT)}))
        return 0
    statistics = analyze_result(
        args.result.resolve(), args.plan.resolve(), args.bootstrap_replicates
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": statistics["passed"], "decision": statistics["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
