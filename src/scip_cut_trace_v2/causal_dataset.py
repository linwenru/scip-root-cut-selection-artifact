"""Build one-context, multi-action causal records from active SCIP experiments."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .causal_context import context_sha256


SCHEMA_VERSION = 1
COMPLETE_NATIVE_STATUSES = frozenset(("optimal", "infeasible", "unbounded", "inforunbd"))
FORBIDDEN_CONTEXT_KEYS = frozenset(
    (
        "outcome",
        "solving_time",
        "primal_dual_integral",
        "final_status",
        "final_primal_bound",
        "final_dual_bound",
    )
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_forbidden_keys(value: Any, location: str = "context") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in FORBIDDEN_CONTEXT_KEYS:
                found.append(child_location)
            found.extend(_find_forbidden_keys(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, f"{location}[{index}]"))
    return found


def _action_label(action_result: dict[str, Any]) -> dict[str, Any]:
    comparison = action_result["comparison"]
    metrics = {
        metric: {
            "native": values["native"],
            "treatment": values["treatment"],
            "delta_treatment_minus_native": values[
                "delta_treatment_minus_native"
            ],
            "relative_saving": values["relative_saving"],
        }
        for metric, values in comparison["metrics"].items()
    }
    return {
        "safe": comparison["safe"],
        "eligible": comparison["eligible"],
        "valid": comparison["valid"],
        "treatment_status": action_result["outcome"]["status"],
        "interventions": action_result["selector"]["interventions"],
        "metrics": metrics,
    }


def _load_shared_context(
    action_results: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    contexts = []
    for action, action_result in action_results.items():
        result_path = Path(action_result["result_path"])
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        raw_result = json.loads(result_path.read_text(encoding="utf-8"))
        records = raw_result["selector"].get("context_records", [])
        if len(records) != 1:
            raise ValueError(
                f"Expected one first-run context for {action}, found {len(records)}"
            )
        record = records[0]
        context = record["decision_context"]
        digest = context_sha256(context)
        if digest != record["context_sha256"]:
            raise ValueError(f"Context payload hash mismatch for {result_path}")
        if digest != action_result["initial_context_sha256"]:
            raise ValueError(f"Manifest context hash mismatch for {result_path}")
        contexts.append((digest, context))

    digests = {digest for digest, _ in contexts}
    if len(digests) != 1:
        raise ValueError("Action arms do not share one pre-intervention context")
    first_context = contexts[0][1]
    if any(context != first_context for _, context in contexts[1:]):
        raise ValueError("Equal context hashes unexpectedly have unequal payloads")
    forbidden = _find_forbidden_keys(first_context)
    if forbidden:
        raise ValueError(f"Post-outcome fields found in context: {forbidden}")
    return first_context, contexts[0][0]


def build_records(
    experiment_manifest: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    if experiment_manifest.get("intervention_scope") != "first-run-only":
        raise ValueError("Causal dataset requires a first-run-only experiment")
    actions = tuple(experiment_manifest["actions"])
    records = []
    seen = set()
    for instance_result in experiment_manifest["per_instance"]:
        for pair in instance_result["pairs"]:
            key = (instance_result["instance_id"], int(pair["seed"]))
            if key in seen:
                raise ValueError(f"Duplicate instance-seed causal record: {key}")
            seen.add(key)
            initial_context = pair["initial_context"]
            if initial_context.get("no_action_observed", False):
                continue
            if initial_context.get("partial_actions_observed", False):
                raise ValueError(f"Only some action contexts were observed for {key}")
            if not initial_context["matching_across_actions"]:
                raise ValueError(f"Mismatched action contexts for {key}")
            if pair["native_outcome"]["status"] not in COMPLETE_NATIVE_STATUSES:
                continue
            context, digest = _load_shared_context(pair["actions"])
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "split": split,
                    "instance_id": instance_result["instance_id"],
                    "instance": instance_result["instance"],
                    "instance_sha256": instance_result["instance_sha256"],
                    "seed": pair["seed"],
                    "context_sha256": digest,
                    "context": context,
                    "native_outcome": pair["native_outcome"],
                    "action_labels": {
                        action: _action_label(pair["actions"][action])
                        for action in actions
                    },
                    "post_hoc_oracle": pair["oracle"],
                }
            )
    return records


def _write_jsonl_gzip(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as gzip_handle:
            for record in records:
                line = json.dumps(
                    record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                )
                gzip_handle.write(line.encode("utf-8") + b"\n")
    temporary.replace(path)


def build_dataset(
    source_manifest: Path | Iterable[Path],
    output: Path,
    dataset_manifest: Path,
    split: str,
) -> dict[str, Any]:
    source_manifests = (
        (source_manifest,)
        if isinstance(source_manifest, Path)
        else tuple(source_manifest)
    )
    if not source_manifests:
        raise ValueError("at least one source manifest is required")
    source_manifests = tuple(path.resolve() for path in source_manifests)
    experiments = [
        json.loads(path.read_text(encoding="utf-8")) for path in source_manifests
    ]
    action_sets = {tuple(experiment["actions"]) for experiment in experiments}
    if len(action_sets) != 1:
        raise ValueError("all source manifests must use the same action set")
    actions = action_sets.pop()

    records = [
        record
        for experiment in experiments
        for record in build_records(experiment, split)
    ]
    record_keys = [(record["instance_id"], record["seed"]) for record in records]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("Duplicate instance-seed record across source manifests")
    source_pairs = sum(
        len(instance_result["pairs"])
        for experiment in experiments
        for instance_result in experiment["per_instance"]
    )
    source_no_context_pairs = sum(
        pair["initial_context"].get("no_action_observed", False)
        for experiment in experiments
        for instance_result in experiment["per_instance"]
        for pair in instance_result["pairs"]
    )
    source_native_incomplete_pairs = sum(
        not pair["initial_context"].get("no_action_observed", False)
        and pair["native_outcome"]["status"] not in COMPLETE_NATIVE_STATUSES
        for experiment in experiments
        for instance_result in experiment["per_instance"]
        for pair in instance_result["pairs"]
    )
    if (
        len(records)
        + source_no_context_pairs
        + source_native_incomplete_pairs
        != source_pairs
    ):
        raise ValueError("Causal dataset source-pair accounting failed")
    _write_jsonl_gzip(output, records)

    action_counts = {
        action: {
            "safe": sum(record["action_labels"][action]["safe"] for record in records),
            "valid": sum(
                record["action_labels"][action]["valid"] for record in records
            ),
            "positive_lp_saving": sum(
                record["action_labels"][action]["valid"]
                and record["action_labels"][action]["metrics"]["lp_iterations"][
                    "relative_saving"
                ]
                > 0.0
                for record in records
            ),
        }
        for action in actions
    }
    candidate_counts = [len(record["context"]["candidates"]) for record in records]
    status_counts = Counter(
        record["native_outcome"]["status"] for record in records
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifests": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in source_manifests
        ],
        "output": str(output.resolve()),
        "output_sha256": _sha256_file(output),
        "split": split,
        "records": len(records),
        "source_pairs": source_pairs,
        "skipped_no_context_pairs": source_no_context_pairs,
        "skipped_native_incomplete_pairs": source_native_incomplete_pairs,
        "instances": len({record["instance_id"] for record in records}),
        "seeds": sorted({record["seed"] for record in records}),
        "actions": list(actions),
        "context_contract": (
            "one shared pre-intervention first-run context per instance and seed; "
            "no solve time, final outcome, or post-action field in model context"
        ),
        "candidate_count": {
            "min": min(candidate_counts) if candidate_counts else None,
            "max": max(candidate_counts) if candidate_counts else None,
            "mean": (
                sum(candidate_counts) / len(candidate_counts)
                if candidate_counts
                else None
            ),
        },
        "native_status_counts": dict(sorted(status_counts.items())),
        "action_label_counts": action_counts,
    }
    dataset_manifest.parent.mkdir(parents=True, exist_ok=True)
    dataset_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        nargs="+",
        default=[
            Path("data/manifests/causal_action_oracle_first_run_pilot_v1.json")
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/causal_first_run_pilot_v1.jsonl.gz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/causal_first_run_dataset_v1.json"),
    )
    parser.add_argument("--split", default="train")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_dataset(
        args.source_manifest, args.output, args.manifest, args.split
    )
    print(json.dumps({"records": manifest["records"], "output": manifest["output"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
