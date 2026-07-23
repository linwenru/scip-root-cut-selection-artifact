"""Audit a supplied SCIP cut-trace corpus without modifying raw outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SPLITS = ("train", "val", "test")
REQUIRED_OUTPUTS = (
    "candidate_cuts.csv",
    "applied_cuts.csv",
    "lp_states.csv",
    "sep_round_transitions.csv",
    "summary.json",
    "scip_statistics.txt",
)
DEFAULT_MIPLIB_METADATA = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "manifests"
    / "miplib2017_benchmark_metadata_2026-07-16.csv"
)


def parse_cutselector_stats(text: str, name: str) -> dict[str, int] | None:
    pattern = (
        rf"^\s*{re.escape(name)}\s*:\s+[\d.]+\s+[\d.]+\s+"
        r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    )
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        return None
    values = tuple(map(int, match.groups()))
    return dict(zip(("calls", "root_calls", "selected", "forced", "filtered"), values))


def parse_number_of_runs(text: str) -> int | None:
    match = re.search(r"number of runs\s*:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def last_csv_record(path: Path) -> list[str] | None:
    """Read the final non-empty CSV row without scanning a large file."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        if end == 0:
            return None
        start = max(0, end - 65536)
        handle.seek(start)
        lines = handle.read().decode("utf-8", errors="replace").splitlines()
    lines = [line for line in lines if line.strip()]
    if not lines or (start == 0 and len(lines) == 1):
        return None
    return next(csv.reader([lines[-1]]))


def family_key(instance_name: str) -> str:
    """Return a conservative heuristic key used only to flag split leakage."""
    name = instance_name.lower()
    name = re.sub(r"\.mps(?:\.gz)?$", "", name)
    pieces = []
    for piece in re.split(r"[-_]", name):
        alpha = re.sub(r"\d+", "", piece)
        if alpha:
            pieces.append(alpha)
    return "-".join(pieces)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _instance_dirs(source: Path) -> Iterable[tuple[str, Path]]:
    for split in SPLITS:
        split_dir = source / f"benchmark_output_{split}"
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.iterdir()):
            if path.is_dir():
                yield split, path


def _read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle), [])


def _split_names(source: Path) -> dict[str, list[str]]:
    result = {}
    for split in SPLITS:
        path = source / "split" / f"{split}.test"
        result[split] = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return result


def _name_family_overlap(source: Path) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for split, names in _split_names(source).items():
        for name in names:
            grouped[family_key(name)][split].append(name)
    ignored = {"", "a", "m", "neos", "ns", "s"}
    findings = []
    for key, by_split in sorted(grouped.items()):
        if key not in ignored and len(by_split) > 1:
            findings.append({"family_key": key, "splits": dict(by_split)})
    return findings


def _load_official_groups(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["instance_name"]: row["group"].strip()
            for row in csv.DictReader(handle)
        }


def _official_group_analysis(source: Path, metadata_path: Path) -> dict[str, object]:
    group_by_instance = _load_official_groups(metadata_path)
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    ungrouped: dict[str, list[str]] = defaultdict(list)
    unknown: dict[str, list[str]] = defaultdict(list)

    for split, names in _split_names(source).items():
        for name in names:
            if name not in group_by_instance:
                unknown[split].append(name)
                continue
            group = group_by_instance[name]
            if not group:
                ungrouped[split].append(name)
                continue
            grouped[group][split].append(name)

    cross_split = [
        {"group": group, "splits": dict(by_split)}
        for group, by_split in sorted(grouped.items())
        if len(by_split) > 1
    ]
    train_groups = {group for group, by_split in grouped.items() if "train" in by_split}
    evaluation_strata = {}
    for split in ("val", "test"):
        seen = []
        unseen = []
        for group, by_split in grouped.items():
            for name in by_split.get(split, []):
                target = seen if group in train_groups else unseen
                target.append({"instance_name": name, "group": group})
        evaluation_strata[split] = {
            "seen_family": sorted(seen, key=lambda item: item["instance_name"]),
            "unseen_family": sorted(unseen, key=lambda item: item["instance_name"]),
            "officially_ungrouped": sorted(ungrouped.get(split, [])),
        }

    return {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": sha256_file(metadata_path),
        "instances_with_official_group": sum(
            len(instances) for by_split in grouped.values() for instances in by_split.values()
        ),
        "instances_without_official_group": sum(len(names) for names in ungrouped.values()),
        "unknown_instances": dict(unknown),
        "cross_split_groups": cross_split,
        "evaluation_strata": evaluation_strata,
    }


def inventory(source: Path, miplib_metadata: Path | None = None) -> dict[str, object]:
    report: dict[str, object] = {
        "source": str(source.resolve()),
        "generated_at_unix": time.time(),
        "instances_by_split": Counter(),
        "solver_status_by_split": Counter(),
        "totals": Counter(),
        "missing_outputs": [],
        "header_mismatches": [],
        "cutselector_parse_failures": [],
        "cutselector_call_mismatches": [],
        "run_count_mismatches": [],
    }
    canonical_headers: dict[str, list[str]] = {}
    total_scip_runs = 0
    multi_run_instances = 0

    for split, instance_dir in _instance_dirs(source):
        name = instance_dir.name
        report["instances_by_split"][split] += 1
        missing = [filename for filename in REQUIRED_OUTPUTS if not (instance_dir / filename).is_file()]
        if missing:
            report["missing_outputs"].append({"split": split, "instance": name, "files": missing})
            continue

        summary = json.loads((instance_dir / "summary.json").read_text())
        report["solver_status_by_split"][f"{split}:{summary.get('status')}"] += 1
        for source_key, total_key in (
            ("n_candidate_cuts", "candidate_rows"),
            ("n_applied_cuts", "applied_event_rows"),
            ("n_lp_states", "lp_state_rows"),
        ):
            report["totals"][total_key] += int(summary.get(source_key) or 0)

        for filename in REQUIRED_OUTPUTS[:4]:
            header = _read_header(instance_dir / filename)
            expected = canonical_headers.setdefault(filename, header)
            if header != expected:
                report["header_mismatches"].append(
                    {"split": split, "instance": name, "file": filename}
                )

        stats_text = (instance_dir / "scip_statistics.txt").read_text(errors="replace")
        logger = parse_cutselector_stats(stats_text, "py_cutsel_logger")
        hybrid = parse_cutselector_stats(stats_text, "hybrid")
        if logger is None or hybrid is None:
            report["cutselector_parse_failures"].append(
                {"split": split, "instance": name, "logger": logger, "hybrid": hybrid}
            )
        else:
            for key in logger:
                report["totals"][f"logger_{key}"] += logger[key]
                report["totals"][f"hybrid_{key}"] += hybrid[key]
            if (logger["calls"], logger["root_calls"]) != (
                hybrid["calls"], hybrid["root_calls"]
            ):
                report["cutselector_call_mismatches"].append(
                    {"split": split, "instance": name, "logger": logger, "hybrid": hybrid}
                )

        scip_runs = parse_number_of_runs(stats_text)
        if scip_runs is not None:
            total_scip_runs += scip_runs
            multi_run_instances += scip_runs > 1
            observed_runs = 0
            per_file = {}
            for filename in ("candidate_cuts.csv", "applied_cuts.csv", "lp_states.csv"):
                row = last_csv_record(instance_dir / filename)
                value = int(row[0]) if row else 0
                per_file[filename] = value
                observed_runs = max(observed_runs, value)
            if observed_runs != scip_runs and observed_runs != 0:
                report["run_count_mismatches"].append(
                    {
                        "split": split,
                        "instance": name,
                        "scip_runs": scip_runs,
                        "observed_runs": observed_runs,
                        "per_file": per_file,
                    }
                )

    transition_rows = 0
    root_transition_rows = 0
    runs_with_transitions = set()
    runs_with_root_transition = set()
    for split, instance_dir in _instance_dirs(source):
        path = instance_dir / "sep_round_transitions.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                transition_rows += 1
                key = (split, instance_dir.name, int(row["run_number"]))
                runs_with_transitions.add(key)
                if int(float(row["node_depth"])) == 0:
                    root_transition_rows += 1
                    runs_with_root_transition.add(key)

    report["totals"]["scip_runs"] = total_scip_runs
    report["totals"]["multi_run_instances"] = multi_run_instances
    report["totals"]["transition_rows"] = transition_rows
    report["totals"]["root_transition_rows"] = root_transition_rows
    report["totals"]["runs_with_transitions"] = len(runs_with_transitions)
    report["totals"]["runs_with_root_transition"] = len(runs_with_root_transition)
    report["name_heuristic_cross_split_candidates"] = _name_family_overlap(source)
    if miplib_metadata is not None and miplib_metadata.is_file():
        report["official_group_analysis"] = _official_group_analysis(source, miplib_metadata)

    code_files = [
        source / "main.py",
        source / "run_benchmark.py",
        source / "scip_cut_logger" / "callbacks.py",
        source / "scip_cut_logger" / "row_features.py",
        source / "scip_cut_logger" / "solver.py",
        source / "scip_cut_logger" / "state_tracker.py",
    ]
    report["source_code_sha256"] = {
        str(path.relative_to(source)): sha256_file(path) for path in code_files if path.is_file()
    }
    report["canonical_headers"] = canonical_headers

    for key in ("instances_by_split", "solver_status_by_split", "totals"):
        report[key] = dict(report[key])
    return report


def _fast_csv_rows(path: Path) -> Iterable[list[str]]:
    """Parse ordinary rows cheaply while preserving correct handling of quoted names."""
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(handle, None)
        if header is None:
            return
        for line in handle:
            if '"' in line:
                yield next(csv.reader([line]))
            else:
                yield line.rstrip("\r\n").split(",")


def scan_candidates(source: Path, headers: dict[str, list[str]]) -> dict[str, object]:
    header = headers["candidate_cuts.csv"]
    index = {name: position for position, name in enumerate(header)}
    counts = Counter()
    origins = Counter()
    missing = Counter()
    split_rows = Counter()
    current_group = None
    seen_ids: set[str] = set()
    group_has_duplicate = False
    start = time.time()
    feature_columns = (
        "cut_id",
        "cut_name",
        "origin_type",
        "rank",
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
        "coeff_sparsity_ratio",
        "efficacy",
        "obj_parallelism",
        "cutoff_distance",
        "n_int_cols",
    )

    def finish_group() -> None:
        if current_group is not None:
            counts["decision_groups"] += 1
            counts["groups_with_duplicate_cut_id"] += group_has_duplicate

    for split, instance_dir in _instance_dirs(source):
        path = instance_dir / "candidate_cuts.csv"
        for row in _fast_csv_rows(path):
            if len(row) != len(header):
                counts["malformed_rows"] += 1
                continue
            counts["rows"] += 1
            split_rows[split] += 1
            group = (
                split,
                instance_dir.name,
                row[index["run_number"]],
                row[index["node_number"]],
                row[index["sep_round_node"]],
            )
            if group != current_group:
                finish_group()
                current_group = group
                seen_ids = set()
                group_has_duplicate = False

            cut_id = row[index["cut_id"]]
            if cut_id in seen_ids:
                counts["duplicate_cut_id_rows"] += 1
                counts["duplicate_applied_rows"] += row[index["is_applied"]] == "True"
                group_has_duplicate = True
            else:
                seen_ids.add(cut_id)

            counts["root_rows"] += row[index["root"]] == "True"
            counts["forced_rows"] += row[index["is_forced"]] == "True"
            counts["selected_blank_rows"] += row[index["is_selected"]] == ""
            counts["selected_true_rows"] += row[index["is_selected"]] == "True"
            counts["applied_rows"] += row[index["is_applied"]] == "True"
            counts["sparsity_one_rows"] += row[index["coeff_sparsity_ratio"]] == "1.0"
            counts["sparsity_zero_rows"] += row[index["coeff_sparsity_ratio"]] == "0.0"
            origins[row[index["origin_type"]]] += 1
            for column in feature_columns:
                missing[column] += row[index[column]] == ""
    finish_group()

    return {
        "elapsed_seconds": round(time.time() - start, 3),
        "counts": dict(counts),
        "rows_by_split": dict(split_rows),
        "origin_types": dict(origins),
        "missing_values": dict(missing),
    }


def validate_report(report: dict[str, object]) -> dict[str, object]:
    totals = report["totals"]
    candidate_scan = report.get("candidate_scan")
    checks = {
        "instance_count_is_233": sum(report["instances_by_split"].values()) == 233,
        "all_required_outputs_present": not report["missing_outputs"],
        "csv_headers_are_consistent": not report["header_mismatches"],
        "cutselector_stats_parse": not report["cutselector_parse_failures"],
        "logger_and_hybrid_calls_match": not report["cutselector_call_mismatches"],
        "observed_runs_match_scip": not report["run_count_mismatches"],
    }
    metrics = {
        "optimal_instance_fraction": (
            sum(value for key, value in report["solver_status_by_split"].items() if key.endswith(":optimal"))
            / sum(report["instances_by_split"].values())
        ),
        "transition_coverage": totals["transition_rows"] / totals["hybrid_calls"],
    }
    if candidate_scan is not None:
        counts = candidate_scan["counts"]
        checks.update(
            {
                "candidate_scan_matches_summaries": counts["rows"] == totals["candidate_rows"],
                "one_decision_group_per_hybrid_call": counts["decision_groups"] == totals["hybrid_calls"],
                "hybrid_accounting_matches_candidates": (
                    totals["hybrid_selected"] + totals["hybrid_filtered"] == counts["rows"]
                ),
                "only_forced_rows_have_selected_label": (
                    counts["selected_true_rows"] == counts["forced_rows"]
                    and counts["selected_blank_rows"] + counts["selected_true_rows"] == counts["rows"]
                ),
                "sparsity_feature_is_nonconstant": counts["sparsity_one_rows"] != counts["rows"],
                "applied_label_has_no_ambiguous_ids": counts["groups_with_duplicate_cut_id"] == 0,
            }
        )
        metrics.update(
            {
                "applied_label_fraction": counts["applied_rows"] / counts["rows"],
                "ambiguous_decision_group_fraction": (
                    counts["groups_with_duplicate_cut_id"] / counts["decision_groups"]
                ),
                "root_candidate_fraction": counts["root_rows"] / counts["rows"],
                "cutoff_distance_missing_fraction": (
                    candidate_scan["missing_values"]["cutoff_distance"] / counts["rows"]
                ),
            }
        )
    return {
        "verdict": "observational_data_usable_after_cleaning_not_causal_training_data",
        "checks": checks,
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Supplied tracer project root")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument(
        "--miplib-metadata",
        type=Path,
        default=DEFAULT_MIPLIB_METADATA,
        help="CSV snapshot containing the official MIPLIB Group column",
    )
    parser.add_argument(
        "--scan-candidates",
        action="store_true",
        help="Read all candidate rows to audit labels, missingness, and duplicate IDs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source.resolve()
    report = inventory(source, args.miplib_metadata)
    if args.scan_candidates:
        report["candidate_scan"] = scan_candidates(source, report["canonical_headers"])
    report["validation"] = validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"Audit written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
