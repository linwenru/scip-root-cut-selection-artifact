"""Build reproducible seen- and unseen-family evaluation manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .audit import DEFAULT_MIPLIB_METADATA, SPLITS, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "vendor" / "tracer_snapshot" / "split"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "manifests" / "evaluation_protocols"
PROTOCOLS = ("official_group_ood", "seen_family", "officially_ungrouped")


def _read_split(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate instance in split file: {path}")
    return names


def load_original_splits(split_dir: Path) -> dict[str, list[str]]:
    """Load train/val/test lists and reject cross-split instance leakage."""
    splits = {split: _read_split(split_dir / f"{split}.test") for split in SPLITS}
    owner: dict[str, str] = {}
    for split, names in splits.items():
        for name in names:
            previous = owner.setdefault(name, split)
            if previous != split:
                raise ValueError(
                    f"Instance {name!r} occurs in both {previous!r} and {split!r}"
                )
    return splits


def load_official_groups(metadata_path: Path) -> dict[str, str]:
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"instance_name", "group"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Metadata must contain columns {sorted(required)}")
        rows = list(reader)
    names = [row["instance_name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate instance in metadata: {metadata_path}")
    return {row["instance_name"]: row["group"].strip() for row in rows}


def _groups_for(names: list[str], group_by_instance: dict[str, str]) -> set[str]:
    return {group_by_instance[name] for name in names if group_by_instance[name]}


def _manifest_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def build_protocols(split_dir: Path, metadata_path: Path) -> dict[str, object]:
    """Classify the supplied split without moving or copying raw traces."""
    splits = load_original_splits(split_dir)
    group_by_instance = load_official_groups(metadata_path)
    all_names = {name for names in splits.values() for name in names}
    missing_metadata = sorted(all_names - group_by_instance.keys())
    if missing_metadata:
        raise ValueError(f"Instances missing from official metadata: {missing_metadata}")

    train_groups = _groups_for(splits["train"], group_by_instance)
    assignments = []
    strata: dict[str, dict[str, list[str]]] = {
        split: {
            "seen_family": [],
            "unseen_family": [],
            "officially_ungrouped": [],
        }
        for split in ("val", "test")
    }

    for split in SPLITS:
        for name in splits[split]:
            group = group_by_instance[name]
            if split == "train":
                stratum = "training_grouped" if group else "training_ungrouped"
            elif not group:
                stratum = "officially_ungrouped"
                strata[split][stratum].append(name)
            elif group in train_groups:
                stratum = "seen_family"
                strata[split][stratum].append(name)
            else:
                stratum = "unseen_family"
                strata[split][stratum].append(name)
            assignments.append(
                {
                    "instance_name": name,
                    "original_split": split,
                    "official_group": group,
                    "evaluation_stratum": stratum,
                }
            )

    lists = {
        "official_group_ood": {
            "train": list(splits["train"]),
            "val": strata["val"]["unseen_family"],
            "test": strata["test"]["unseen_family"],
        },
        "seen_family": {
            "train": list(splits["train"]),
            "val": strata["val"]["seen_family"],
            "test": strata["test"]["seen_family"],
        },
        "officially_ungrouped": {
            "train": list(splits["train"]),
            "val": strata["val"]["officially_ungrouped"],
            "test": strata["test"]["officially_ungrouped"],
        },
    }

    ood_val_groups = _groups_for(lists["official_group_ood"]["val"], group_by_instance)
    ood_test_groups = _groups_for(lists["official_group_ood"]["test"], group_by_instance)
    seen_val_groups = _groups_for(lists["seen_family"]["val"], group_by_instance)
    seen_test_groups = _groups_for(lists["seen_family"]["test"], group_by_instance)
    checks = {
        "all_instances_have_metadata": not missing_metadata,
        "evaluation_partition_is_complete": all(
            sum(len(names) for names in strata[split].values()) == len(splits[split])
            for split in ("val", "test")
        ),
        "ood_val_groups_disjoint_from_train": train_groups.isdisjoint(ood_val_groups),
        "ood_test_groups_disjoint_from_train": train_groups.isdisjoint(ood_test_groups),
        "ood_val_and_test_groups_are_disjoint": ood_val_groups.isdisjoint(ood_test_groups),
        "seen_val_groups_are_in_train": seen_val_groups <= train_groups,
        "seen_test_groups_are_in_train": seen_test_groups <= train_groups,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Protocol invariants failed: {failed}")

    split_hashes = {
        f"{split}.test": sha256_file(split_dir / f"{split}.test") for split in SPLITS
    }
    summary = {
        "schema_version": 1,
        "source": {
            "split_dir": _manifest_path(split_dir),
            "split_sha256": split_hashes,
            "miplib_metadata": _manifest_path(metadata_path),
            "miplib_metadata_sha256": sha256_file(metadata_path),
        },
        "semantics": {
            "official_group_ood": (
                "Primary protocol: validation and test Group values are absent from training; "
                "validation and test Groups are also disjoint."
            ),
            "seen_family": (
                "Secondary protocol: validation and test Group values are represented in training."
            ),
            "officially_ungrouped": (
                "Diagnostic protocol: MIPLIB publishes no Group, so family overlap is unknown."
            ),
            "training_caveat": (
                "All protocols use the original training pool. Its ungrouped instances cannot be "
                "certified family-disjoint from evaluation instances."
            ),
        },
        "original_counts": {split: len(names) for split, names in splits.items()},
        "training_group_counts": {
            "grouped_instances": sum(bool(group_by_instance[name]) for name in splits["train"]),
            "ungrouped_instances": sum(not group_by_instance[name] for name in splits["train"]),
            "distinct_official_groups": len(train_groups),
        },
        "protocol_counts": {
            protocol: {split: len(names) for split, names in by_split.items()}
            for protocol, by_split in lists.items()
        },
        "official_groups": {
            "train": sorted(train_groups),
            "ood_val": sorted(ood_val_groups),
            "ood_test": sorted(ood_test_groups),
            "seen_val": sorted(seen_val_groups),
            "seen_test": sorted(seen_test_groups),
        },
        "checks": checks,
    }
    return {"assignments": assignments, "lists": lists, "summary": summary}


def write_protocols(protocols: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "instance_assignments.csv"
    with assignments_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "instance_name",
                "original_split",
                "official_group",
                "evaluation_stratum",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(protocols["assignments"])

    for protocol in PROTOCOLS:
        protocol_dir = output_dir / protocol
        protocol_dir.mkdir(exist_ok=True)
        for split in SPLITS:
            names = protocols["lists"][protocol][split]
            text = "".join(f"{name}\n" for name in names)
            (protocol_dir / f"{split}.test").write_text(text, encoding="utf-8")

    summary_path = output_dir / "protocol_summary.json"
    summary_path.write_text(
        json.dumps(protocols["summary"], indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--miplib-metadata", type=Path, default=DEFAULT_MIPLIB_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocols = build_protocols(args.split_dir.resolve(), args.miplib_metadata.resolve())
    write_protocols(protocols, args.output_dir)
    counts = protocols["summary"]["protocol_counts"]
    print(f"Evaluation protocols written to {args.output_dir}")
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
