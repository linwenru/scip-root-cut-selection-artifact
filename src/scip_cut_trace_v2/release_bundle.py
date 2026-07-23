"""Build and verify deterministic publication artifact bundles."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PREFIX = "scip-root-cut-selection-artifact"

RELEASE_TIERS: Mapping[str, tuple[str, ...]] = {
    "source": (
        ".dockerignore",
        ".gitattributes",
        ".gitignore",
        ".zenodo.json",
        "CITATION.cff",
        "Dockerfile",
        "LICENSE*",
        "NOTICE",
        "README.md",
        "pyproject.toml",
        "requirements-core.lock",
        "setup.py",
        "docs/**/*.md",
        "models/ranking_imitation_xgb_v1/model.ubj",
        "src/**/*.py",
        "src/**/*.pyx",
        "tests/**/*.py",
        "vendor/tracer_snapshot/**/*.csv",
        "vendor/tracer_snapshot/**/*.md",
        "vendor/tracer_snapshot/**/*.py",
        "vendor/tracer_snapshot/**/*.test",
        "vendor/tracer_snapshot/**/*.txt",
    ),
    "manifests": (
        "data/manifests/**/*.csv",
        "data/manifests/**/*.json",
    ),
    "active-results": (
        "experiments/**/*.json",
        "models/**/*.ubj",
    ),
    "derived-data": (
        "data/datasets/**/*.npz",
        "data/processed/*.jsonl.gz",
    ),
    "observational-view": (
        "data/processed/root_observational_v1/**/*.csv.gz",
    ),
}

# The working repository remains private. Public archives deliberately omit the
# manuscript, reviewer correspondence, and internal revision notes.
PUBLIC_RELEASE_EXCLUDED_PREFIXES = ("paper/",)
PUBLIC_RELEASE_EXCLUDED_PATHS = frozenset({"docs/MAJOR_REVISION_PLAN.md"})

DEFAULT_TIERS = ("source", "manifests", "active-results", "derived-data")


def _is_external_release_inventory(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    return (
        relative.parent == Path("data/manifests")
        and relative.name.startswith("release_")
        and relative.suffix == ".json"
    )


def _is_private_release_material(root: Path, path: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    return relative in PUBLIC_RELEASE_EXCLUDED_PATHS or relative.startswith(
        PUBLIC_RELEASE_EXCLUDED_PREFIXES
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_tier(root: Path, patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(
            path
            for path in root.glob(pattern)
            if path.is_file()
            and not _is_external_release_inventory(root, path)
            and not _is_private_release_material(root, path)
        )
    return sorted(paths)


def build_release_inventory(
    root: Path = PROJECT_ROOT,
    tiers: Sequence[str] = DEFAULT_TIERS,
    tier_patterns: Mapping[str, tuple[str, ...]] = RELEASE_TIERS,
) -> dict[str, Any]:
    """Return a content-addressed inventory for the selected release tiers."""
    root = root.resolve()
    unknown = sorted(set(tiers) - set(tier_patterns))
    if unknown:
        raise ValueError(f"Unknown release tiers: {', '.join(unknown)}")

    owners: dict[Path, set[str]] = defaultdict(set)
    for tier in tiers:
        paths = _resolve_tier(root, tier_patterns[tier])
        if not paths:
            raise FileNotFoundError(f"Release tier matched no files: {tier}")
        for path in paths:
            owners[path].add(tier)

    files = []
    tier_summary = {
        tier: {"files": 0, "bytes": 0}
        for tier in tiers
    }
    for path in sorted(owners):
        relative = str(path.relative_to(root))
        size = path.stat().st_size
        file_tiers = sorted(owners[path])
        files.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "bytes": size,
                "tiers": file_tiers,
            }
        )
        for tier in file_tiers:
            tier_summary[tier]["files"] += 1
            tier_summary[tier]["bytes"] += size

    return {
        "schema_version": SCHEMA_VERSION,
        "archive_prefix": ARCHIVE_PREFIX,
        "tiers": list(tiers),
        "tier_summary": tier_summary,
        "files": files,
        "summary": {
            "files": len(files),
            "bytes": sum(record["bytes"] for record in files),
        },
    }


def _normalized_tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def write_release_bundle(
    output: Path,
    inventory: dict[str, Any],
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Write a deterministic gzip-compressed tar archive and return its record."""
    root = root.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = (
        json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    with output.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
                manifest_name = f"{ARCHIVE_PREFIX}/RELEASE_MANIFEST.json"
                archive.addfile(
                    _normalized_tar_info(manifest_name, len(manifest_payload)),
                    io.BytesIO(manifest_payload),
                )
                for record in inventory["files"]:
                    path = root / record["path"]
                    info = _normalized_tar_info(
                        f"{ARCHIVE_PREFIX}/{record['path']}", record["bytes"]
                    )
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)

    return {
        "path": output.name,
        "sha256": _sha256_file(output),
        "bytes": output.stat().st_size,
        "manifest_sha256": _sha256_bytes(manifest_payload),
    }


def verify_release_bundle(path: Path) -> dict[str, Any]:
    """Verify member names, sizes, and hashes against the embedded manifest."""
    mismatches = []
    prefix = f"{ARCHIVE_PREFIX}/"
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            mismatches.append({"reason": "duplicate-member"})
        unsafe = [
            name
            for name in names
            if not name.startswith(prefix) or ".." in Path(name).parts
        ]
        if unsafe:
            mismatches.append({"reason": "unsafe-member", "members": unsafe})

        manifest_name = f"{ARCHIVE_PREFIX}/RELEASE_MANIFEST.json"
        manifest_member = archive.getmember(manifest_name)
        manifest_handle = archive.extractfile(manifest_member)
        if manifest_handle is None:
            raise ValueError("Release manifest is not a regular file")
        inventory = json.load(manifest_handle)

        expected_names = {
            manifest_name,
            *(f"{ARCHIVE_PREFIX}/{record['path']}" for record in inventory["files"]),
        }
        for name in sorted(expected_names - set(names)):
            mismatches.append({"path": name, "reason": "missing"})
        for name in sorted(set(names) - expected_names):
            mismatches.append({"path": name, "reason": "unexpected"})

        for record in inventory["files"]:
            name = f"{ARCHIVE_PREFIX}/{record['path']}"
            try:
                member = archive.getmember(name)
            except KeyError:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                mismatches.append({"path": name, "reason": "not-a-file"})
                continue
            payload = handle.read()
            if len(payload) != record["bytes"] or _sha256_bytes(payload) != record["sha256"]:
                mismatches.append({"path": name, "reason": "content-mismatch"})

    return {
        "passed": not mismatches,
        "checked_files": len(inventory["files"]),
        "tiers": inventory["tiers"],
        "mismatches": mismatches,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory-output", type=Path)
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=sorted(RELEASE_TIERS),
        default=list(DEFAULT_TIERS),
    )
    parser.add_argument("--verify", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify is not None:
        result = verify_release_bundle(args.verify)
        print(json.dumps(result))
        return 0 if result["passed"] else 2
    if args.output is None or args.inventory_output is None:
        raise SystemExit("--output and --inventory-output are required when building")

    inventory = build_release_inventory(args.root, args.tiers)
    archive = write_release_bundle(args.output, inventory, args.root)
    release_record = {**inventory, "archive": archive}
    args.inventory_output.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_output.write_text(
        json.dumps(release_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"archive": archive, "summary": inventory["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
