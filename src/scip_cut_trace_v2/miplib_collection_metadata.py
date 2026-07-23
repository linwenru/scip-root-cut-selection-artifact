"""Build frozen MIPLIB Collection metadata from official downloaded files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
COLUMNS = (
    "instance_name",
    "status",
    "submitter",
    "group",
    "tags",
    "variables",
    "binaries",
    "integers",
    "continuous",
    "constraints",
    "nonzeros",
)
SOURCE_URLS = {
    "collection_html": "https://miplib.zib.de/set_collection.html",
    "easy_test": "https://miplib.zib.de/downloads/easy-v18.test",
    "infeasible_test": "https://miplib.zib.de/downloads/infeasible-v7.test",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _CollectionTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_body = False
        self.in_row = False
        self.in_cell = False
        self.cell_parts: list[str] = []
        self.row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "miplibtable":
            self.in_table = True
        elif self.in_table and tag == "tbody":
            self.in_body = True
        elif self.in_body and tag == "tr":
            self.in_row = True
            self.row = []
        elif self.in_row and tag == "td":
            self.in_cell = True
            self.cell_parts = []

    def handle_data(self, data: str) -> None:
        if self.in_cell and data.strip():
            self.cell_parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.in_cell:
            self.row.append(" ".join(self.cell_parts))
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.row:
                self.rows.append(self.row)
            self.in_row = False
        elif tag == "tbody" and self.in_body:
            self.in_body = False
        elif tag == "table" and self.in_table:
            self.in_table = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _read_test(path: Path) -> set[str]:
    return {
        line.strip().removesuffix(".mps.gz").removesuffix(".mps")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def parse_collection_html(path: Path) -> list[dict[str, str]]:
    parser = _CollectionTableParser()
    parser.feed(path.read_text(encoding="utf-8"))
    expected_cells = 11
    malformed = [row for row in parser.rows if len(row) != expected_cells]
    if malformed:
        raise ValueError(f"Collection table contains malformed rows: {malformed[:2]}")
    records = []
    for row in parser.rows:
        (
            instance,
            variables,
            binaries,
            integers,
            continuous,
            constraints,
            nonzeros,
            submitter,
            group,
            status,
            objective,
        ) = row
        records.append(
            {
                "instance": instance,
                "variables": variables,
                "binaries": binaries,
                "integers": integers,
                "continuous": continuous,
                "constraints": constraints,
                "nonzeros": nonzeros,
                "submitter": submitter,
                "group": "" if group in {"-", "–", "—"} else group,
                "page_status": status.lower(),
                "objective": objective,
            }
        )
    if len({record["instance"] for record in records}) != len(records):
        raise ValueError("Collection table contains duplicate instance names")
    return records


def build_metadata(
    collection_html: Path, easy_test: Path, infeasible_test: Path
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    source_records = parse_collection_html(collection_html)
    easy = _read_test(easy_test)
    infeasible = _read_test(infeasible_test)
    collection_names = {record["instance"] for record in source_records}
    if not easy <= collection_names or not infeasible <= collection_names:
        raise ValueError("A status test file contains names outside the Collection table")

    rows = []
    for record in source_records:
        instance = record["instance"]
        is_infeasible = instance in infeasible
        if is_infeasible:
            status = "infeasible"
        elif instance in easy:
            status = "easy"
        elif record["page_status"] == "easy":
            status = "not-easy-current"
        else:
            status = record["page_status"]
        rows.append(
            {
                "instance_name": f"{instance}.mps.gz",
                "status": status,
                "submitter": record["submitter"],
                "group": record["group"],
                "tags": "infeasible" if is_infeasible else "",
                "variables": record["variables"],
                "binaries": record["binaries"],
                "integers": record["integers"],
                "continuous": record["continuous"],
                "constraints": record["constraints"],
                "nonzeros": record["nonzeros"],
            }
        )

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "freeze official MIPLIB 2017 Collection Group metadata and current status "
            "sets for leakage-safe learned-policy pilot selection"
        ),
        "sources": {
            name: {
                "url": SOURCE_URLS[name],
                "path": _manifest_path(path),
                "sha256": _sha256_file(path),
            }
            for name, path in (
                ("collection_html", collection_html),
                ("easy_test", easy_test),
                ("infeasible_test", infeasible_test),
            )
        },
        "status_precedence": (
            "infeasible-v7 overrides easy-v18; easy-v18 overrides the Collection HTML; "
            "an HTML-only easy label absent from easy-v18 becomes not-easy-current"
        ),
        "summary": {
            "instances": len(rows),
            "grouped_instances": sum(bool(row["group"]) for row in rows),
            "unique_nonempty_groups": len({row["group"] for row in rows if row["group"]}),
            "status_counts": status_counts,
        },
    }
    return rows, provenance


def write_metadata(
    output: Path, provenance_output: Path, rows: list[dict[str, str]], provenance: dict[str, Any]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(provenance)
    provenance["output"] = {
        "path": _manifest_path(output),
        "sha256": _sha256_file(output),
    }
    provenance_output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-html", type=Path, required=True)
    parser.add_argument("--easy-test", type=Path, required=True)
    parser.add_argument("--infeasible-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, provenance = build_metadata(
        args.collection_html.resolve(),
        args.easy_test.resolve(),
        args.infeasible_test.resolve(),
    )
    write_metadata(args.output.resolve(), args.provenance_output.resolve(), rows, provenance)
    print(json.dumps(provenance["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
