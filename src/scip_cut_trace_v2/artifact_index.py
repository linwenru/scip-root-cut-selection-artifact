"""Build and verify the machine-readable paper evidence index."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLAIMS = (
    {
        "claim_id": "self-generated-trace-corpus",
        "statement": "The tracer, batch runner, 233-instance easy subset, and fixed 163/35/35 split were generated and versioned by this study.",
        "patterns": (
            "vendor/tracer_snapshot/prepare_split.py",
            "vendor/tracer_snapshot/run_benchmark.py",
            "vendor/tracer_snapshot/scip_cut_logger/*.py",
            "vendor/tracer_snapshot/split/benchmark_tags.csv",
            "vendor/tracer_snapshot/split/instance_metadata.csv",
            "vendor/tracer_snapshot/split/train.test",
            "vendor/tracer_snapshot/split/val.test",
            "vendor/tracer_snapshot/split/test.test",
        ),
    },
    {
        "claim_id": "group-aware-evaluation",
        "statement": "Official MIPLIB Group-disjoint evaluation is separated from seen-family interpolation.",
        "patterns": (
            "data/manifests/evaluation_protocols/instance_assignments.csv",
            "data/manifests/evaluation_protocols/protocol_summary.json",
            "docs/EVALUATION_PROTOCOLS.md",
        ),
    },
    {
        "claim_id": "leakage-safe-observational-view",
        "statement": "The root observational and ranking matrices exclude post-outcome fields and use instance-balanced queries.",
        "patterns": (
            "data/manifests/root_observational_v1.json",
            "data/manifests/ranking_imitation_v1.json",
            "docs/ROOT_OBSERVATIONAL_DATA.md",
            "docs/RANKING_IMITATION_DATA.md",
        ),
    },
    {
        "claim_id": "imitation-not-online-authorized",
        "statement": "The XGBoost imitation model fails its external gate and is not an online treatment policy.",
        "patterns": (
            "data/manifests/ranking_imitation_xgb_v1.json",
            "data/manifests/ranking_imitation_xgb_group_cv_v1.json",
            "docs/IMITATION_BASELINE.md",
            "docs/IMITATION_ONLINE_AUDIT.md",
        ),
    },
    {
        "claim_id": "causal-instrumentation-parity",
        "statement": "The strict 36/36 parity gate failed, while all 34 complete callback-exercised pairs per neutral arm matched structurally.",
        "patterns": (
            "data/manifests/causal_noop_parity_v1.json",
            "data/manifests/causal_direct_hybrid_parity_v1.json",
            "docs/CAUSAL_HARNESS.md",
            "src/scip_cut_trace_v2/causal_harness.py",
            "src/scip_cut_trace_v2/native_hybrid.py",
            "tests/test_causal_harness.py",
        ),
    },
    {
        "claim_id": "causal-risk-no-policy",
        "statement": "The action library, grouped risk model, and independent cohort C do not establish a safe generalized policy.",
        "patterns": (
            "data/manifests/causal_risk_grouped_oof_v1.json",
            "data/manifests/causal_first_run_efficacy_cohort_c_confirmation_v1.json",
            "data/manifests/paper_statistics_cohort_c_development_v1.json",
            "docs/CAUSAL_RISK_DIAGNOSTIC.md",
        ),
    },
    {
        "claim_id": "frozen-learned-policy-pilot-no-go",
        "statement": "A pre-registered one-shot complete-solve pilot on 18 new Group-disjoint instances rejects the frozen XGBoost imitation ranker and deployment path.",
        "patterns": (
            "data/manifests/learned_policy_pilot_plan_v1.json",
            "data/manifests/causal_learned_policy_pilot_v1.json",
            "data/manifests/learned_policy_pilot_statistics_v1.json",
            "data/manifests/learned_policy_pilot_diagnostics_v1.json",
            "models/ranking_imitation_xgb_v1/model.ubj",
            "docs/LEARNED_POLICY_PILOT.md",
            "src/scip_cut_trace_v2/learned_policy_pilot.py",
            "src/scip_cut_trace_v2/learned_policy_diagnostics.py",
            "tests/test_learned_policy_pilot.py",
            "tests/test_learned_policy_diagnostics.py",
        ),
    },
    {
        "claim_id": "fixed-baseline-pilot-rejection",
        "statement": "Random, efficacy, and Adaptive-score fixed baselines fail the advancement gate in both the original pilot and its pre-registered disjoint correction.",
        "patterns": (
            "data/manifests/causal_fixed_baseline_train_pilot_plan_v1.json",
            "data/manifests/causal_fixed_baseline_train_pilot_v1.json",
            "data/manifests/paper_statistics_fixed_baseline_train_pilot_v1.json",
            "data/manifests/causal_fixed_baseline_train_pilot_decision_v1.json",
            "data/manifests/causal_fixed_baseline_disjoint_train_pilot_plan_v2.json",
            "data/manifests/causal_fixed_baseline_disjoint_train_pilot_v2.json",
            "data/manifests/paper_statistics_fixed_baseline_disjoint_train_pilot_v2.json",
            "data/manifests/causal_fixed_baseline_disjoint_train_pilot_decision_v2.json",
            "docs/FIXED_BASELINE_PILOT_AUDIT.md",
            "src/scip_cut_trace_v2/cut_selection_baselines.py",
            "src/scip_cut_trace_v2/paper_statistics.py",
            "src/scip_cut_trace_v2/fixed_baseline_pilot.py",
            "tests/test_cut_selection_baselines.py",
            "tests/test_paper_statistics.py",
            "tests/test_fixed_baseline_pilot.py",
        ),
    },
    {
        "claim_id": "publication-strengthening-negative-result",
        "statement": "A frozen 40-instance, three-seed experiment finds no fixed root-ranking action that passes the solve-level performance and safety gate.",
        "patterns": (
            "data/manifests/publication_strengthening_plan_v1.json",
            "data/manifests/publication_active_plan_v1.json",
            "data/manifests/causal_publication_fixed_baselines_v1.json",
            "data/manifests/paper_statistics_publication_fixed_baselines_v1.json",
            "docs/PUBLICATION_STRENGTHENING_V1.md",
            "src/scip_cut_trace_v2/publication_protocol.py",
            "tests/test_publication_protocol.py",
        ),
    },
    {
        "claim_id": "online-compatible-imitation-baseline",
        "statement": "The 37-feature XGBoost arm is online-compatible but remains smoke-only because its official-Group-OOD authorization gate fails.",
        "patterns": (
            "data/manifests/ranking_imitation_online_v1.json",
            "data/manifests/ranking_imitation_online_xgb_v1.json",
            "data/manifests/causal_online_imitation_smoke_v1.json",
            "docs/IMITATION_ONLINE_AUDIT.md",
            "src/scip_cut_trace_v2/online_imitation.py",
            "tests/test_online_imitation.py",
        ),
    },
    {
        "claim_id": "release-ready-artifact",
        "statement": "Publication evidence is packaged by content-addressed tiers with a pinned SCIP container and explicit author-controlled release blockers.",
        "patterns": (
            ".dockerignore",
            ".zenodo.json",
            "CITATION.cff",
            "Dockerfile",
            "LICENSE",
            "NOTICE",
            "docs/ARTIFACT_RELEASE.md",
            "src/scip_cut_trace_v2/release_bundle.py",
            "tests/test_release_bundle.py",
        ),
    },
    {
        "claim_id": "consolidated-stop-decision",
        "statement": "Native SCIP remains the only authorized online policy under the frozen stopping rules.",
        "patterns": (
            "docs/PAPER_PROTOCOL.md",
            "docs/PROJECT_STATUS.md",
            "docs/REPRODUCIBILITY.md",
            "pyproject.toml",
            "src/scip_cut_trace_v2/artifact_index.py",
            "tests/test_artifact_index.py",
        ),
    },
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_patterns(root: Path, patterns: Iterable[str]) -> list[Path]:
    resolved = []
    for pattern in patterns:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            raise FileNotFoundError(f"Evidence pattern matched no files: {pattern}")
        resolved.extend(matches)
    return sorted(set(resolved))


def build_evidence_index(
    root: Path = PROJECT_ROOT, claims: Iterable[dict[str, Any]] = CLAIMS
) -> dict[str, Any]:
    root = root.resolve()
    claim_records = []
    artifacts: dict[str, dict[str, Any]] = {}
    for claim in claims:
        paths = _resolve_patterns(root, claim["patterns"])
        relative_paths = [str(path.relative_to(root)) for path in paths]
        claim_records.append(
            {
                "claim_id": claim["claim_id"],
                "statement": claim["statement"],
                "evidence": relative_paths,
            }
        )
        for path, relative in zip(paths, relative_paths):
            artifacts[relative] = {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root_name": root.name,
        "claims": claim_records,
        "artifacts": dict(sorted(artifacts.items())),
        "summary": {
            "claims": len(claim_records),
            "artifacts": len(artifacts),
            "bytes": sum(record["bytes"] for record in artifacts.values()),
        },
    }


def verify_evidence_index(index: dict[str, Any], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = root.resolve()
    mismatches = []
    for relative, expected in index["artifacts"].items():
        path = root / relative
        if not path.is_file():
            mismatches.append({"path": relative, "reason": "missing"})
            continue
        observed_hash = _sha256_file(path)
        observed_bytes = path.stat().st_size
        if observed_hash != expected["sha256"] or observed_bytes != expected["bytes"]:
            mismatches.append(
                {
                    "path": relative,
                    "reason": "content-mismatch",
                    "expected_sha256": expected["sha256"],
                    "observed_sha256": observed_hash,
                    "expected_bytes": expected["bytes"],
                    "observed_bytes": observed_bytes,
                }
            )
    return {
        "passed": not mismatches,
        "checked_artifacts": len(index["artifacts"]),
        "mismatches": mismatches,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "paper_evidence_index_v2.json",
    )
    parser.add_argument("--verify", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify is not None:
        index = json.loads(args.verify.read_text(encoding="utf-8"))
        result = verify_evidence_index(index)
        print(json.dumps(result))
        return 0 if result["passed"] else 2

    index = build_evidence_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": index["summary"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
