# Reproducibility

## Supported Runtime

The active causal bridge is ABI-guarded for SCIP 10.0.2. The Python package
pins PySCIPOpt 6.2.1 and records Python, PySCIPOpt, SCIP, parameter, seed, and
instance hashes in every arm result. Active solves use one SCIP thread and one
fresh process per instance, seed, and arm.

The repository does not redistribute MIPLIB instance files or the roughly
22 GB source traces. Published historical manifests replace machine-specific
prefixes with `${PROJECT_ROOT}`, `${TRACE_SOURCE_ROOT}`,
`${MIPLIB_INSTANCES_DIR}`, and `${MIPLIB_ROOT}` while retaining official
instance names and SHA-256 hashes. The publication release archives include
active arm JSON, derived ranking matrices, and the leakage-safe observational
view; see `docs/ARTIFACT_RELEASE.md` for the content tiers and checksums.

## Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install --upgrade pip==25.2
SCIPOPTDIR=/opt/homebrew .venv/bin/pip install \
  --no-binary PySCIPOpt -r requirements-core.lock
SCIPOPTDIR=/opt/homebrew .venv/bin/pip install \
  --no-build-isolation --no-deps -e .
```

Use the actual SCIP installation prefix in `SCIPOPTDIR`. The native-hybrid
extension refuses any SCIP version other than 10.0.2.

`requirements-core.lock` records the exact Python snapshot used for model
training and the online harness.

## Verification

Run the complete test suite against the checked-out source:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
  -m unittest discover -s tests
```

Verify every file referenced by the study evidence index:

```bash
PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.artifact_index \
  --verify data/manifests/paper_evidence_index_v2.json
```

## Recompute Statistics

Recompute the 40-instance publication-strengthening estimate:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m \
  scip_cut_trace_v2.paper_statistics \
  data/manifests/causal_publication_fixed_baselines_v1.json \
  --actions random-rank efficacy-rank adaptive-score \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260717 \
  --output /tmp/paper_statistics_publication_fixed_baselines_v1.json
```

The expected gate result is false for all three actions. The expected PAR-2
treatment/native ratios are 1.1114 for random rank, 1.0647 for efficacy rank,
and 1.0019 for adaptive score.

Recompute the fixed-baseline pilot's instance-equal PAR-2 analysis from the
committed causal manifest:

```bash
PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.paper_statistics \
  data/manifests/causal_fixed_baseline_train_pilot_v1.json \
  --actions random-rank efficacy-rank adaptive-score \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260717 \
  --output /tmp/paper_statistics_fixed_baseline_train_pilot_v1.json
```

Reapply the pre-registered advancement gate:

```bash
PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.fixed_baseline_pilot \
  --plan data/manifests/causal_fixed_baseline_train_pilot_plan_v1.json \
  --causal-manifest data/manifests/causal_fixed_baseline_train_pilot_v1.json \
  --statistics data/manifests/paper_statistics_fixed_baseline_train_pilot_v1.json \
  --output /tmp/causal_fixed_baseline_train_pilot_decision_v1.json
```

The expected result is `passed=false` and no selected action.

Recompute the disjoint v2 correction in the same way:

```bash
PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.paper_statistics \
  data/manifests/causal_fixed_baseline_disjoint_train_pilot_v2.json \
  --actions random-rank efficacy-rank adaptive-score \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260717 \
  --output /tmp/paper_statistics_fixed_baseline_disjoint_train_pilot_v2.json

PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.fixed_baseline_pilot \
  --plan data/manifests/causal_fixed_baseline_disjoint_train_pilot_plan_v2.json \
  --causal-manifest data/manifests/causal_fixed_baseline_disjoint_train_pilot_v2.json \
  --statistics /tmp/paper_statistics_fixed_baseline_disjoint_train_pilot_v2.json \
  --output /tmp/causal_fixed_baseline_disjoint_train_pilot_decision_v2.json
```

The expected v2 result is also `passed=false` with no selected action. All
native arms time out, so the PAR-2 ratios and confidence intervals are
undefined; this is a feasibility failure rather than independent performance
confirmation.

## Rerun SCIP Pilot

Only rerun after verifying that all five MPS files match the hashes in the plan:

```bash
PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.causal_harness \
  action-oracle-suite \
  "$MIPLIB_INSTANCES_DIR/exp-1-500-5-5.mps.gz" \
  "$MIPLIB_INSTANCES_DIR/neos-1171448.mps.gz" \
  "$MIPLIB_INSTANCES_DIR/neos8.mps.gz" \
  "$MIPLIB_INSTANCES_DIR/piperout-27.mps.gz" \
  "$MIPLIB_INSTANCES_DIR/swath1.mps.gz" \
  --actions random-rank efficacy-rank adaptive-score \
  --seeds 0 1 2 \
  --time-limit 30 \
  --intervention-scope first-run-only \
  --output-dir experiments/fixed_baseline_train_pilot_v1-rerun \
  --manifest /tmp/causal_fixed_baseline_train_pilot_v1-rerun.json
```

Wall-clock values need not be bit-identical across reruns. Status, bounds,
intervention records, context hashes, timeout handling, and the direction and
uncertainty of aggregate effects must be reported rather than silently replaced
with the committed run.

## Sealed Data Boundary

The official test matrices remain unscored. The fixed-baseline pilot failed on
train, so validation and test are not opened. Reusing those splits to search for
a favorable action would violate `docs/PAPER_PROTOCOL.md`.

## Release Artifact

The deterministic release builder, archive tiers, verification commands,
container recipe, and DOI checklist are documented in
`docs/ARTIFACT_RELEASE.md`. The container is pinned to SCIP 10.0.2 and
PySCIPOpt 6.2.1. The history-free public artifact repository is
<https://github.com/linwenru/scip-root-cut-selection-artifact>. Original
project materials are licensed under Apache-2.0; an archival DOI must still be
supplied before the artifact is described as a versioned archival release.
