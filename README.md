# SCIP Root Cut Selection Audit

This is a clean restart of the SCIP cut-selection project. The immediate goal is
not to train another model from observational CSV files. It is to establish
whether a root-only intervention can produce reproducible solve-level gains over
SCIP's native hybrid cut selector.

Public artifact repository (created from a history-free allowlisted snapshot):
<https://github.com/linwenru/scip-root-cut-selection-artifact>

## Current decision

The pre-registered one-shot learned-policy pilot completed 162 fresh SCIP runs
on 18 previously unseen MIPLIB Group keys. It failed the frozen joint gate: the
Group-equal full-process wall-time PAR-2 ratio was 1.4931 (95% cluster-bootstrap
interval 1.1309--2.1078), and one native-complete run timed out under active
ranking. Learned-policy development is therefore stopped; the original sealed
test remains unopened. See `docs/LEARNED_POLICY_PILOT.md` for the frozen result
and the explicitly post-hoc diagnostic decomposition.

## Research contract

1. Baseline traces may be used for feature analysis and imitation pretraining.
2. `is_applied` is an observational label, not evidence that a cut improves a solve.
3. Online decisions are limited to the root node. Historical experiments allow
   at most once per SCIP run; the strengthened confirmatory protocol uses only
   the first eligible callback of the first run.
4. A policy advances only after paired, multi-seed active validation.
5. The primary validation and test protocol must be family-disjoint from training.

The project reports two evaluation protocols. The primary protocol holds out
official MIPLIB `Group` values to measure unseen-family generalization. A
secondary seen-family protocol measures specialization to new instances from
families represented in training. These results must not be merged into one
headline number.

## Layout

- `vendor/tracer_snapshot/`: immutable snapshot of the tracer used for collection.
- `data/raw/`: machine-local links to the self-generated CSV directories; ignored by Git.
- `data/manifests/`: reproducible inventory and audit results.
- `src/scip_cut_trace_v2/`: audited tooling developed in this project.
- `docs/`: decisions, audit reports, and stage gates.

Commands below use `TRACE_SOURCE_DIR` for the locally generated CSV tree and
`MIPLIB_INSTANCES_DIR` for a local MIPLIB instance directory. These external
inputs are intentionally not redistributed.

## Evaluation protocols

Generate non-copying views of the original traces using official MIPLIB Groups:

```bash
PYTHONPATH=src python3 -m scip_cut_trace_v2.split_protocols
```

The primary `official_group_ood` view measures unseen-family generalization, the
secondary `seen_family` view measures within-family specialization, and the
`officially_ungrouped` view is reported separately. See
`docs/EVALUATION_PROTOCOLS.md` for the exact contract.

## Audit

```bash
python -m scip_cut_trace_v2.audit \
  --source "$TRACE_SOURCE_DIR" \
  --output data/manifests/initial_audit.json \
  --scan-candidates
```

The full candidate scan reads roughly 22 GB and can take several minutes. Omit
`--scan-candidates` for a fast inventory based on summaries, headers, SCIP
statistics, and separation-round transitions.

## Root observational view

Build the leakage-safe, per-instance compressed root tables with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m scip_cut_trace_v2.observational \
  --source "$TRACE_SOURCE_DIR" \
  --resume
```

The builder reconstructs decision-time score ranks, separates exact duplicate
cuts from detected 32-bit source-ID collisions, excludes forced cuts from the
ranking candidates, and enforces at most one eligible intervention per SCIP run.
See `docs/ROOT_OBSERVATIONAL_DATA.md` for the schema and full-build audit.

## Ranking imitation matrices

Build the train-only feature contract and instance-balanced ranking matrices
with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m scip_cut_trace_v2.ranking_dataset
```

The resulting NPZ files use one eligible root decision as one ranking query.
They support an SCIP-imitation baseline only: `observed_logical_is_applied` is
not a causal solve-benefit label. Test statistics remain sealed. See
`docs/RANKING_IMITATION_DATA.md` for the 44-feature contract, query weights, and
the rationale for retaining at most one intervention per run.

## XGBoost imitation baseline

Train the fixed-configuration model and run the official-Group cross-validation
diagnostic with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m scip_cut_trace_v2.ranking_train

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m scip_cut_trace_v2.ranking_crossval
```

The external validation stage gate currently rejects direct online use even
though the Group-disjoint cross-validation finds a modest imitation signal. See
`docs/IMITATION_BASELINE.md` for the complete results and
`docs/IMITATION_ONLINE_AUDIT.md` for the feature-contract reasons it remains an
offline diagnostic.

## Online causal validity gate

Before testing any learned intervention, compare native SCIP with an exercised
high-priority no-op cut selector in fresh paired processes:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m \
  scip_cut_trace_v2.causal_harness noop-parity \
  /path/to/instance.mps.gz --seeds 0 1 2 --time-limit 300
```

This gate checks structural complete-B&B parity and reports, but does not gate
on, wall-clock time and primal-dual integral. See `docs/CAUSAL_HARNESS.md`.

Build the SCIP 10.0.2 native-hybrid bridge and run its parity gate with:

```bash
SCIPOPTDIR=/opt/homebrew .venv/bin/pip install .

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m \
  scip_cut_trace_v2.causal_harness direct-hybrid-parity \
  /path/to/instance.mps.gz --seeds 0 1 2 --time-limit 300
```

The experimental `boundary-swap-pair` and `boundary-swap-suite` commands apply
at most one root intervention per SCIP run. The initial fixed action has strong
causal effects but is rejected because the training suite contains a catastrophic
timeout regression.

The next training-only diagnostic is `action-oracle-suite`. It evaluates a
three-action, predeclared root library against a shared native baseline and
reports a post-hoc LP-iteration oracle as an upper bound. Oracle choices are not
a policy, and action-level safety failures remain gating evidence.

The five-instance pilot found a positive post-hoc oracle upper bound, but its
leave-one-seed-out action rule regressed and incurred an unsafe timeout. The
current action library is therefore rejected as a deployable policy; see the
causal-harness document for the full distinction between opportunity and
predictability.

For causal data identification, the harness also supports
`--intervention-scope first-run-only`. It records one leakage-safe pre-action
root context and never intervenes after a restart. The first pilot confirmed
matching contexts across action arms, but its cross-seed selector still failed;
the resulting 15 contexts are a schema validation set, not training data.

Build the deduplicated active records with
`scip-cut-trace-v2-causal-data`. One record contains a shared pre-action context
and all native/treatment outcomes for one instance and seed. See
`docs/CAUSAL_DATASET.md`. The pilot and two pre-registered cohorts currently
provide 64 attributable contexts from 22 training instances.

The frozen leave-one-instance-out risk diagnostic expands these contexts to 188
eligible context-action rows:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m \
  scip_cut_trace_v2.causal_risk
```

Its six-part gate fails: the model does not generalize the nine unsafe outcomes
across the four affected instances, and even the strongest structural baseline
must reject most safe actions to catch every unsafe one. No learned action is
allowed online. See `docs/CAUSAL_RISK_DIAGNOSTIC.md` for the frozen protocol,
results, and continuation boundary.

The independent single-action continuation also fails. On cohort C,
`efficacy-promote` has a small positive instance-equal LP-iteration mean but
produces one attributable timeout and only 3 wins against 3 losses, with 2
ineligible-instance ties. The project therefore stops online policy development
at the native SCIP fallback instead of starting another model iteration. See
`docs/PROJECT_STATUS.md` for the consolidated evidence and reopening criteria.

## Paper-strengthening protocol

`docs/PAPER_PROTOCOL.md` freezes the research questions, first-run intervention
contract, baselines, timeout handling, instance-equal aggregation, and stopping
rules before new validation or test results are inspected.

The causal harness now exposes three additional fixed arms through
`action-oracle-suite`: `random-rank`, `efficacy-rank`, and `adaptive-score`.
Each keeps SCIP native hybrid's selected-cut count and leaves forced cuts
untouched. `adaptive-score` independently ports the Turner et al. scoring and
parallelism-filter family; it is not presented as a reproduction of their GCNN.
The command keeps the historical three-action default unchanged, so fixed arms
must be named explicitly.

Generate the frozen instance-equal PAR-2 analysis for a causal suite with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m \
  scip_cut_trace_v2.paper_statistics \
  data/manifests/causal_action_efficacy_first_run_cohort_c_v1.json \
  --actions efficacy-promote \
  --bootstrap-replicates 10000 \
  --output data/manifests/paper_statistics_cohort_c_development_v1.json
```

The cohort C development replay still fails: one of eight attributable
instances has a safety failure, and the penalized-time ratio confidence interval
crosses `1.0`. This retrospective analysis does not convert cohort C into new
confirmation evidence.

See `docs/REPRODUCIBILITY.md` for the exact environment, test, statistics,
pilot-rerun, and sealed-data commands. The machine-readable evidence index maps
each study claim to hashed manifests, implementation code, and active-arm
results. Private manuscript files are deliberately outside the public artifact.

The expanded publication experiment is now complete: all 480 planned arms for
40 training instances and three seeds were collected. Random rank and efficacy
rank are slower on average and have attributable safety failures. Adaptive
score has no observed attributable failure and more wins than losses, but its
PAR-2 treatment/native time ratio is 1.0019 with a 95% instance-clustered
interval of [0.8481, 1.1486]. No fixed action passes the frozen advancement
gate, so validation and test remain sealed.

`docs/ARTIFACT_RELEASE.md` describes deterministic core and observational-data
archives, per-file SHA-256 verification, the SCIP 10.0.2 container, and the
remaining author-controlled steps for assigning an archival DOI.

## License

Original project code, documentation, and generated research outputs are
released under the Apache License 2.0. See `LICENSE` and `NOTICE`. SCIP and
MIPLIB materials are not relicensed by this repository; SCIP is installed as
an external dependency, and MIPLIB instance files are not redistributed.
