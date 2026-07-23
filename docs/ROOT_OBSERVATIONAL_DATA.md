# Root observational dataset V1

## Purpose

This dataset is a leakage-safe view of the supplied passive traces. It supports
feature analysis and optional imitation pretraining. It does not establish that
an applied cut improved the solve, so it is not the final causal policy dataset.

The builder reads all 53,233,883 source candidate rows but writes only root-node
decisions. Raw traces are never modified.

## Tables

Each instance has two gzip-compressed CSV files under
`data/processed/root_observational_v1/<split>/<instance>/`:

- `root_decisions.csv.gz` has one row per root cut-selector callback. It contains
  decision identity, candidate counts, policy eligibility, and only pre-decision
  LP state.
- `root_candidates.csv.gz` has one row per logical candidate. Exact duplicate
  occurrences are collapsed, the original post-hoc `rank` is discarded, and
  `score_rank_pre` is reconstructed from the decision-time hybrid score.

`observed_logical_is_applied` is an observational imitation label. It must never
be used as an input feature. Post-state and delta columns are not emitted.

## Identity repair

The source `cut_id` is a process-randomized 32-bit Python hash. Within one
decision, the builder first groups by `source_cut_id` and then by a SHA-256 digest
of every available cut name and feature. This produces two different cases:

1. Identical signatures are duplicate occurrences and collapse to one logical
   candidate with `candidate_multiplicity > 1`.
2. Different signatures under one source ID are detected hash collisions. Both
   candidates are preserved, `source_cut_id_collision` and
   `observed_label_ambiguous` are true, and the decision is ineligible.

The digest cannot recover the missing coefficient vector, so
`logical_candidate_id` is stable for this trace snapshot, not across a new SCIP
run.

## Online eligibility

At most one decision per SCIP run is eligible. It is the first root decision in
that run which:

- has a recorded pre-decision LP state;
- contains at least two logical optional candidates;
- has no detected source cut-ID collision.

Forced cuts are counted at decision level but excluded from ranking candidates.

## Full-build audit

Builder revision 3 produced:

| Metric | Count |
|---|---:|
| Instances | 233 |
| Source candidate rows | 53,233,883 |
| Root rows including forced cuts | 8,455,696 |
| Ordinary root occurrences | 8,454,995 |
| Forced root cuts | 701 |
| Root decisions | 9,103 |
| Logical candidates | 8,414,571 |
| Exact duplicate occurrences collapsed | 40,424 |
| Decisions with exact duplicates | 357 |
| Detected 32-bit cut-ID collisions | 137 |
| Candidates affected by collisions | 274 |
| Policy-eligible decisions | 318 |
| Policy-eligible candidates | 219,892 |
| Applied labels in eligible decisions | 26,544 |

All eight accounting, state-availability, collision-exclusion, and feature
boundary checks passed. There are 25 instances without root candidates and 28
without an eligible decision. Exact names and per-instance counts are in
`data/manifests/root_observational_v1.json`.

## Build

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m scip_cut_trace_v2.observational \
  --source "$TRACE_SOURCE_DIR" \
  --resume
```

The processed directory is reproducible and ignored by Git. The committed JSON
manifest contains the schema, allowed model fields, prohibited fields, protocol
counts, per-instance summaries, and quality checks.
