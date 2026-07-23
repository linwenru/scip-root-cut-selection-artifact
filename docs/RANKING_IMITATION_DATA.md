# Instance-balanced ranking imitation data V1

## Scope

This stage prepares a reproducible learning-to-rank baseline from passive SCIP
traces. It answers whether the recorded pre-decision features can imitate which
logical cuts SCIP applied. It does not answer whether replacing SCIP's ranking
improves a complete solve.

One eligible root cut-selector decision is one ranking query. The rows in that
query are its logical optional cuts, and `observed_logical_is_applied` is the
binary imitation label. Forced cuts, ambiguous source-ID collisions, post-state
fields, and outcome deltas are excluded from the model input.

## Intervention budget

The current online research contract permits at most one intervention per SCIP
run. This is not hard-coded to `sep_round_node == 1`: the observational builder
selects the first root decision in each run that has a pre-state, at least two
logical optional candidates, and no detected cut-ID collision.

Of the 318 eligible decisions, 311 occur at node separation round 1. The seven
exceptions occur at rounds 2 (four decisions), 3 (two), and 10 (one). If every
otherwise qualified root round were retained, the same traces would contain
8,764 decisions across 318 runs; 308 runs would have more than one decision, and
one run would have 383. Those later rounds are baseline-policy observations.
After an earlier ML intervention, their LP state and candidate set may change,
so they are not valid on-policy evidence for repeated intervention.

Repeated root intervention is therefore deferred. The first active experiment
will use one intervention per run. Only after paired multi-seed solve-level
validation succeeds should the budget be expanded incrementally, with new
trajectories collected under the expanded policy.

## Dataset partitions

The matrices follow the committed MIPLIB Group protocols. Test labels are stored
for a future final evaluation but their statistics and metrics remain sealed.

| Matrix | Instances | Queries | Candidates | Applied labels |
|---|---:|---:|---:|---:|
| `train` | 142 | 215 | 162,776 | 18,265 |
| `official_group_ood_val` | 20 | 27 | 9,597 | 2,002 |
| `official_group_ood_test` | 14 | 22 | 3,647 | sealed |
| `seen_family_val` | 5 | 11 | 2,407 | 512 |
| `seen_family_test` | 10 | 12 | 21,720 | sealed |
| `officially_ungrouped_val` | 7 | 13 | 17,505 | 3,084 |
| `officially_ungrouped_test` | 7 | 18 | 2,240 | sealed |

## Instance balancing

Raw candidate rows are severely instance-dominated. In training,
`k1mushroom.mps.gz` contributes 25.9995% of all candidates, the largest five
instances contribute 59.0124%, and the candidate-volume effective instance
count is only 9.35 despite 142 represented instances.

XGBoost ranking weights are query weights rather than row weights. For query
`q` belonging to instance `i`, the matrix therefore stores:

```text
w_q = number_of_queries / (number_of_instances * queries_for_instance_i)
```

This makes every instance contribute the same total query weight while keeping
the mean query weight equal to one. `group_weight` covers all queries.
`group_has_effective_pair` marks queries containing at least one positive and
one negative label, and `effective_group_weight` applies the same balancing only
to those informative queries. Training should use that mask and effective
weight; 204 of the 215 training queries are informative.

## Feature contract

The feature contract is fit only on eligible decisions from the original
training split. It contains 44 encoded columns:

- 40 non-constant numeric or Boolean pre-decision features;
- four one-hot `origin_type` columns: `CONS`, `CONSHDLR`, `SEPA`, and `<UNK>`.

`is_local`, `is_modifiable`, `pre_state_available`, and `n_open_nodes_pre` are
constant or empty in training and are dropped. `pre_lp_status` is also dropped
because only `OPTIMAL` occurs in training. Unknown validation or test origin
types map to `<UNK>`. Missing numeric values are represented as `NaN`; the
explicit `cutoff_distance_available` feature preserves the known availability
signal.

## NPZ schema

Each compressed matrix contains:

- `X`, `y`: feature matrix and observational imitation labels;
- `qid`, `group_ptr`, `group_sizes`: ranking-query boundaries;
- `group_weight`, `effective_group_weight`, `group_has_effective_pair`;
- `feature_names` and candidate identities;
- baseline score rank and query-level instance, run, split, and Group metadata.

The generated NPZ files are ignored by Git because they are reproducible. Their
paths, sizes, SHA-256 values, feature contract, partition statistics, and checks
are committed in `data/manifests/ranking_imitation_v1.json` and
`data/manifests/eligible_ranking_analysis_v1.json`.

## Build

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m scip_cut_trace_v2.ranking_dataset
```

All seven matrices must pass partition accounting, feature-boundary, test-seal,
and instance-balance checks before the manifest is written.
