# Online Audit Of The Imitation Ranker

## Decision

The existing model at
`models/ranking_imitation_xgb_v1/model.ubj` remains an **offline imitation
diagnostic**. It is not an admissible online treatment arm under the frozen
paper protocol.

This is a methodological exclusion, not a claim that XGBoost cannot run in a
SCIP callback.

## What The Model Learns

The row label is whether a logical candidate was observed as applied by SCIP.
The model therefore learns to imitate native SCIP decisions. It does not learn
a causal solve-level reward such as lower penalized solving time, fewer nodes,
or lower primal-dual integral.

Even perfect imitation is expected to recover native behavior, not establish a
mechanism for outperforming native behavior. Offline NDCG and overlap are
useful tracer and representation diagnostics, but they are not evidence of a
solve-level gain.

## Existing Evidence

- The model uses 44 encoded features and was trained on 162,776 candidate rows,
  215 ranking queries, and 135 train instances.
- Its official-Group-OOD validation point estimates improve over the hybrid-
  score rank baseline, but the predeclared NDCG@10 confidence gate fails.
- The training manifest therefore already states: do not start active SCIP
  intervention from this model.
- No official test matrix was scored.

These facts are preserved. The model is not retuned on validation.

## Feature-Contract Audit

### Available before intervention

The current causal callback can capture coefficient statistics, sides and
constant, efficacy, objective parallelism, directed cutoff distance when an
incumbent exists, integer support, row flags, origin type, LP dimensions and
objective, LP iterations, bounds, gap, applied-cut count, current node, current
node separation rounds, candidate count, and run number.

### Derivable but not yet parity-verified

`score`, `score_rank_pre`, and `score_rank_fraction_pre` can be recomputed from
SCIP hybrid parameters before selection. They are legal pre-action features,
but a replay-versus-live numeric parity test is required before using them in a
model. They must not be replaced with `native_rank`, which is the output of
native selection after parallelism filtering.

### Not equivalent in the current online representation

- `sep_round_run` and `sep_round_global` require restart-aware counters;
- `lp_round_node`, `lp_round_run`, and `lp_round_global` require the original
  event semantics rather than substituting total LP count;
- `n_cuts_generated_node_pre` requires a matching generation event tracker;
- offline rows are logical candidates deduplicated by signature, whereas the
  live callback receives candidate occurrences and must return an occurrence
  ordering;
- the training set contains first eligible root decisions from several SCIP
  runs, while the confirmatory intervention scope is first-run-only.

For a first run some counters may happen to be numerically equal. Coincidence is
not an acceptable feature contract.

## Leakage Guard

The live causal context records `native_rank` and `native_selected` solely to
audit the native budget and the treatment delta. Neither field may be an ML
input. Likewise, no post-LP, terminal, elapsed-time, applied-label, or treatment
outcome field may enter the online matrix.

## Requirements For Future Admission

An imitation arm may be reconsidered only after all of the following are
implemented and frozen on train data:

1. a first-run-only training view matching the online intervention scope;
2. a deterministic logical-candidate-to-occurrence mapping that preserves the
   native selected-cut budget;
3. an explicitly reduced online feature list, with unavailable counters removed
   before training rather than imputed at inference;
4. replay-versus-live feature parity tests on raw callback snapshots;
5. retraining from scratch without validation or test labels;
6. a passed official-Group-OOD offline gate;
7. active evaluation as a fixed policy under the same paired complete-B&B
   protocol as heuristic baselines.

Until then, the scientifically stronger comparison is native SCIP versus the
frozen random, efficacy, and Adaptive-score fixed baselines. The imitation
model remains evidence about observational learnability and the gap between
imitation quality and causal utility.
