# Initial audit of the supplied 233-instance corpus

## Gate decision

The corpus is useful, but it is not a clean causal training set. Preserve it as
an immutable observational baseline. Build cleaned root-round views from it for
feature analysis and optional imitation pretraining, then collect paired active
intervention outcomes for the actual policy target.

## Confirmed strengths

- All 233 instance directories contain the six expected output files.
- Split counts are 163 train, 35 validation, and 35 test.
- The passive Python cut selector and native SCIP hybrid selector have equal call
  and root-call counts on all 233 instances.
- The maximum observed run across candidate/applied/LP files agrees with SCIP's
  reported run count whenever any event was recorded.
- There are 67 multi-run instances, so the run dimension is material and is
  represented in the supplied data.
- The corpus contains 53,233,883 candidate rows across 533,056 cut-selection
  rounds. Of these, 8,455,696 rows occur at root nodes.

## Confirmed limitations

- Solver outcomes: 71 optimal, 160 time limit, and 2 infeasible. Most traces are
  fixed-budget partial trajectories, not completed solves.
- `is_selected` is empty for every ordinary candidate. Only forced cuts are true.
  The usable observational label is the post-hoc `is_applied` match.
- `is_applied` is true for 5,152,675 candidates. There are 65,437 duplicate-ID
  rows in 755 decision rounds, including 7,631 duplicate rows marked applied.
- `rank` is overwritten after solving so applied cuts come first. It is not a
  decision-time feature and would leak the target when predicting `is_applied`.
- `cut_id` uses Python's randomized 32-bit `hash()`. It is not stable across
  processes and has ambiguous duplicates within some decision rounds.
- `coeff_sparsity_ratio` is computed as `nnz / len(nonzero_columns)`, making it
  identically one in all 53,233,883 candidate rows.
- `cutoff_distance` is absent in 4,834,018 rows. The cleaned schema needs an
  explicit incumbent-availability indicator instead of silently imputing it.
- The supplied schema records row origin type but not the producing separator or
  constraint-handler name.
- The original split intentionally mixes seen-family and unseen-family cases.
  The supplied split generator does not preserve MIPLIB's official `Group`
  column, so the distinction was not reproducible from its metadata alone.
- A name-only heuristic flags 19 cross-split candidates, but this is not the
  authoritative family definition. The official MIPLIB metadata identifies 15
  cross-split Groups among the 233 instances.
- The official cross-split Groups are `app`, `blp`, `bnatt`, `cryptanalysis`,
  `csched`, `dano`, `gmu`, `map`, `physiciansched`, `radiation`, `rmatr`,
  `rococo`, `sing`, `sp9`, and `uccase`.
- Validation contains 5 seen-family, 22 unseen-family, and 8 officially
  ungrouped instances. Test contains 10, 17, and 8 respectively.
- The corpus does not record exact SCIP/PySCIPOpt versions, random seed, complete
  parameter settings, instance checksum, or the Git revision that produced each
  output.
- Observed application is not a counterfactual benefit label. It can support
  imitation, not a claim that changing SCIP's decision improves the solve.

## V2 data rules

1. Treat current `rank` as post-outcome metadata; reconstruct `score_rank_pre`
   from `score` within each decision round.
2. Drop or collapse decision rounds with duplicate `cut_id` before using
   `is_applied` as an imitation target.
3. Do not use `solving_time`, post-round deltas, or any applied-derived fields as
   decision-time model inputs.
4. Report two separate protocols: official-Group-disjoint validation/test for
   unseen-family generalization, and a secondary seen-family evaluation. Never
   merge them into one headline metric. Existing traces can be reassigned or
   stratified without re-solving because all source splits used the same policy.
5. Weight or cap instances/rounds so a handful of very large traces cannot
   dominate training.
6. Define the final label from paired active solve outcomes, not from SCIP's
   baseline selection decision.
