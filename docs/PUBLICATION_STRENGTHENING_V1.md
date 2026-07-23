# Publication Strengthening Protocol V1

## Purpose

This protocol strengthens the negative-result study without reopening model
tuning or inspecting validation/test outcomes. The independent statistical
unit remains the MPS instance; paired SCIP seeds are repeated measurements
within an instance.

The machine-readable pre-registration is
`data/manifests/publication_strengthening_plan_v1.json`. The active wall-time
execution amendment is frozen separately in
`data/manifests/publication_active_plan_v1.json` so the completed parity plan
hash remains immutable.

## Online-compatible learned baseline

The original 44-feature XGBoost model cannot be executed online under an
equivalent feature contract. Seven trace fields have no reliable like-for-like
value in the live cut-selection callback:

- node, run, and global separation-round counters;
- node, run, and global LP-round counters;
- the trace event handler's node-local generated-cut counter.

The online dataset mechanically removes those seven fields and retains 37
pre-action features. Candidate hybrid scores, score ranks, logical-row
deduplication, row structure, and solver state are recomputed in the callback.
The model, dataset manifest, and feature order are all hash-checked before an
arm can run.

The retrained model has positive point estimates on all three validation
strata, but its primary official-Group-OOD NDCG@10 interval crosses zero. It
therefore fails the pre-existing offline authorization gate. The
`xgb-imitation-rank` arm is limited to a single-instance integration smoke test
and cannot enter the publication performance experiment. This is a protocol
decision, not an implementation failure.

## Multi-instance parity

The parity cohort contains 12 training instances from 12 distinct official
MIPLIB Group keys. It uses seeds 0, 1, and 2 and a 60-second limit. Each native
arm is paired with:

- an exercised no-op selector;
- a direct call to SCIP's native hybrid selector.

Every status, bound, gap, node count, LP iteration count, LP count, and applied
cut count must match. Wall time and primal-dual integral are reported but remain
non-gating callback-overhead measurements.

## Expanded active cohort

The active cohort contains 40 unique training instances from 40 distinct Group
keys. Selection uses only source trace runtime, official Group, eligibility
counts, and instance ID:

- 30 completion-enriched instances are the first 30 distinct Group keys in
  source-runtime order;
- 10 hardness-coverage instances are selected from ten equal-rank bins over
  the remaining runtime-ordered pool.

The fixed arms are deterministic random rank, efficacy rank, and the
independently ported Turner et al. adaptive-score family. Every arm preserves
SCIP's native selected-cut count and may change only the first eligible root
callback of the first SCIP run. The experiment uses seeds 0, 1, and 2, a
300-second limit, one solver thread, and one fresh process per arm.

Only one SCIP arm runs at a time. Within each of the 120 instance-seed blocks,
the four arms follow one of four balanced Williams/Latin sequences. Blocks are
assigned to sequence rows by a frozen SHA-256 key, so every arm appears exactly
30 times in each execution position. This prevents concurrent CPU contention
and balances systematic run-order drift in the wall-time outcome.

This is a precision-strengthening estimate on the training split. Some fast
instances appeared in earlier development cohorts, but no previous treatment
outcome enters the deterministic selection rule. The expanded cohort is not an
independent validation set and cannot authorize opening validation or test.

## Resumption and execution

Each `instance x seed x arm` result is an individual JSON file. On resume, the
runner verifies the instance hash, seed, limits, runtime parameters,
intervention scope, and learned-model hash when applicable. Only missing or
explicitly non-reused arms are executed.

```bash
PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.publication_protocol \
  run-parity --reuse-existing --jobs 2

PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.publication_protocol \
  run-active --reuse-existing --jobs 1
```

The validation and test splits remain sealed throughout V1.

## Parity result

The pre-registered all-pair gate did not pass. For each candidate arm, 34 of 36
pairs passed every structural check and 10 of 12 instances passed on all three
seeds. The two non-passing pairs had distinct feasibility causes:

- `hypothyroid-k1`, seed 1, completed in all arms but did not invoke any cut
  selector callback, so neither callback implementation was exercised;
- `neos-1582420`, seed 2, reached the 60-second limit in native, no-op, and
  direct-hybrid arms; wall-clock truncation produced different partial search
  counts.

Among the 34 complete pairs per arm in which the callback was exercised,
no-op and direct-hybrid both reproduced every gated structural field in 34 of
34 cases. There were zero complete, exercised structural mismatches. This
diagnostic subset does not overwrite the failed all-pair gate, and no instances
are replaced post hoc.

## Expanded active result

The frozen active queue completed all 480 planned arms: 40 instances, three
seeds, and four arms per instance-seed block. The output contains exactly 120
records for each of native, random rank, efficacy rank, and adaptive score, with
no missing, extra, unparsable, or internally mismatched records. Every one of
the 113 root contexts observed by a treatment arm has the same hash across all
three treatment actions. Seven instance-seed blocks have no eligible treatment
context, and none is partially observed.

The pre-registered primary analysis uses PAR-2 wall time, aggregates seeds
within each instance, gives every instance equal weight, and resamples
instances with 10,000 cluster-bootstrap replicates. Native SCIP completes at
least one seed on 31 of the 40 instances; the other nine instances remain
reported as native-incomplete rather than being used for treatment attribution.

| Action | Actual interventions | Fallback pairs | PAR-2 time ratio | 95% interval | Instance W/L | Safety failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random rank | 113 | 7 | 1.1114 | [0.9880, 1.3323] | 13/18 | 2 pairs, 1 instance |
| efficacy rank | 103 | 17 | 1.0647 | [0.9156, 1.2979] | 15/16 | 3 pairs, 2 instances |
| adaptive score | 62 | 58 | 1.0019 | [0.8481, 1.1486] | 18/13 | 0 pairs, 0 instances |

No action passes all three frozen gates. Random and efficacy ranking have more
instance losses than wins, confidence intervals that include no improvement,
and attributable status failures. Adaptive score has more wins than losses and
no observed attributable failure, but its point estimate is neutral and its
confidence interval is wide. Its zero observed failures also has a one-sided
95% instance-failure upper bound of 0.0921, so this cohort cannot establish a
deployment-level zero-risk claim.

The expanded experiment therefore strengthens the negative result: a single
root intervention has causal leverage, but none of the fixed ranking families
establishes a safe solve-level advantage over native SCIP. Validation and test
remain sealed.
