# Active causal dataset

The active dataset uses one row per `instance_id x SCIP seed`, not one row per
action arm. Each row contains exactly one pre-intervention first-run context and
the complete-B&B outcomes of native SCIP and every predeclared action.

## Context boundary

The model context contains only information available after native hybrid has
ranked the first eligible root candidate set and before any treatment changes
that set. It includes:

- run, callback, node, LP, bound, candidate-count, and selected-count state;
- the native hybrid candidate order and selected boundary;
- row bounds, coefficient statistics, origin, efficacy, objective parallelism,
  integer support, and row flags;
- forced rows, when present.

It does not contain solve time, final status, final bounds, primal-dual integral,
or any post-action outcome. Cut names remain for audit only and are not a model
feature contract.

## Labels

Each action label records whether the pair was safe, eligible, and valid, plus
native and treatment values, deltas, and relative savings for LP iterations,
nodes, LP count, applied cuts, solve time, and primal-dual integral. An unsafe
pair is a safety/risk label; its censored metric delta must not be treated as an
ordinary regression target.

`post_hoc_oracle` is retained for diagnosis. It is not a deployable label: it
uses the held-out action outcomes to choose an action.

## Build

```bash
.venv/bin/scip-cut-trace-v2-causal-data \
  --source-manifest data/manifests/causal_action_oracle_first_run_pilot_v1.json \
  --output data/processed/causal_first_run_pilot_v1.jsonl.gz \
  --manifest data/manifests/causal_first_run_dataset_v1.json \
  --split train
```

The pilot produces 15 records from 5 training instances and 3 seeds. Its 45
treatment arm contexts collapse to 15 shared contexts. Candidate counts range
from 17 to 787, and the compressed JSONL is about 193 KB. Repeated builds are
byte-identical.

This pilot validates the schema only. Fifteen contexts are insufficient for
model fitting or threshold selection. The next active cohort is pre-registered
in `data/manifests/causal_first_run_cohort_a_plan_v1.json` before its solve
outcomes are generated.

## Cohort A

Cohort A pre-registered nine additional training instances from nine distinct
official MIPLIB Groups. Selection required a successful source trace, at most
30 seconds of source trace elapsed time, at least one observationally eligible
root decision, and an available local MPS file. Outcomes were not inspected
until the plan had been committed.

The active run produced 27 instance-seed pairs. One `hypothyroid-k1` seed had no
cut-selector callback in any action arm and was excluded as a no-decision pair,
leaving 26 records. All 26 observed contexts matched across actions. Candidate
counts range from 7 to 4051 and the compressed dataset is about 621 KB.

Action-label counts are:

- `boundary-swap`: 25 valid records, 14 positive LP-iteration savings, and one
  unsafe `cbs-cta` timeout;
- `boundary-swap-2`: 24 valid records and 10 positive savings;
- `efficacy-promote`: 24 valid records and 13 positive savings.

The cohort's post-hoc oracle reports +24.35% instance-equal mean relative
LP-iteration saving. Its leave-one-seed-out selector still fails with -16.39%,
3 instance wins, 1 tie, 5 losses, and one unsafe held-out seed. Positive action
labels therefore exist, but they are not yet predictably selectable.

Pilot plus cohort A contain 41 contexts from 14 training instances. This remains
too small for a credible XGBoost action or abstention model, especially with
unsafe outcomes represented by only two instance-seed-action records.

## Cohort B And Combined Train Data

Cohort B pre-registered eight slower training instances whose source trace
elapsed times were greater than 30 and at most 60 seconds. All 24
instance-seed contexts matched across actions and every treatment recorded one
first-run context and at most one intervention.

Ten raw treatment arms were unsafe. Nine came from all three actions on all
three seeds of `neos-3004026-krka`; one came from `boundary-swap-2` on
`fhnw-binpack4-4`. The third `neos-3004026-krka` native arm also timed out, so
that instance-seed is excluded from causal action labels rather than being
misclassified as action-induced risk. Cohort B therefore contributes 23
attributable contexts.

Its post-hoc oracle reports +13.36% mean relative LP-iteration saving. The
leave-one-seed-out rule is nearly neutral at -0.03%, with 4 instance wins, 3
ties, 1 loss, and one unsafe held-out seed. Near-zero mean does not satisfy the
safety gate.

The combined train dataset merges the pilot and cohorts A and B with source
hashes and cross-source duplicate checks. It contains:

- 64 contexts from 22 training instances;
- 1 skipped no-decision pair and 1 skipped native-timeout pair;
- candidate counts from 7 to 4051;
- 192 action labels, of which 9 are attributable unsafe outcomes across 4
  instances;
- 29, 24, and 33 positive LP-saving labels for `boundary-swap`,
  `boundary-swap-2`, and `efficacy-promote` respectively.

This is sufficient for a preliminary instance-grouped risk separability
diagnostic. It is not sufficient for deployment threshold selection or a claim
of cross-family safety. Additional active cohorts should be chosen only after
that diagnostic, not by continuing an unconditional sweep.
