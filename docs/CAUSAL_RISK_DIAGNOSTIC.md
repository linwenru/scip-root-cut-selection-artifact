# Causal Risk Separability Diagnostic

## Question

Before collecting another active cohort, test whether a leakage-safe
pre-intervention context contains a generalized signal for unsafe declared root
actions. This is a diagnostic of the safety-gate premise, not an online policy
or a deployment threshold experiment.

## Unit And Label

One row is one eligible `(instance, seed, first-run context, action)` tuple.
The positive label is an unsafe treatment outcome relative to a completed
native SCIP arm. Contexts whose native arm did not complete are excluded by the
causal dataset builder, because their treatment risk is not attributable.

The three declared actions remain fixed:

- `boundary-swap`;
- `boundary-swap-2`;
- `efficacy-promote`.

## Feature Contract

Features are derived only from the shared pre-intervention context and the
identity of the proposed action. They include root LP dimensions and progress,
native selection budget, compact selected/unselected candidate summaries, and
the structural difference between the cut removed and the cut added by that
action.

The model does not receive instance identity, seed, solve time, native final
outcome, treatment outcome, post-action metrics, or oracle labels. Missing and
non-finite values become `NaN` and use XGBoost's native missing-value handling.

## Evaluation Contract

Use leave-one-instance-out evaluation. Every seed and every action from a held
instance stays in the held-out fold, preventing seed-level leakage. Fit a
fixed, shallow XGBoost binary classifier for 120 rounds without early stopping
or result-dependent hyperparameter search.

Compare out-of-fold scores with four structural baselines:

- constant unsafe prevalence;
- candidate count;
- native selected-cut count;
- a Laplace-smoothed action unsafe rate learned only from other instances.

Report average precision, ROC AUC, unsafe recall among the highest-risk 5%, 10%,
and 20% of rows, and the fraction of safe rows that must be rejected to recall
all observed unsafe rows.

## Pre-Registered Gate

All checks must pass:

1. Average precision is at least twice unsafe prevalence.
2. ROC AUC is at least 0.75.
3. The top 20% risk region recalls every unsafe row.
4. Full unsafe recall rejects at most 20% of safe rows.
5. Average precision exceeds every structural baseline.
6. The safe-row rejection cost at full recall beats every structural baseline.

Passing only justifies freezing the protocol and evaluating it on a new,
independent active cohort. It does not justify deployment. Failure means the
current active contexts and action library do not support a useful generalized
risk gate; thresholds must not be tuned on the same 22 instances to rescue it.

## Result

The frozen diagnostic was run on 188 eligible context-action rows from 22
instances. Nine unsafe rows came from four instances. All six pre-registered
checks failed:

- XGBoost average precision was `0.0435`, below the unsafe prevalence of
  `0.0479`;
- ROC AUC was `0.2790`;
- the highest-risk 20% recalled only one of nine unsafe rows;
- recalling all unsafe rows required rejecting `98.3%` of safe rows.

The strongest structural baseline was native selected-cut count, with average
precision `0.1532` and ROC AUC `0.7998`. It still had to reject `60.3%` of safe
rows to recall all unsafe rows and therefore also failed the operational safety
criterion. Candidate count was weaker, with average precision `0.0819` and ROC
AUC `0.6592`.

The failure is not evidence that no pre-intervention safety signal can ever
exist. It shows that nine unsafe actions clustered in four instances do not
provide a cross-instance risk rule under the current feature and action
contract. The model scores particularly low on the held-out unsafe patterns,
which is consistent with instance-specific failure modes rather than one shared
risk mechanism.

Do not tune XGBoost, choose a threshold, or reinterpret the stronger structural
baseline on these same 22 instances. Any continuation must first obtain new
active causal outcomes under a separately pre-registered sampling protocol,
with enough independent unsafe instances to evaluate a risk rule. Until then,
the online policy remains native SCIP.

## Single-Action Continuation

The next experiment narrows the action library to `efficacy-promote`, which had
the most positive LP-saving labels in the combined discovery data while using
the same one-cut, fixed-budget intervention contract. This is an explicitly
selected discovery action, so it must be confirmed on instances absent from the
64-context dataset.

Cohort C completely enumerates the eight remaining eligible train instances
whose source trace elapsed time is greater than 60 and at most 120 seconds.
It compares native SCIP with only `efficacy-promote` over three seeds and keeps
the first-run-only intervention scope. The frozen instance list and fixed-action
confirmation gate are recorded in
`data/manifests/causal_first_run_efficacy_cohort_c_plan_v1.json`.

### Cohort C Result

All 48 solve arms completed as processes and all 24 pre-intervention contexts
were present and matched. Two native arms did not finish within 120 seconds, so
22 pairs can attribute treatment safety. One attributable pair,
`n2seq36q` seed 1, has a completed native arm and a treatment timeout. The two
other raw `safe=false` comparisons cannot attribute treatment harm: one has a
native timeout and completed treatment, and one has both arms time out.

The fixed action has a `+2.17%` instance-equal mean relative LP-iteration saving
when ineligible instances fall back to native and benefit is measured on valid
eligible pairs. It produces 3 instance wins, 2 fallback ties, and 3 instance
losses. The worst valid instance mean is `-18.90%`.

The pre-registered confirmation gate therefore fails two decisive checks:
there is one attributable unsafe treatment and instance wins do not outnumber
losses. The positive `+7.43%` post-hoc oracle figure in the experiment manifest
is not the fixed-action result; it lets native SCIP win each seed-level tie or
loss and is only an opportunity upper bound.

`efficacy-promote` is rejected as a generally safe fixed action. Together with
the failed grouped risk diagnostic, this also leaves too little independent
unsafe-instance support for another ML gate: the new action-specific unsafe
outcome adds only one instance. The current evidence does not justify another
model iteration or threshold search.
