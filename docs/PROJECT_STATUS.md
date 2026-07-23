# Project Status

## Goal

The project asks whether an ML-controlled root cut-selection intervention can
produce reproducible solve-level gains over SCIP's native hybrid selector while
remaining safe on unseen instances and families.

The current answer is **not demonstrated**. Native SCIP remains the only
authorized online policy.

## Evidence Reached

The restart established several reusable foundations:

- the self-generated trace data and duplicate-cut behavior were audited;
- official MIPLIB Group-disjoint and seen-family evaluation protocols were
  separated;
- leakage-safe root observational and ranking-imitation datasets were built;
- native no-op and direct-hybrid parity were checked in isolated processes;
- first-run-only root interventions were captured with matching pre-action
  contexts;
- paired, multi-seed, complete-B&B causal outcomes were collected for a fixed
  action library;
- active causal records and instance-grouped risk diagnostics were made
  reproducible.

These are infrastructure and negative-result contributions. They do not imply
an ML solve-level improvement.

## Decisive Negative Results

1. The three-action library has positive post-hoc oracle opportunities, but
   fixed actions and leave-one-seed-out choices produce regressions and
   treatment timeouts. Oracle opportunity is not a deployable policy.
2. The frozen leave-one-instance-out risk model fails all six checks. Its
   average precision is below unsafe prevalence, ROC AUC is `0.279`, and full
   unsafe recall rejects `98.3%` of safe rows.
3. The stronger native-selected-count baseline is still unusable: full unsafe
   recall rejects `60.3%` of safe rows.
4. The independent cohort C confirmation of the single selected action,
   `efficacy-promote`, fails. Among 22 attributable pairs it causes one
   treatment timeout after a completed native solve. Its fixed-action result is
   `+2.17%` mean relative LP-iteration saving with 3 instance wins, 2 fallback
   ties, and 3 losses.
5. The `+7.43%` cohort C oracle is only a post-hoc upper bound that lets native
   SCIP win each seed-level loss or tie. It must not be reported as the fixed
   action result.

## Decision

Stop the current objective of training and deploying an ML cut-selection policy.
Do not tune another XGBoost model, threshold, action selector, or abstention gate
on the same active instances. That would reuse failed evidence and create an
optimistic model version without new causal support.

This is not a proof that no ML cut selector can ever outperform SCIP. It is a
finding that the current one-cut root action family, available pre-action
features, and active sample support do not yield a safe generalized policy.

## Reopening Criteria

Online ML work should reopen only after an intervention design or new external
dataset supplies all of the following without tuning on the current cohorts:

- a pre-registered action with no attributable safety failure on an independent,
  multi-seed active cohort;
- more instance wins than losses under instance-equal solve-level aggregation;
- enough independent unsafe and beneficial instances to evaluate a grouped
  safety rule rather than seed duplicates from a few instances;
- a family-disjoint confirmation that remains positive before test data are
  opened.

Until those conditions exist, the scientifically correct output of this repo is
the causal harness, audited datasets, reproducible negative results, and native
SCIP as the online fallback.

## Paper-Strengthening Checkpoint

The negative-result path is now governed by `docs/PAPER_PROTOCOL.md`. It freezes
native SCIP, deterministic random rank, efficacy rank, and an independently
ported Turner et al. Adaptive-score family as fixed baselines. All treatment
arms preserve native selected-cut count and use only the first eligible root
callback of the first run in confirmatory experiments.

The original 44-feature XGBoost ranker remains offline-only. Its label imitates
native application rather than solve benefit, and its official-Group-OOD
confidence gate failed. A separate 37-feature model now removes seven
trace-only counters and matches the live callback contract; it also fails the
same confidence gate and is therefore restricted to an integration smoke test,
not active performance evaluation.

The new paper-statistics replay of cohort C uses PAR-2 timeout penalties and
instance-clustered bootstrap. It reports a treatment/native penalized-time ratio
of `0.9502` (95% interval `[0.7670, 1.1349]`), 4 instance wins and 4 losses, and
one safety-failure instance out of 8. All three frozen gates fail. The positive
point estimate therefore does not alter the stop decision.

### Fixed-baseline train pilot

Commit `1493ac0` pre-registered five train instances from five distinct official
MIPLIB Groups before its new arms were run. A later manuscript audit found that
all five had already been used by the first-run action-library pilot, contrary
to the v1 selection contract. The v1 results below are therefore reused
development comparisons, not independent confirmation. The complete finding
and corrective protocol are recorded in `docs/FIXED_BASELINE_PILOT_AUDIT.md`.

The pilot used seeds 0-2, a 30-second limit, first-run-only intervention, and
fixed native selected-cut budget for `random-rank`, `efficacy-rank`, and
`adaptive-score`.

No action passes the pre-registered pilot gate:

| Action | Changed instances | Safety failures | PAR-2 time ratio | 95% interval | W/L |
| --- | ---: | ---: | ---: | ---: | ---: |
| random-rank | 5/5 | 0 | 1.0193 | [0.9781, 1.0623] | 2/3 |
| efficacy-rank | 5/5 | 1 | 1.0007 | [0.8806, 1.1485] | 2/3 |
| adaptive-score | 2/5 | 1 | 1.0719 | [0.9794, 1.2564] | 2/3 |

Two native seed runs also reached the 30-second limit, so the pilot's required
all-native-complete check fails for every action. Separately, adaptive-score
caused a native-optimal/treatment-timeout pair on `neos-1171448` seed 0, and
efficacy-rank caused one on `piperout-27` seed 1. Random rank is safe in this
small pilot but slower on average with more losses than wins.

The machine-readable decision is: no fixed baseline advances, do not enlarge or
tune these actions on pilot outcomes, and do not open validation or test.

### Disjoint corrective pilot v2

The v2 plan was frozen before active outcomes. It selects the first eight
qualifying train instances under a deterministic runtime-ordered rule after
excluding every instance in the action-library pilot and cohorts A, B, and C.
Validation and test remain sealed. The v2 plan does not rewrite or erase the v1
protocol violation.

The run is complete, but it does not provide an independent performance
estimate: every one of the 24 native arms reached the 30-second limit. Hence
all three PAR-2 ratios and confidence intervals are undefined under the frozen
attribution rule. Random rank and efficacy rank changed selected sets on 6/8
instances, while adaptive score changed them on 5/8, but there are zero
completed native controls. All actions fail the pre-registered feasibility,
confidence, and win/loss gates; no action advances. Zero attributable safety
failures must not be interpreted as safety evidence in the absence of an
attributable pair.

### Publication-strengthening V1

The 12-instance, three-seed structural parity extension is complete. Its strict
all-pair gate fails because one seed never exercised a cut-selector callback and
one seed reached the 60-second limit in all arms. For each of no-op and
direct-hybrid, 34/36 pairs pass; all 34 complete pairs with an exercised
callback pass, and there are zero complete structural mismatches. Ten of 12
instances pass on all seeds. The strict failure and the diagnostic subset are
both retained.

The online-compatible imitation contract now contains 37 features after seven
trace-only counters were removed. Its retrained XGBoost model has positive
validation point estimates but again fails the frozen official-Group-OOD
confidence gate. The online arm is integration-smoke-only and is not authorized
for active performance evaluation.

The frozen 40-instance, three-seed fixed-baseline experiment is complete. Its
480 raw arms cover 40 distinct official Group keys, three seeds, native SCIP,
and three fixed treatment arms. All expected combinations are present exactly
once, and all 113 treatment-observed initial contexts match across actions.

| Action | Native-complete instances | Interventions | PAR-2 time ratio | 95% interval | W/L | Safety-failure instances |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random rank | 31 | 113 | 1.1114 | [0.9880, 1.3323] | 13/18 | 1 |
| efficacy rank | 31 | 103 | 1.0647 | [0.9156, 1.2979] | 15/16 | 2 |
| adaptive score | 31 | 62 | 1.0019 | [0.8481, 1.1486] | 18/13 | 0 |

Random and efficacy ranking are slower on average and have attributable safety
failures. Adaptive score is structurally safer in this cohort but statistically
neutral: its estimated treatment/native ratio is 1.0019 and its interval spans
substantial benefit and harm. None passes the frozen advancement gate.

This larger estimate closes the planned publication-strengthening experiment
without changing the stop decision. Validation and test remain sealed, the
online-compatible XGBoost arm remains smoke-only after failing its offline
official-Group-OOD gate, and native SCIP remains the only authorized policy.
