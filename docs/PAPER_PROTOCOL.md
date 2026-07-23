# Paper Protocol

## Scope

This protocol freezes the strengthened empirical study before new validation or
test outcomes are inspected. The study asks whether a single, auditable root
cut-selection intervention can improve complete branch-and-bound solves over
SCIP 10.0.2's native hybrid cut selector.

The paper is a methods, audit, and causal-evaluation study. It does not claim
that the current ML policy outperforms SCIP. Existing cohort A-C results are
development evidence and remain reported as such.

## Research Questions

1. **Trace validity.** Can candidate cuts, native decisions, restarts, and
   repeated logical rows be represented without identity collisions or
   outcome-derived feature leakage?
2. **Intervention validity.** Does an exercised no-op callback, and a direct
   call to SCIP's native hybrid selector, reproduce native structural outcomes
   in fresh processes?
3. **Causal leverage.** Do fixed first-root ranking interventions change
   complete-solve outcomes, and are any gains stable across random seeds and
   instances?
4. **Generalization and safety.** Can pre-action information identify a policy
   that improves unseen instances or MIPLIB Groups without causing an
   attributable incomplete solve or final-solution mismatch?

## Evaluation Units And Splits

- The independent statistical unit is an MPS instance, not a cut, callback, or
  SCIP seed.
- Seeds are paired repeated measurements nested within each instance. Seed
  outcomes are aggregated within an instance before instances receive equal
  weight.
- The primary generalization protocol is MIPLIB Group-disjoint. A seen-Group
  split is a secondary interpolation analysis and must not be described as
  out-of-family generalization.
- Development uses only train data. Baseline definitions and analysis code are
  frozen before validation. The official test split is opened once, only if a
  method passes the validation gate.
- Existing cohorts A-C are not reusable as independent confirmation data for a
  policy tuned after inspecting their outcomes.

## Solver And Intervention Contract

- Solver: SCIP 10.0.2 through PySCIPOpt 6.2.1, one thread, paired SCIP seeds,
  identical limits, and one fresh process per instance, seed, and arm.
- Control: unmodified native SCIP hybrid cut selection.
- Primary intervention scope: the first eligible cut-selection callback at the
  root of the first SCIP run only.
- Every primary ranking arm preserves the number of optional cuts selected by
  native hybrid and leaves forced cuts untouched.
- Later callbacks and restart runs delegate to SCIP. A restart-aware per-run
  intervention is exploratory and requires a separate pre-registration.
- A pair is attributable only when its pre-action decision context is captured
  and the selected cut set actually changes.

## Frozen Baselines

1. **Native SCIP:** the performance and safety reference.
2. **No-op parity:** an instrumentation check, not a performance method.
3. **Direct-hybrid parity:** an ABI/callback parity check, not a performance
   method.
4. **Random rank:** a deterministic hash permutation at native selected-cut
   budget. Results must be averaged over predeclared SCIP seeds; no favorable
   random permutation may be selected post hoc.
5. **Efficacy rank:** descending SCIP cut efficacy at native selected-cut
   budget with stable native-order ties.
6. **Adaptive-score port:** an independent implementation of the normalized
   efficacy, directed-cutoff-distance, integer-support, and objective-
   parallelism score and parallelism filter used by Turner et al. (2023), at
   native selected-cut budget. This is a scoring-family baseline, **not** a
   reproduction of their GCNN, fixed-ten-cuts, presolved-instance protocol.
7. **Imitation ranker:** the existing XGBoost model is observational evidence.
   It may enter active evaluation only after an online-compatible feature
   contract is frozen and retrained on train data; unavailable live features
   may not be silently imputed from outcomes.

References for the learned baselines considered during design:

- Turner et al., *Adaptive Cut Selection in Mixed-Integer Linear Programming*,
  OJMO 4 (2023), <https://doi.org/10.5802/ojmo.25>.
- Paulus et al., *Learning to Cut by Looking Ahead: Cutting Plane Selection via
  Imitation Learning*, ICML 2022,
  <https://proceedings.mlr.press/v162/paulus22a.html>.

## Outcomes

### Safety gate

A treatment must have zero attributable safety failures on the validation
cohort. A safety failure includes native completion paired with treatment
timeout, inconsistent terminal status, objective sense, primal bound, or dual
bound. The observed failure count and an exact one-sided 95% binomial upper
confidence bound are reported; zero observed failures is not called proof of
zero risk.

### Primary performance outcome

The confirmatory performance estimand is the instance-equal geometric mean of
paired penalized solving-time ratios. A complete solve uses observed solving
time; an incomplete arm receives PAR-2 (`2 * time_limit`). Seeds are averaged
on the log-ratio scale within each instance.

Historical experiments that pre-registered LP iterations as primary retain
that label. They are not retroactively re-scored as confirmatory wall-time
experiments.

### Secondary outcomes

- LP iterations;
- total processed nodes;
- primal-dual integral;
- cuts applied;
- win/tie/loss counts at instance level;
- callback overhead measured by parity arms.

Secondary metrics cannot rescue a failed safety gate or a failed primary
outcome.

## Statistical Analysis

- Report the instance-equal point estimate and a paired cluster bootstrap 95%
  confidence interval, resampling instances with all their seeds.
- Use a fixed bootstrap seed and at least 10,000 replicates for the final
  report.
- A performance baseline passes only when the upper confidence limit of its
  penalized-time geometric-mean ratio is below `1.0`, it has more instance wins
  than losses, and the safety gate passes.
- When several fixed baselines are tested on validation, adjust confirmatory
  p-values with Holm's method and report all unadjusted effect sizes. The test
  split evaluates only the single method chosen by the frozen validation rule.
- Post-hoc per-seed oracles are labeled upper bounds. They are never compared
  with a deployable policy as if they were executable decisions.

## Stopping Rules

- A baseline with an attributable safety failure cannot advance to test.
- Do not tune thresholds, actions, or feature sets on validation or test
  outcomes after the protocol is frozen.
- If no fixed baseline passes validation, stop active ML development and report
  the audit, causal harness, parity checks, oracle gap, and negative result.
- Reopening requires a new pre-action intervention family or genuinely external
  instances, followed by a new dated protocol version.

## Reproducibility Deliverables

The artifact must include exact split manifests, instance hashes, solver and
Python versions, parameter snapshots, one JSON result per arm, context hashes,
aggregate scripts, tests, and a machine-readable table mapping every paper
claim to its source manifest. Large MIPLIB files need not be redistributed;
their official names and hashes are sufficient.

## Publication-strengthening amendment V1

The frozen extension in `docs/PUBLICATION_STRENGTHENING_V1.md` adds two
train-only experiments before the manuscript is revised:

- structural no-op and direct-hybrid parity on 12 distinct MIPLIB Group keys
  and three paired seeds;
- fixed-baseline causal estimation on 40 distinct Group keys, three paired
  seeds, and a 300-second limit.

The 40-instance cohort is an expanded development estimate, not independent
validation. Its deterministic selection rule does not use treatment outcomes,
but it includes fast instances seen in earlier development work. Validation
and test remain sealed.

An online-compatible 37-feature XGBoost imitation model was retrained under the
existing gate. Its official-Group-OOD point estimate is positive but its 95%
interval crosses zero, so it is not authorized for active performance
evaluation. A single-instance smoke test may verify integration and artifact
hashing only.
