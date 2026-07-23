# Fixed-baseline Pilot Integrity Audit

## Finding

During the manuscript evidence audit on 2026-07-17, the five instances in the
pre-registered fixed-baseline train pilot were found to be identical to the
five instances in the earlier first-run action-library pilot:

| Instance | Earlier action-library pilot | Fixed-baseline pilot v1 |
| --- | ---: | ---: |
| `exp-1-500-5-5` | yes | yes |
| `neos-1171448` | yes | yes |
| `neos8` | yes | yes |
| `piperout-27` | yes | yes |
| `swath1` | yes | yes |

The fixed-baseline plan was committed as `1493ac0` after the earlier pilot was
already present in commit `a77f7eb`. Its selection contract nevertheless
stated that selected instances had not been used by the first-run pilot or
cohorts A, B, or C. That requirement was violated.

## Consequence

The raw runs, result manifests, and v1 plan remain unchanged as historical
artifacts. Their numerical results may be used only as a fixed-action
comparison on reused development instances. They must not be described as an
independent cohort, as evidence from previously unused instances, or as a
confirmation result.

This finding does not reverse the v1 rejection: no fixed action passed its
gate, and the reused data did not authorize validation or test access. It does
reduce the independence of that particular strengthening experiment.

## Corrective Experiment

`data/manifests/causal_fixed_baseline_disjoint_train_pilot_plan_v2.json`
pre-registers a replacement development pilot before any v2 active outcome is
inspected. Its instances are disjoint from the first-run pilot and cohorts A,
B, and C. Selection is deterministic from frozen observational artifacts, and
the validation and test splits remain sealed.

The v2 run completed on 2026-07-17. All 24 native arms reached the 30-second
limit, so no action has a completed native control and the frozen PAR-2 ratio is
not estimable. Random rank and efficacy rank intervened in 17 seed arms over 6
instances; adaptive score intervened in 11 seed arms over 5 instances. The
coverage condition therefore passes, but the all-native-complete,
confidence-bound, and win/loss conditions fail for every action.

The absence of attributable safety failures is not safety evidence because
there are zero attributable pairs. No action advances. The correction is
reported as a feasibility failure, not as independent fixed-baseline
performance evidence. No v1 artifact is silently rewritten.

Machine-readable artifacts:

- `data/manifests/causal_fixed_baseline_disjoint_train_pilot_v2.json`
- `data/manifests/paper_statistics_fixed_baseline_disjoint_train_pilot_v2.json`
- `data/manifests/causal_fixed_baseline_disjoint_train_pilot_decision_v2.json`
