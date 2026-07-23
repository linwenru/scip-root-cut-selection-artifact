# One-Shot Learned-Policy Pilot

## Purpose

This pilot gives the already frozen 42-round XGBoost imitation ranker one
bounded complete-solve test. It is not a new model-development loop. A failed
joint gate ends learned-policy development for this study; the pilot outcomes
must not be used to train or tune another version.

The existing 163/35/35 benchmark splits remain untouched. Pilot instances come
from the larger MIPLIB 2017 Collection and must be disjoint from every official
Group key already present in all 233 traced instances. Officially ungrouped
instances receive instance-specific keys.

## Frozen Design

- 18 previously unused Group keys, selected by a fixed SHA-256 order;
- one easy, feasible instance per Group key from the frozen acquisition list;
- paired SCIP seeds 0, 1, and 2;
- native, `xgb-imitation-shadow`, and `xgb-imitation-rank` arms;
- all six three-arm execution orders balanced across 54 blocks;
- one fresh single-threaded SCIP process at a time;
- 300-second limit and full B&B solve;
- at most one intervention, at the first eligible callback of run 1;
- fallback retained as part of the learned policy.

The primary estimand is the Group-equal geometric mean active/native
full-process wall-time PAR-2 ratio. Seeds are averaged on the log scale within
Group before Groups receive equal weight.

## Go/No-Go Gate

The unchanged model advances only when every frozen condition passes:

1. all 54 shadow pairs are structurally identical to native;
2. no observed pre-action context mismatch occurs;
3. at least 12 of 18 Group keys receive an active intervention;
4. no active correctness failure is observed;
5. no Group has a native-complete/active-incomplete outcome;
6. the primary point ratio is at most 0.95;
7. the one-sided 80% percentile-bootstrap upper ratio is at most 1.0.

These are exploratory continuation criteria, not a confirmatory efficacy or
safety claim. A pass authorizes a separately frozen study with at least 78
independent Group keys. A failure stops the learned-policy branch without
opening the existing sealed test.

## Frozen Decision

The run completed all 18 Group keys, 54 paired seed blocks, and 162 arm records
on 2026-07-21. The frozen decision was **No-Go**:

- active/native full-process wall-time PAR-2 ratio: 1.4931;
- 95% Group-cluster bootstrap interval: [1.1309, 2.1078];
- one-sided 80% upper ratio: 1.6993;
- secondary SCIP-solving-time ratio: 1.0681, with interval [0.9929, 1.1664];
- 40 active-intervention pairs and 14 policy fallbacks;
- 15 of 18 Group keys exposed to at least one intervention;
- one native-complete/active-timeout pair (`mine-90-10`, seed 2).

The model must not be tuned on these outcomes, the existing test remains
sealed, and no further learned-policy iteration is authorized by this study.

## Post-Hoc Diagnostics

The non-gating diagnostic explains, but does not revise, the decision:

- all 35 completed shadow/native pairs passed the full safety comparison;
- the remaining 19 shadow/native pairs had the same incomplete time-limit
  status, but terminal node, LP, or bound snapshots differed under callback
  overhead, so the pre-registered 54/54 full-structure gate failed;
- active/shadow full-process PAR-2 was 1.0299, while shadow/native was 1.4497;
  most deployment loss therefore arose on the Python/XGBoost execution path,
  with active ranking adding a further adverse point estimate;
- the sole completed-pair objective disagreement occurred on `neos-585467`.
  The active objective, 399.373917848, agrees with the official MIPLIB value
  399.3739, whereas native and shadow returned 398.411678521. The frozen
  consistency gate remains failed, but this is reported as numerical
  sensitivity rather than attributed to active-policy incorrectness. See the
  [official instance record](https://miplib.zib.de/instance_details_neos-585467.html).

The frozen statistics are in
`data/manifests/learned_policy_pilot_statistics_v1.json`; the separate post-hoc
record is `data/manifests/learned_policy_pilot_diagnostics_v1.json`.

## Commands

The acquisition list is frozen before any new-instance solver outcome is seen.
Download uses explicit HTTP byte ranges, validates every `Content-Range`, and
accepts a file only after its size and gzip stream pass validation:

```bash
PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.learned_policy_pilot \
  plan-downloads

PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.learned_policy_pilot download

PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.learned_policy_pilot freeze \
  --collection-metadata data/manifests/miplib2017_collection_metadata_2026-07-21.csv \
  --instances-dir data/raw/miplib2017_collection

PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.learned_policy_pilot run

PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.learned_policy_pilot analyze

PYTHONPATH=src .venv/bin/python -m scip_cut_trace_v2.learned_policy_diagnostics
```

The `freeze` command refuses to substitute another local instance when an item
on the frozen acquisition list is missing, and it refuses to overwrite an
existing plan or result. The run command verifies every source, split, download
record, model, and instance hash before launching SCIP.
