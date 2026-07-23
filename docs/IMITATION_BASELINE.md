# XGBoost imitation baseline V1

## Decision being tested

This stage asks whether pre-decision cut and LP features can improve on SCIP's
raw hybrid-score ordering when predicting the cuts that SCIP eventually
applied. It supplies a candidate ranker for the current policy budget of at most
one root intervention per run. It does not test solve-level benefit and cannot
authorize deployment by itself.

No test matrix is loaded or scored. The primary external validation remains the
official-Group-disjoint validation split. Seen-family validation is a secondary
specialization check, and officially ungrouped validation is diagnostic only.

## Fixed training contract

- XGBoost 3.2.x is pinned because it supports the project's Python 3.10 floor;
  XGBoost 3.3.x requires Python 3.12.
- The objective is `rank:ndcg` with the `mean` pair strategy and four sampled
  pairs per candidate.
- Only queries containing both applied and non-applied candidates are used.
- One cut-selector decision is one ranking query. XGBoost receives one
  instance-balanced weight per query.
- The model has depth 4, learning rate 0.05, fixed regularization, and fixed
  random seed 20260716. There is no hyperparameter search.
- Early stopping uses primary OOD validation `NDCG@10` with an 80-round
  patience. The best model contains four trees (`best_iteration == 3`).

Metrics are first averaged across queries belonging to the same instance and
then equally across instances. Confidence intervals use deterministic
instance-level bootstrap resampling with 5,000 samples. The baseline ordering
is ascending `score_rank_pre`.

## External validation

| Protocol | Metric | SCIP score | Model | Delta | 95% delta interval |
|---|---|---:|---:|---:|---:|
| Unseen Group | `NDCG@10` | 0.7068 | 0.7364 | +0.0296 | [-0.0078, +0.0870] |
| Unseen Group | average precision | 0.5132 | 0.5370 | +0.0239 | [+0.0012, +0.0544] |
| Unseen Group | selection overlap | 0.4697 | 0.4957 | +0.0259 | [-0.0101, +0.0833] |
| Seen family | `NDCG@10` | 0.7195 | 0.7973 | +0.0778 | [+0.0340, +0.1417] |
| Ungrouped | `NDCG@10` | 0.6869 | 0.6967 | +0.0098 | [0.0000, +0.0218] |

The primary point estimates improve, but the unseen-Group `NDCG@10` interval
still includes zero. The external stage gate therefore fails and the model is
not eligible for active SCIP intervention.

A single predeclared safety transformation was subsequently evaluated: keep
SCIP's rank-1 candidate fixed and use ML only to rerank the tail. On unseen-Group
validation, anchored `NDCG@10` improves by +0.0353 with interval
[-0.0071, +0.0959]. It removes the Top-1 risk but still fails the external gate.

## Seed stability

The same fixed configuration was also trained with four adjacent seeds. All
five seeds improve primary OOD `NDCG@10` in point estimate:

| Seed | Best iteration | Delta | 95% delta interval |
|---:|---:|---:|---:|
| 20260716 | 3 | +0.0296 | [-0.0078, +0.0870] |
| 20260717 | 12 | +0.0337 | [-0.0017, +0.0752] |
| 20260718 | 16 | +0.0365 | [-0.0011, +0.0910] |
| 20260719 | 155 | +0.0442 | [-0.0032, +0.1026] |
| 20260720 | 132 | +0.0260 | [-0.0154, +0.0851] |

The direction is seed-stable, but every interval includes zero. Averaging the
five per-instance deltas shows eight OOD instances regress, three tie, and nine
improve. The uncertainty is therefore dominated by instance heterogeneity, not
one unlucky training seed.

## Official-Group cross-validation

A fixed-configuration five-fold diagnostic uses only `train.npz`. Entire
non-empty official Groups stay in one fold. Of 135 effective training
instances, 97 instances in 87 official Groups participate; 38 instances without
an official Group are explicitly excluded rather than being treated as one
artificial family.

| Metric | SCIP score | Out-of-fold model | Delta | 95% delta interval |
|---|---:|---:|---:|---:|
| `NDCG@1` | 0.9605 | 0.9502 | -0.0103 | [-0.0309, 0.0000] |
| `NDCG@5` | 0.7748 | 0.7925 | +0.0177 | [+0.0007, +0.0367] |
| `NDCG@10` | 0.7222 | 0.7499 | +0.0278 | [+0.0129, +0.0440] |
| average precision | 0.5751 | 0.5916 | +0.0165 | [+0.0081, +0.0252] |
| selection overlap | 0.5216 | 0.5360 | +0.0144 | [+0.0038, +0.0270] |

At `NDCG@10`, 38 instances improve, 13 regress, and 46 tie. This supports a
real but modest cross-family imitation signal. Each fold uses itself for early
stopping, so this diagnostic is mildly optimistic and cannot override the
external validation failure.

The SCIP-Top-1 anchored version raises Group-CV `NDCG@10` by +0.0302 with
interval [+0.0152, +0.0465], while preserving baseline `NDCG@1` by construction.

## Stage decision

The model contains a real imitation signal, but neither the raw nor the anchored
candidate passes the external stage gate. Adjacent seeds are not substituted
after seeing validation results, and thresholds or blend weights will not be
swept. The observational-imitation-to-online route therefore stops here before
any active SCIP experiment.

The project-level objective is not yet rejected, because imitation quality is
only a proxy for solve benefit. The next methodological stage must be a separate
online causal harness: first prove no-op parity with SCIP, then measure whether
one controlled root intervention per run can cause reproducible solve-level
changes under paired seeds. Only causal leverage, not another imitation model,
can justify reopening ML policy work.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m scip_cut_trace_v2.ranking_train

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 \
  -m scip_cut_trace_v2.ranking_crossval
```

Generated model binaries are ignored by Git. Committed manifests contain the
model hash, fixed parameters, feature importance, per-instance validation
results, fold assignments, and stage-gate decisions.
