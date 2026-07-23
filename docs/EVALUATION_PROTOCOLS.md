# Evaluation protocols

The supplied 233-instance corpus is retained in its original 163/35/35 split.
No SCIP solve or raw CSV is copied when these protocol views are generated.

## Protocols

1. `official_group_ood` is the primary generalization result. Validation and
   test contain only instances whose non-empty official MIPLIB `Group` is absent
   from training. Validation and test Groups are also mutually disjoint.
2. `seen_family` is a secondary specialization result. Every validation or test
   Group is represented in training.
3. `officially_ungrouped` is diagnostic only. MIPLIB publishes no Group for
   these instances, so they cannot support either a seen- or unseen-family claim.

All three views use the original 163-instance training pool. That pool contains
officially ungrouped instances, whose hidden family relationships cannot be
verified. Claims must therefore say "official-Group-disjoint", not "provably
unrelated MILPs".

For the supplied corpus, the resulting counts are:

| View | Train | Validation | Test |
|---|---:|---:|---:|
| Official-Group OOD | 163 | 22 | 17 |
| Seen family | 163 | 5 | 10 |
| Officially ungrouped | 163 | 8 | 8 |

The training pool has 116 grouped and 47 officially ungrouped instances. OOD
validation covers 19 Groups and OOD test covers 16 Groups.

## Generate

```bash
PYTHONPATH=src python3 -m scip_cut_trace_v2.split_protocols
```

The command writes `data/manifests/evaluation_protocols/`:

- `instance_assignments.csv` records one authoritative stratum per instance.
- `protocol_summary.json` records source hashes, counts, Group sets, and passed
  invariants.
- Each protocol directory contains ordinary `train.test`, `val.test`, and
  `test.test` lists.

Model selection may inspect validation results only. Test results remain sealed
until the policy, hyperparameters, intervention gate, and seed protocol are fixed.
