# Duplicate cut ID audit

## Conclusion

The candidate recorder is not proven to be duplicating rows itself: it iterates
once over the `cuts` list supplied to a single PySCIPOpt cut-selector callback,
and different `original_index` values show that repeated logical cuts are
already present at distinct positions in that list.

The tracer nevertheless has an implementation bug. It assumes that a content
hash uniquely identifies one candidate occurrence inside a decision round. The
observed callback input violates that assumption.

## Evidence

Examples of repeated logical cuts in one `(instance, run, node, sep_round)`:

| Instance | Run/node/round | Cut | Original indices | Stored ranks | Applied |
|---|---|---|---|---|---|
| csched007 | 1/211/1 | cmir30_8 | 23, 101 | 1, 12 | true, true |
| mzzv42z | 1/315/4 | implbd2_29 | 45, 225 | 1, 37 | true, true |
| neos-4738912-atrato | 1/1/7 | cmir3_137 | 28, 296 | 25, 26 | false, false |
| neos-860300 | 3/1/36 | clique150_5 | 21, 44 | 1, 4 | true, true |
| physiciansched3-3 | 1/1/2 | clique0_0 | 21, 116 | 20, 113 | true, true |

All paired records above have equal names, origin types, scores, dimensions,
sides, constants, and coefficient norms. They are duplicate logical cuts, not
evidence of a 32-bit collision.

A later full root-node rebuild also found genuine 32-bit collisions, distinct
from the examples above. There are 137 source IDs mapping to 274 observably
different candidates in 35 decisions:

- `k1mushroom`: 49 collided IDs across 23 decisions;
- `splice1k1`: 86 collided IDs across 11 decisions;
- `supportcase7`: 2 collided IDs in 1 decision.

For example, `k1mushroom` run 1/node 1/round 1 assigns source ID `129648504`
to both `andgate_70_301` and `andgate_41_2726`, with different scores and
efficacies. V2 preserves both candidates and marks their application labels
ambiguous.

## Collection-tracer issues found during audit

1. `_get_cut_id()` calls Python `hash()` and truncates to 32 bits. The result is
   not stable across processes and cannot be a persistent dataset identifier.
2. `_mark_applied_candidates()` uses set membership by logical ID. If one
   occurrence is applied, every duplicate occurrence is labelled applied.
3. `_backfill_candidate_ranks()` stores only the first occurrence through
   `setdefault()`. Later duplicates cannot be matched to application order.
4. The original decision-time score rank is overwritten with a post-hoc rank,
   making the stored `rank` unsafe as a model input.

## V2 representation

- `decision_id`: instance, run, node, and separation round.
- `candidate_occurrence_id`: decision ID plus original callback position.
- `source_signature_sha256`: deterministic digest of all observable row fields.
- `logical_candidate_id`: decision, source ID, and observable-signature digest.
- `candidate_multiplicity`: number of occurrences of the logical cut.
- `score_rank_pre`: rank available before native selection.
- `selected_order_post`: separate nullable post-selection field.

Training views should contain one row per logical cut. If duplicate occurrences
cannot be matched individually, application is labelled only at the logical-cut
level and never copied back as an occurrence-level fact.

Rows involved in an observed source-ID collision are excluded from imitation
training because the collection tracer copied application labels by source ID.
