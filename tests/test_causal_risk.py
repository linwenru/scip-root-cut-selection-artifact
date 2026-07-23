import copy
import unittest

import numpy as np

from scip_cut_trace_v2.causal_risk import (
    ACTIONS,
    action_candidate_indices,
    build_risk_matrix,
    leave_one_instance_out_masks,
    risk_metrics,
)


def _candidate(rank, efficacy, selected):
    return {
        "native_rank": rank,
        "native_selected": selected,
        "efficacy": efficacy,
        "obj_parallelism": 0.1 * (rank + 1),
        "cutoff_distance": 0.2 * (rank + 1),
        "nnz": rank + 2,
        "n_int_cols": rank + 1,
        "coeff_norm_l2": rank + 0.5,
        "coeff_max_abs": rank + 1.5,
        "coeff_std_abs": rank / 10,
        "is_integral": rank % 2 == 0,
        "is_local": False,
        "in_global_cutpool": True,
    }


def _record(instance="sample", seed=0):
    candidates = [
        _candidate(0, 0.3, True),
        _candidate(1, 0.1, True),
        _candidate(2, 0.2, False),
        _candidate(3, 0.4, False),
    ]
    solver_state = {
        "lp_rows": 10,
        "lp_cols": 5,
        "lp_iterations_total": 20,
        "lp_iterations_node": 2,
        "lp_count": 1,
        "separation_rounds_node": 0,
        "gap": 0.5,
        "processed_nodes": 1,
        "total_nodes": 1,
        "cuts_applied": 0,
        "candidate_cuts": 4,
        "forced_cuts": 0,
        "native_selected_cuts": 2,
    }
    labels = {
        action: {"eligible": True, "safe": True, "final_outcome": "ignored"}
        for action in ACTIONS
    }
    return {
        "instance_id": instance,
        "seed": seed,
        "context": {
            "solver_state": solver_state,
            "candidates": candidates,
            "forced_candidates": [],
        },
        "action_labels": labels,
    }


class CausalRiskTest(unittest.TestCase):
    def test_recovers_declared_action_pairs(self):
        candidates = _record()["context"]["candidates"]
        self.assertEqual(action_candidate_indices("boundary-swap", candidates, 2), (1, 2))
        self.assertEqual(action_candidate_indices("boundary-swap-2", candidates, 2), (1, 3))
        self.assertEqual(action_candidate_indices("efficacy-promote", candidates, 2), (1, 3))

    def test_features_do_not_depend_on_outcome_labels(self):
        safe_record = _record()
        unsafe_record = copy.deepcopy(safe_record)
        unsafe_record["action_labels"]["boundary-swap"]["safe"] = False
        unsafe_record["action_labels"]["boundary-swap"]["final_outcome"] = "timelimit"

        safe = build_risk_matrix([safe_record])
        unsafe = build_risk_matrix([unsafe_record])

        np.testing.assert_equal(safe.features, unsafe.features)
        self.assertEqual(safe.labels.tolist(), [0, 0, 0])
        self.assertEqual(unsafe.labels.tolist(), [1, 0, 0])
        self.assertFalse(any("instance" in name or "seed" in name for name in safe.feature_names))

    def test_excludes_context_ineligible_actions(self):
        record = _record()
        record["context"]["candidates"] = record["context"]["candidates"][:3]
        record["context"]["solver_state"]["candidate_cuts"] = 3
        record["action_labels"]["boundary-swap-2"]["eligible"] = False
        record["action_labels"]["efficacy-promote"]["eligible"] = True

        matrix = build_risk_matrix([record])

        self.assertEqual(matrix.actions.tolist(), ["boundary-swap", "efficacy-promote"])

    def test_leave_one_instance_out_keeps_all_rows_together(self):
        records = [_record("first", 0), _record("first", 1), _record("second", 0)]
        matrix = build_risk_matrix(records)

        folds = leave_one_instance_out_masks(matrix)

        self.assertEqual(len(folds), 2)
        for instance, train, test in folds:
            self.assertEqual(set(matrix.instances[test]), {instance})
            self.assertNotIn(instance, set(matrix.instances[train]))

    def test_perfect_risk_scores_have_perfect_metrics(self):
        labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
        scores = np.asarray([0.1, 0.9, 0.2, 0.8])

        metrics = risk_metrics(labels, scores)

        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["full_unsafe_recall"]["unsafe_recall"], 1.0)
        self.assertEqual(metrics["full_unsafe_recall"]["safe_abstention_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
