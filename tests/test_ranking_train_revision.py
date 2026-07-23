import unittest

import numpy as np

from scip_cut_trace_v2.ranking_train import RankingMatrix
from scip_cut_trace_v2.ranking_train_revision import (
    assign_independent_group_folds,
    cluster_bootstrap_delta,
    independent_group_keys,
)


def _matrix():
    return RankingMatrix(
        name="synthetic",
        X=np.zeros((8, 1), dtype=np.float32),
        y=np.asarray([1, 0] * 4, dtype=np.uint8),
        group_ptr=np.asarray([0, 2, 4, 6, 8], dtype=np.int64),
        group_sizes=np.asarray([2, 2, 2, 2], dtype=np.int32),
        group_weights=np.ones(4, dtype=np.float32),
        group_instances=np.asarray(["a", "b", "c", "d"]),
        group_official_groups=np.asarray(["family", "family", "", ""]),
        group_decision_ids=np.asarray(["qa", "qb", "qc", "qd"]),
        feature_names=np.asarray(["x"]),
        baseline_score_rank_pre=np.asarray([1, 2] * 4, dtype=np.int32),
    )


class RankingTrainRevisionTest(unittest.TestCase):
    def test_official_groups_stay_together_and_ungrouped_instances_are_unique(self):
        matrix = _matrix()

        keys = independent_group_keys(matrix)
        folds, summaries = assign_independent_group_folds(matrix, 2)

        self.assertEqual(keys.tolist()[:2], ["family", "family"])
        self.assertNotEqual(keys[2], keys[3])
        self.assertEqual(folds[0], folds[1])
        self.assertEqual(sum(item["group_key_count"] for item in summaries), 3)

    def test_cluster_bootstrap_uses_group_as_highest_level(self):
        baseline = {
            "a": {"metric": 0.0},
            "b": {"metric": 0.0},
            "c": {"metric": 0.0},
        }
        model = {
            "a": {"metric": 0.1},
            "b": {"metric": 0.2},
            "c": {"metric": -0.1},
        }
        clusters = {"a": "family", "b": "family", "c": "other"}

        result = cluster_bootstrap_delta(
            baseline,
            model,
            clusters,
            "metric",
            seed=7,
            samples=500,
        )

        self.assertEqual(result["instances"], 3)
        self.assertEqual(result["clusters"], 2)
        self.assertAlmostEqual(result["delta"], (0.1 + 0.2 - 0.1) / 3.0)


if __name__ == "__main__":
    unittest.main()
