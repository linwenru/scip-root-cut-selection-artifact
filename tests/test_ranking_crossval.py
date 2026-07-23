import unittest

import numpy as np

from scip_cut_trace_v2.ranking_crossval import assign_official_group_folds
from scip_cut_trace_v2.ranking_train import RankingMatrix


class RankingCrossValidationTest(unittest.TestCase):
    def test_assigns_whole_official_groups_and_excludes_ungrouped(self):
        matrix = RankingMatrix(
            name="synthetic",
            X=np.zeros((10, 1), dtype=np.float32),
            y=np.asarray([1, 0] * 5, dtype=np.uint8),
            group_ptr=np.arange(0, 11, 2, dtype=np.int64),
            group_sizes=np.asarray([2] * 5, dtype=np.int32),
            group_weights=np.ones(5, dtype=np.float32),
            group_instances=np.asarray(["a1", "a2", "b", "c", "u"]),
            group_official_groups=np.asarray(["A", "A", "B", "C", ""]),
            group_decision_ids=np.asarray(["a1", "a2", "b", "c", "u"]),
            feature_names=np.asarray(["x"]),
            baseline_score_rank_pre=np.asarray([1, 2] * 5),
        )

        assignments, folds = assign_official_group_folds(matrix, n_folds=2)

        self.assertEqual(assignments[0], assignments[1])
        self.assertEqual(assignments[-1], -1)
        assigned_groups = [group for fold in folds for group in fold["official_groups"]]
        self.assertEqual(sorted(assigned_groups), ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
