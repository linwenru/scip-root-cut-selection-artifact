import tempfile
import unittest
from pathlib import Path

import numpy as np

from scip_cut_trace_v2.ranking_train import (
    RankingMatrix,
    anchor_scip_top_candidate,
    compare_with_baseline,
    evaluate_scores,
    instance_balanced_weights,
    load_effective_matrix,
    subset_matrix,
)


class RankingTrainTest(unittest.TestCase):
    @staticmethod
    def _matrix():
        return RankingMatrix(
            name="synthetic",
            X=np.zeros((8, 2), dtype=np.float32),
            y=np.asarray([1, 0, 0, 1, 0, 1, 0, 0], dtype=np.uint8),
            group_ptr=np.asarray([0, 3, 5, 8], dtype=np.int64),
            group_sizes=np.asarray([3, 2, 3], dtype=np.int32),
            group_weights=np.asarray([0.75, 0.75, 1.5], dtype=np.float32),
            group_instances=np.asarray(["a", "a", "b"]),
            group_official_groups=np.asarray(["A", "A", "B"]),
            group_decision_ids=np.asarray(["a1", "a2", "b1"]),
            feature_names=np.asarray(["x", "z"]),
            baseline_score_rank_pre=np.asarray([1, 2, 3, 1, 2, 1, 2, 3]),
        )

    def test_metrics_are_averaged_per_instance(self):
        matrix = self._matrix()
        perfect_scores = np.asarray([3, 2, 1, 2, 1, 3, 2, 1], dtype=np.float32)
        metrics, per_instance = evaluate_scores(matrix, perfect_scores)

        self.assertEqual(set(per_instance), {"a", "b"})
        self.assertAlmostEqual(metrics["ndcg@10"], 1.0)
        self.assertAlmostEqual(metrics["selection_overlap"], 1.0)

    def test_comparison_reports_instance_bootstrap_delta(self):
        matrix = self._matrix()
        worse_baseline_rank = np.asarray([3, 1, 2, 2, 1, 3, 1, 2])
        matrix = RankingMatrix(
            **{
                **matrix.__dict__,
                "baseline_score_rank_pre": worse_baseline_rank,
            }
        )
        perfect_scores = np.asarray([3, 2, 1, 2, 1, 3, 2, 1], dtype=np.float32)
        result = compare_with_baseline(
            matrix, perfect_scores, seed=7, bootstrap_samples=200
        )

        self.assertGreater(result["comparison"]["ndcg@10"]["delta"], 0.0)
        self.assertEqual(result["comparison"]["ndcg@10"]["instances"], 2)

    def test_loader_filters_ineffective_queries_and_rejects_test(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "train.npz"
            np.savez_compressed(
                path,
                X=np.arange(10, dtype=np.float32).reshape(5, 2),
                y=np.asarray([1, 0, 1, 1, 0], dtype=np.uint8),
                group_ptr=np.asarray([0, 2, 4, 5], dtype=np.int64),
                group_sizes=np.asarray([2, 2, 1], dtype=np.int32),
                group_has_effective_pair=np.asarray([True, False, False]),
                effective_group_weight=np.asarray([1.0, 0.0, 0.0]),
                group_instance_name=np.asarray(["a", "b", "c"]),
                group_official_group=np.asarray(["A", "B", "C"]),
                group_decision_id=np.asarray(["a1", "b1", "c1"]),
                feature_names=np.asarray(["x", "z"]),
                baseline_score_rank_pre=np.asarray([1, 2, 1, 2, 1]),
            )

            matrix = load_effective_matrix(path)
            self.assertEqual(matrix.group_sizes.tolist(), [2])
            self.assertEqual(matrix.y.tolist(), [1, 0])

            sealed = root / "official_group_ood_test.npz"
            sealed.touch()
            with self.assertRaisesRegex(ValueError, "sealed"):
                load_effective_matrix(sealed)

    def test_subset_rebalances_query_weights_by_instance(self):
        matrix = self._matrix()
        subset = subset_matrix(matrix, np.asarray([True, True, False]), "a-only")

        self.assertEqual(subset.group_sizes.tolist(), [3, 2])
        self.assertEqual(subset.group_weights.tolist(), [1.0, 1.0])
        weights = instance_balanced_weights(np.asarray(["a", "a", "b"]))
        self.assertEqual(weights.tolist(), [0.75, 0.75, 1.5])

    def test_anchor_preserves_scip_top_candidate(self):
        matrix = self._matrix()
        scores = np.asarray([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.9, 0.8])

        anchored = anchor_scip_top_candidate(matrix, scores)

        for start, stop in zip(matrix.group_ptr[:-1], matrix.group_ptr[1:]):
            model_order = np.argsort(-anchored[start:stop])
            scip_top = int(
                np.argmin(matrix.baseline_score_rank_pre[start:stop])
            )
            self.assertEqual(model_order[0], scip_top)


if __name__ == "__main__":
    unittest.main()
