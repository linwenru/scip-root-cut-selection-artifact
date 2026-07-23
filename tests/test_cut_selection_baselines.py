import unittest

from scip_cut_trace_v2.cut_selection_baselines import (
    AdaptiveScoreWeights,
    adaptive_score_rank,
    adaptive_scores,
    deterministic_random_rank,
    efficacy_rank,
)


class FakeRow:
    def __init__(self, name, nonzeros=10, in_global_pool=False):
        self.name = name
        self._nonzeros = nonzeros
        self._in_global_pool = in_global_pool

    def getNNonz(self):
        return self._nonzeros

    def isInGlobalCutpool(self):
        return self._in_global_pool


class FakeModel:
    def __init__(
        self,
        efficacies,
        integer_columns=None,
        objective_parallelism=None,
        parallelism=None,
        cutoff_distances=None,
        has_incumbent=False,
    ):
        self.efficacies = efficacies
        self.integer_columns = integer_columns or {}
        self.objective_parallelism = objective_parallelism or {}
        self.parallelism = parallelism or {}
        self.cutoff_distances = cutoff_distances or {}
        self.has_incumbent = has_incumbent

    def getCutEfficacy(self, row):
        return self.efficacies[row.name]

    def getRowNumIntCols(self, row):
        return self.integer_columns.get(row.name, 0)

    def getRowObjParallelism(self, row):
        return self.objective_parallelism.get(row.name, 0.0)

    def getRowParallelism(self, left, right):
        return self.parallelism.get((left.name, right.name), 0.0)

    def getNSols(self):
        return int(self.has_incumbent)

    def getBestSol(self):
        return object()

    def getCutLPSolCutoffDistance(self, row, solution):
        del solution
        return self.cutoff_distances[row.name]


class CutSelectionBaselineTest(unittest.TestCase):
    def test_random_rank_is_keyed_and_does_not_mutate_input(self):
        cuts = [FakeRow(name) for name in ("a", "b", "c", "d", "e")]

        first = deterministic_random_rank(cuts, 2, "experiment-a")
        repeated = deterministic_random_rank(cuts, 2, "experiment-a")
        different = deterministic_random_rank(cuts, 2, "experiment-b")

        self.assertEqual(
            [row.name for row in first.cuts], [row.name for row in repeated.cuts]
        )
        self.assertNotEqual(
            [row.name for row in first.cuts], [row.name for row in different.cuts]
        )
        self.assertEqual([row.name for row in cuts], ["a", "b", "c", "d", "e"])
        self.assertEqual(first.nselectedcuts, 2)

    def test_efficacy_rank_uses_stable_native_order_for_ties(self):
        cuts = [FakeRow(name) for name in ("a", "b", "c", "d")]
        model = FakeModel({"a": 0.2, "b": 0.5, "c": 0.5, "d": float("nan")})

        selection = efficacy_rank(model, cuts, 2)

        self.assertEqual([row.name for row in selection.cuts], ["b", "c", "a", "d"])
        self.assertEqual(selection.nselectedcuts, 2)

    def test_adaptive_score_transfers_cutoff_weight_without_incumbent(self):
        cuts = [FakeRow("a", nonzeros=10), FakeRow("b", nonzeros=5)]
        model = FakeModel(
            {"a": 1.0, "b": 3.0},
            integer_columns={"a": 5, "b": 5},
            objective_parallelism={"a": 0.2, "b": 0.4},
        )
        weights = AdaptiveScoreWeights(
            directed_cutoff_distance=0.25,
            efficacy=0.5,
            integer_support=0.1,
            objective_parallelism=0.2,
        )

        scores, components, has_incumbent = adaptive_scores(model, cuts, weights)

        self.assertFalse(has_incumbent)
        self.assertAlmostEqual(components[1]["efficacy"], 0.75)
        self.assertAlmostEqual(components[1]["directed_cutoff_distance"], 0.0)
        self.assertAlmostEqual(components[1]["integer_support"], 0.1)
        self.assertAlmostEqual(components[1]["objective_parallelism"], 0.08)
        self.assertGreater(scores[1], scores[0])

    def test_adaptive_rank_filters_parallel_normal_cut_then_fills_if_needed(self):
        cuts = [FakeRow(name) for name in ("a", "b", "c")]
        model = FakeModel(
            {"a": 1.0, "b": 0.9, "c": 0.8},
            parallelism={("a", "b"): 0.2, ("a", "c"): 0.0},
        )

        selection = adaptive_score_rank(
            model, cuts, forcedcuts=[], nselectedcuts=2, root=True
        )
        filled = adaptive_score_rank(
            model, cuts, forcedcuts=[], nselectedcuts=3, root=True
        )

        self.assertEqual([row.name for row in selection.cuts[:2]], ["a", "c"])
        self.assertEqual([row.name for row in filled.cuts], ["a", "c", "b"])
        self.assertEqual(selection.metadata["parallelism_filtered"], 1)

    def test_rejects_invalid_native_budget(self):
        with self.assertRaises(ValueError):
            deterministic_random_rank([FakeRow("a")], 2, "key")


if __name__ == "__main__":
    unittest.main()
