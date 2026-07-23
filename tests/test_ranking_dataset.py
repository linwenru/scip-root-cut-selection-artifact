import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scip_cut_trace_v2.observational import (
    CANDIDATE_MODEL_FEATURES,
    DECISION_MODEL_FEATURES,
)
from scip_cut_trace_v2.ranking_dataset import build_ranking_dataset


class RankingDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.processed = self.root / "processed"
        self.assignments_path = self.root / "assignments.csv"
        self.source_manifest = self.root / "source_manifest.json"
        self.output = self.root / "matrices"
        self.manifest = self.root / "manifest.json"
        self.analysis = self.root / "analysis.json"

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _decision(decision_id, run, size, positives, **updates):
        row = {feature: "0" for feature in DECISION_MODEL_FEATURES}
        row.update(
            {
                "decision_id": decision_id,
                "run_number": str(run),
                "is_policy_eligible_decision": "True",
                "pre_state_available": "True",
                "has_source_cut_id_collision": "False",
                "n_logical_candidates": str(size),
                "n_observed_applied_logical_cuts": str(positives),
                "pre_lp_status": "OPTIMAL",
                "n_candidate_occurrences": str(size),
            }
        )
        row.update({key: str(value) for key, value in updates.items()})
        return row

    @staticmethod
    def _candidate(decision_id, index, applied, origin_type, score):
        row = {feature: "0" for feature in CANDIDATE_MODEL_FEATURES}
        row.update(
            {
                "decision_id": decision_id,
                "logical_candidate_id": f"{index:064d}",
                "source_cut_id": str(index),
                "observed_logical_is_applied": str(applied),
                "score_rank_pre": str(index + 1),
                "score_rank_fraction_pre": str(index / 2),
                "score": str(score),
                "nnz": str(index + 2),
                "cutoff_distance": "" if index == 0 else str(score),
                "cutoff_distance_available": str(index != 0),
                "is_local": str(index % 2 == 0),
                "is_modifiable": "False",
                "is_removable": "True",
                "is_integral": str(index % 2 == 1),
                "in_global_cutpool": "False",
                "origin_type": origin_type,
            }
        )
        return row

    def _write_instance(self, assignment, decision_specs, origin_type):
        stem = assignment["instance_name"].removesuffix(".mps.gz")
        instance_dir = self.processed / assignment["original_split"] / stem
        instance_dir.mkdir(parents=True)
        decisions = []
        candidates = []
        for decision_id, run, labels in decision_specs:
            decisions.append(self._decision(decision_id, run, len(labels), sum(labels)))
            for index, applied in enumerate(labels):
                candidates.append(
                    self._candidate(
                        decision_id,
                        index,
                        applied,
                        origin_type,
                        score=1.0 - index / 10,
                    )
                )
        decision_fields = list(dict.fromkeys(
            list(DECISION_MODEL_FEATURES)
            + [
                "decision_id",
                "is_policy_eligible_decision",
                "has_source_cut_id_collision",
                "n_logical_candidates",
                "n_observed_applied_logical_cuts",
            ]
        ))
        candidate_fields = list(dict.fromkeys(
            list(CANDIDATE_MODEL_FEATURES)
            + [
                "decision_id",
                "logical_candidate_id",
                "source_cut_id",
                "observed_logical_is_applied",
            ]
        ))
        with gzip.open(
            instance_dir / "root_decisions.csv.gz", "wt", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=decision_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(decisions)
        with gzip.open(
            instance_dir / "root_candidates.csv.gz", "wt", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=candidate_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(candidates)

    def test_builds_query_matrices_with_equal_total_instance_weight(self):
        assignments = [
            {
                "instance_name": "a.mps.gz",
                "original_split": "train",
                "official_group": "A",
                "evaluation_stratum": "training_grouped",
            },
            {
                "instance_name": "b.mps.gz",
                "original_split": "train",
                "official_group": "B",
                "evaluation_stratum": "training_grouped",
            },
            {
                "instance_name": "v.mps.gz",
                "original_split": "val",
                "official_group": "V",
                "evaluation_stratum": "unseen_family",
            },
            {
                "instance_name": "t.mps.gz",
                "original_split": "test",
                "official_group": "T",
                "evaluation_stratum": "unseen_family",
            },
        ]
        with self.assignments_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=assignments[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(assignments)

        self._write_instance(
            assignments[0],
            [("a-run-1", 1, [True, False]), ("a-run-2", 2, [False, True, False])],
            "SEPA",
        )
        self._write_instance(
            assignments[1], [("b-run-1", 1, [True, False])], "CONSHDLR"
        )
        self._write_instance(
            assignments[2], [("v-run-1", 1, [False, True])], "SEPA"
        )
        self._write_instance(
            assignments[3], [("t-run-1", 1, [True, False])], "SEPA"
        )
        self.source_manifest.write_text(
            json.dumps({"totals": {"policy_eligible_decisions": 5}})
        )

        result = build_ranking_dataset(
            self.processed,
            self.assignments_path,
            self.source_manifest,
            self.output,
            self.manifest,
            self.analysis,
        )

        with np.load(self.output / "train.npz") as matrix:
            self.assertEqual(matrix["X"].shape[0], 7)
            self.assertEqual(matrix["group_sizes"].tolist(), [2, 3, 2])
            self.assertEqual(matrix["group_weight"].tolist(), [0.75, 0.75, 1.5])
            self.assertEqual(matrix["group_has_effective_pair"].tolist(), [True, True, True])
            self.assertEqual(
                matrix["effective_group_weight"].tolist(), [0.75, 0.75, 1.5]
            )
            self.assertEqual(matrix["qid"].tolist(), [0, 0, 1, 1, 1, 2, 2])
            feature_names = matrix["feature_names"].tolist()
            self.assertIn("origin_type==SEPA", feature_names)
            self.assertIn("origin_type==CONSHDLR", feature_names)
            self.assertNotIn("pre_lp_status==OPTIMAL", feature_names)

        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["test_policy"], "matrices constructed and sealed; no test metric computed")
        self.assertEqual(result["matrices"]["official_group_ood_test"]["groups"], 1)
        self.assertEqual(
            result["matrices"]["official_group_ood_test"]["label_statistics"], "sealed"
        )
        self.assertNotIn("positives", result["matrices"]["official_group_ood_test"])
        analysis = json.loads(self.analysis.read_text())
        self.assertEqual(
            analysis["subsets"]["official_group_ood_test"]["label_statistics"], "sealed"
        )


if __name__ == "__main__":
    unittest.main()
