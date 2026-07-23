import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from scip_cut_trace_v2.observational import (
    CANDIDATE_COPY_FIELDS,
    CANDIDATE_MODEL_FEATURES,
    DECISION_MODEL_FEATURES,
    PRE_STATE_FIELDS,
    PROHIBITED_MODEL_FIELDS,
    build_instance,
    quality_checks,
    schema_contract,
)


INPUT_FIELDS = (
    "run_number",
    "node_number",
    "node_depth",
    "sep_round_node",
    "sep_round_run",
    "sep_round_global",
    "lp_round_node",
    "lp_round_run",
    "lp_round_global",
    "root",
    "cut_id",
    "is_forced",
    "is_selected",
    "is_applied",
    "original_index",
    "rank",
    "coeff_sparsity_ratio",
    "lp_position",
) + CANDIDATE_COPY_FIELDS


class ObservationalBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source"
        self.output = self.root / "output"
        self.source.mkdir()
        self.assignment = {
            "instance_name": "example.mps.gz",
            "original_split": "train",
            "official_group": "example",
            "evaluation_stratum": "training_grouped",
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _candidate(cut_id, score, original_index, sep_round=1, **updates):
        row = {field: "" for field in INPUT_FIELDS}
        row.update(
            {
                "run_number": "1",
                "node_number": "1",
                "node_depth": "0",
                "sep_round_node": str(sep_round),
                "sep_round_run": str(sep_round),
                "sep_round_global": str(sep_round),
                "lp_round_node": "0",
                "lp_round_run": "0",
                "lp_round_global": "0",
                "root": "True",
                "cut_id": str(cut_id),
                "cut_name": f"cut-{cut_id}",
                "origin_type": "SEPA",
                "is_forced": "False",
                "is_selected": "",
                "is_applied": "False",
                "original_index": str(original_index),
                "rank": "999",
                "score": str(score),
                "nnz": "2",
                "rhs": "1.0",
                "lhs": "-1e20",
                "constant": "0.0",
                "coeff_norm_l2": "1.0",
                "coeff_norm_l1": "1.0",
                "coeff_max_abs": "1.0",
                "coeff_min_abs": "1.0",
                "coeff_mean_abs": "1.0",
                "coeff_std_abs": "0.0",
                "coeff_sparsity_ratio": "1.0",
                "efficacy": "0.5",
                "obj_parallelism": "0.1",
                "cutoff_distance": "",
                "n_int_cols": "2",
                "is_local": "False",
                "is_modifiable": "False",
                "is_removable": "True",
                "is_integral": "True",
                "in_global_cutpool": "False",
                "lp_position": "-1",
            }
        )
        row.update({key: str(value) for key, value in updates.items()})
        return row

    def _write_source(self):
        rows = [
            self._candidate(10, 0.5, 5, is_applied="True"),
            self._candidate(20, 0.9, 2),
            self._candidate(10, 0.5, 1, is_applied="True"),
            self._candidate(30, 2.0, -1, is_forced="True", is_applied="True"),
            self._candidate(40, 0.8, 0, sep_round=2),
            self._candidate(50, 0.7, 1, sep_round=2),
            self._candidate(
                60,
                0.6,
                0,
                sep_round=3,
                node_number="2",
                node_depth="1",
                root="False",
            ),
        ]
        with (self.source / "candidate_cuts.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INPUT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        transition_fields = (
            "run_number",
            "node_number",
            "node_depth",
            "sep_round_node",
        ) + PRE_STATE_FIELDS + ("lp_obj_val_post", "delta_lp_obj_val")
        with (self.source / "sep_round_transitions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=transition_fields, lineterminator="\n")
            writer.writeheader()
            for sep_round in (1, 2):
                row = {field: "" for field in transition_fields}
                row.update(
                    {
                        "run_number": "1",
                        "node_number": "1",
                        "node_depth": "0",
                        "sep_round_node": str(sep_round),
                        "pre_lp_status": "OPTIMAL",
                        "n_lp_rows_pre": "10",
                        "lp_obj_val_pre": "4.0",
                        "lp_obj_val_post": "5.0",
                        "delta_lp_obj_val": "1.0",
                    }
                )
                writer.writerow(row)

    def test_collapses_duplicates_and_reconstructs_pre_decision_rank(self):
        self._write_source()
        summary = build_instance(self.source, self.output, self.assignment)

        with gzip.open(self.output / "root_candidates.csv.gz", "rt", newline="") as handle:
            candidates = list(csv.DictReader(handle))
        with gzip.open(self.output / "root_decisions.csv.gz", "rt", newline="") as handle:
            decisions = list(csv.DictReader(handle))

        first_decision = [row for row in candidates if row["decision_id"].endswith("sep=1")]
        self.assertEqual([row["source_cut_id"] for row in first_decision], ["20", "10"])
        self.assertEqual([row["score_rank_pre"] for row in first_decision], ["1", "2"])
        duplicate = next(row for row in first_decision if row["source_cut_id"] == "10")
        self.assertEqual(duplicate["candidate_multiplicity"], "2")
        self.assertEqual(duplicate["observed_logical_is_applied"], "True")
        self.assertEqual(duplicate["first_original_index"], "1")
        self.assertEqual(duplicate["source_cut_id_collision"], "False")
        self.assertEqual(len(decisions), 2)
        self.assertEqual([row["is_policy_eligible_decision"] for row in decisions], ["True", "False"])
        self.assertEqual(decisions[0]["n_forced_cuts"], "1")
        self.assertEqual(decisions[0]["n_duplicate_occurrences"], "1")
        self.assertEqual(decisions[0]["pre_lp_status"], "OPTIMAL")
        self.assertEqual(summary["counts"]["source_candidate_rows"], 7)
        self.assertEqual(summary["counts"]["source_root_rows"], 6)

    def test_preserves_hash_collisions_but_marks_their_labels_ambiguous(self):
        rows = [
            self._candidate(10, 0.9, 0, cut_name="first"),
            self._candidate(10, 0.8, 1, cut_name="second", is_applied="True"),
            self._candidate(20, 0.7, 0, sep_round=2),
            self._candidate(30, 0.6, 1, sep_round=2),
        ]
        with (self.source / "candidate_cuts.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INPUT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        transition_fields = (
            "run_number",
            "node_number",
            "node_depth",
            "sep_round_node",
        ) + PRE_STATE_FIELDS
        with (self.source / "sep_round_transitions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=transition_fields, lineterminator="\n")
            writer.writeheader()
            row = {field: "" for field in transition_fields}
            row.update(
                {
                    "run_number": "1",
                    "node_number": "1",
                    "node_depth": "0",
                    "sep_round_node": "2",
                    "pre_lp_status": "OPTIMAL",
                }
            )
            writer.writerow(row)

        summary = build_instance(self.source, self.output, self.assignment)
        with gzip.open(self.output / "root_candidates.csv.gz", "rt", newline="") as handle:
            candidates = list(csv.DictReader(handle))
        with gzip.open(self.output / "root_decisions.csv.gz", "rt", newline="") as handle:
            decisions = list(csv.DictReader(handle))

        first_decision = [row for row in candidates if row["decision_id"].endswith("sep=1")]
        self.assertEqual(len(first_decision), 2)
        self.assertTrue(all(row["source_cut_id_collision"] == "True" for row in first_decision))
        self.assertTrue(all(row["observed_label_ambiguous"] == "True" for row in first_decision))
        self.assertEqual(decisions[0]["is_policy_eligible_decision"], "False")
        self.assertEqual(decisions[1]["is_policy_eligible_decision"], "True")
        self.assertEqual(summary["counts"]["source_cut_id_collisions"], 1)
        self.assertEqual(summary["counts"]["duplicate_occurrences_collapsed"], 0)

    def test_model_feature_contract_excludes_post_outcome_and_source_rank(self):
        model_features = set(CANDIDATE_MODEL_FEATURES) | set(DECISION_MODEL_FEATURES)
        self.assertTrue(model_features.isdisjoint(PROHIBITED_MODEL_FIELDS))
        contract = schema_contract()
        self.assertNotIn("rank", contract["candidate_columns"])
        self.assertNotIn("lp_obj_val_post", contract["decision_columns"])
        self.assertEqual(contract["observational_label"], "observed_logical_is_applied")

    def test_quality_checks_enforce_accounting_and_feature_boundary(self):
        from collections import Counter

        totals = Counter(
            {
                "source_root_rows": 6,
                "root_candidate_occurrences": 5,
                "root_forced_rows": 1,
                "logical_candidates": 4,
                "duplicate_occurrences_collapsed": 1,
            }
        )
        self.assertTrue(all(quality_checks(totals, 1, 1).values()))
        totals["root_forced_rows"] = 2
        self.assertFalse(quality_checks(totals, 1, 1)["root_row_accounting_closes"])


if __name__ == "__main__":
    unittest.main()
