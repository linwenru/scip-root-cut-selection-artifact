import csv
import tempfile
import unittest
from pathlib import Path

from scip_cut_trace_v2.audit import (
    _official_group_analysis,
    family_key,
    last_csv_record,
    parse_cutselector_stats,
    validate_report,
)


class AuditHelpersTest(unittest.TestCase):
    def test_parse_cutselector_stats(self):
        text = "  hybrid           :       0.03       0.00         99         11       2838          0      10789"
        self.assertEqual(
            parse_cutselector_stats(text, "hybrid"),
            {"calls": 99, "root_calls": 11, "selected": 2838, "forced": 0, "filtered": 10789},
        )

    def test_family_key_removes_numeric_variant(self):
        self.assertEqual(family_key("supportcase33.mps.gz"), "supportcase")
        self.assertEqual(family_key("app1-2.mps.gz"), "app")

    def test_official_group_analysis_separates_seen_and_unseen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_dir = root / "split"
            split_dir.mkdir()
            (split_dir / "train.test").write_text("a1.mps.gz\nc1.mps.gz\n")
            (split_dir / "val.test").write_text("a2.mps.gz\nb1.mps.gz\n")
            (split_dir / "test.test").write_text("b2.mps.gz\nu1.mps.gz\n")
            metadata = root / "metadata.csv"
            metadata.write_text(
                "instance_name,group\n"
                "a1.mps.gz,A\n"
                "a2.mps.gz,A\n"
                "b1.mps.gz,B\n"
                "b2.mps.gz,B\n"
                "c1.mps.gz,C\n"
                "u1.mps.gz,\n"
            )
            result = _official_group_analysis(root, metadata)
            self.assertEqual([item["group"] for item in result["cross_split_groups"]], ["A", "B"])
            self.assertEqual(len(result["evaluation_strata"]["val"]["seen_family"]), 1)
            self.assertEqual(len(result["evaluation_strata"]["test"]["unseen_family"]), 1)
            self.assertEqual(result["evaluation_strata"]["test"]["officially_ungrouped"], ["u1.mps.gz"])

    def test_last_csv_record_handles_quoted_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["run_number", "cut_name"])
                writer.writerow([1, "assignTimes(A,B)"])
                writer.writerow([2, "last"])
            self.assertEqual(last_csv_record(path), ["2", "last"])

    def test_validation_rejects_constant_and_ambiguous_features(self):
        report = {
            "instances_by_split": {"train": 163, "val": 35, "test": 35},
            "solver_status_by_split": {"train:optimal": 163, "val:optimal": 35, "test:optimal": 35},
            "missing_outputs": [],
            "header_mismatches": [],
            "cutselector_parse_failures": [],
            "cutselector_call_mismatches": [],
            "run_count_mismatches": [],
            "totals": {
                "transition_rows": 2,
                "hybrid_calls": 2,
                "candidate_rows": 4,
                "hybrid_selected": 2,
                "hybrid_filtered": 2,
            },
            "candidate_scan": {
                "counts": {
                    "rows": 4,
                    "decision_groups": 2,
                    "selected_true_rows": 1,
                    "selected_blank_rows": 3,
                    "forced_rows": 1,
                    "sparsity_one_rows": 4,
                    "groups_with_duplicate_cut_id": 1,
                    "applied_rows": 2,
                    "root_rows": 4,
                },
                "missing_values": {"cutoff_distance": 1},
            },
        }
        validation = validate_report(report)
        self.assertFalse(validation["checks"]["sparsity_feature_is_nonconstant"])
        self.assertFalse(validation["checks"]["applied_label_has_no_ambiguous_ids"])



if __name__ == "__main__":
    unittest.main()
