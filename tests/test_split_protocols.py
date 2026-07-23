import csv
import tempfile
import unittest
from pathlib import Path

from scip_cut_trace_v2.split_protocols import build_protocols, write_protocols


class SplitProtocolsTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.split_dir = self.root / "split"
        self.split_dir.mkdir()
        (self.split_dir / "train.test").write_text("a1.mps.gz\nu0.mps.gz\n")
        (self.split_dir / "val.test").write_text("a2.mps.gz\nb1.mps.gz\nv0.mps.gz\n")
        (self.split_dir / "test.test").write_text("a3.mps.gz\nc1.mps.gz\nt0.mps.gz\n")
        self.metadata = self.root / "metadata.csv"
        self.metadata.write_text(
            "instance_name,group\n"
            "a1.mps.gz,A\n"
            "a2.mps.gz,A\n"
            "a3.mps.gz,A\n"
            "b1.mps.gz,B\n"
            "c1.mps.gz,C\n"
            "u0.mps.gz,\n"
            "v0.mps.gz,\n"
            "t0.mps.gz,\n"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_builds_disjoint_and_seen_family_protocols(self):
        result = build_protocols(self.split_dir, self.metadata)

        self.assertEqual(result["lists"]["official_group_ood"]["val"], ["b1.mps.gz"])
        self.assertEqual(result["lists"]["official_group_ood"]["test"], ["c1.mps.gz"])
        self.assertEqual(result["lists"]["seen_family"]["val"], ["a2.mps.gz"])
        self.assertEqual(result["lists"]["seen_family"]["test"], ["a3.mps.gz"])
        self.assertEqual(
            result["lists"]["officially_ungrouped"]["test"], ["t0.mps.gz"]
        )
        self.assertTrue(all(result["summary"]["checks"].values()))

    def test_rejects_instance_present_in_two_splits(self):
        (self.split_dir / "test.test").write_text("a1.mps.gz\n")
        with self.assertRaisesRegex(ValueError, "occurs in both"):
            build_protocols(self.split_dir, self.metadata)

    def test_rejects_missing_official_metadata(self):
        with self.metadata.open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["instance_name", "group"])
            writer.writerow(["a1.mps.gz", "A"])
        with self.assertRaisesRegex(ValueError, "missing from official metadata"):
            build_protocols(self.split_dir, self.metadata)

    def test_rejects_same_unseen_group_in_validation_and_test(self):
        rows = self.metadata.read_text().replace("c1.mps.gz,C", "c1.mps.gz,B")
        self.metadata.write_text(rows)
        with self.assertRaisesRegex(ValueError, "ood_val_and_test_groups_are_disjoint"):
            build_protocols(self.split_dir, self.metadata)

    def test_writes_all_protocol_files(self):
        result = build_protocols(self.split_dir, self.metadata)
        output_dir = self.root / "output"
        write_protocols(result, output_dir)

        self.assertTrue((output_dir / "instance_assignments.csv").is_file())
        self.assertEqual(
            (output_dir / "official_group_ood" / "val.test").read_text(),
            "b1.mps.gz\n",
        )
        self.assertTrue((output_dir / "protocol_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
