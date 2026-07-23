import json
import tempfile
import unittest
from pathlib import Path

from scip_cut_trace_v2.artifact_index import (
    build_evidence_index,
    verify_evidence_index,
)


class ArtifactIndexTest(unittest.TestCase):
    def test_builds_and_verifies_custom_claim_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidence").mkdir()
            (root / "evidence" / "a.json").write_text(
                json.dumps({"value": 1}), encoding="utf-8"
            )
            claims = (
                {
                    "claim_id": "claim-a",
                    "statement": "A test claim.",
                    "patterns": ("evidence/*.json",),
                },
            )

            index = build_evidence_index(root, claims)
            verification = verify_evidence_index(index, root)

            self.assertEqual(index["summary"]["claims"], 1)
            self.assertEqual(index["summary"]["artifacts"], 1)
            self.assertTrue(verification["passed"])

            (root / "evidence" / "a.json").write_text("changed", encoding="utf-8")
            changed = verify_evidence_index(index, root)
            self.assertFalse(changed["passed"])
            self.assertEqual(changed["mismatches"][0]["reason"], "content-mismatch")

    def test_missing_pattern_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            claims = (
                {
                    "claim_id": "missing",
                    "statement": "Missing evidence.",
                    "patterns": ("does-not-exist.json",),
                },
            )
            with self.assertRaises(FileNotFoundError):
                build_evidence_index(Path(directory), claims)


if __name__ == "__main__":
    unittest.main()
