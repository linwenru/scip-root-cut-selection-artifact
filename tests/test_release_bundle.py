import tempfile
import unittest
from pathlib import Path

from scip_cut_trace_v2.release_bundle import (
    RELEASE_TIERS,
    build_release_inventory,
    verify_release_bundle,
    write_release_bundle,
)


class ReleaseBundleTest(unittest.TestCase):
    def test_source_tier_contains_publication_metadata_and_locks(self):
        patterns = set(RELEASE_TIERS["source"])

        self.assertTrue(
            {
                ".gitignore",
                ".zenodo.json",
                "CITATION.cff",
                "LICENSE*",
                "models/ranking_imitation_xgb_v1/model.ubj",
                "NOTICE",
                "requirements-core.lock",
            }
            <= patterns
        )

    def test_public_bundle_excludes_private_manuscript_material(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "paper").mkdir()
            (root / "docs").mkdir()
            (root / "README.md").write_text("public\n", encoding="utf-8")
            (root / "paper" / "manuscript.pdf").write_text(
                "private manuscript\n", encoding="utf-8"
            )
            (root / "docs" / "MAJOR_REVISION_PLAN.md").write_text(
                "private review notes\n", encoding="utf-8"
            )
            (root / "docs" / "REPRODUCIBILITY.md").write_text(
                "public protocol\n", encoding="utf-8"
            )

            inventory = build_release_inventory(
                root,
                ("source",),
                {
                    "source": (
                        "README.md",
                        "paper/**/*",
                        "docs/**/*.md",
                    )
                },
            )

            self.assertEqual(
                [record["path"] for record in inventory["files"]],
                ["README.md", "docs/REPRODUCIBILITY.md"],
            )

    def test_bundle_is_deterministic_and_verifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            (root / "results").mkdir()
            (root / "source" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "results" / "run.json").write_text('{"status":"ok"}\n', encoding="utf-8")
            patterns = {
                "source": ("source/*.py",),
                "results": ("results/*.json",),
            }
            inventory = build_release_inventory(
                root, ("source", "results"), patterns
            )
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"

            first_record = write_release_bundle(first, inventory, root)
            second_record = write_release_bundle(second, inventory, root)

            self.assertEqual(first_record["sha256"], second_record["sha256"])
            self.assertEqual(inventory["summary"]["files"], 2)
            self.assertTrue(verify_release_bundle(first)["passed"])

    def test_unknown_or_empty_tier_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                build_release_inventory(root, ("unknown",), {})
            with self.assertRaises(FileNotFoundError):
                build_release_inventory(root, ("empty",), {"empty": ("*.json",)})

    def test_external_release_inventory_is_not_self_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = root / "data" / "manifests"
            manifests.mkdir(parents=True)
            (manifests / "experiment.json").write_text("{}\n", encoding="utf-8")
            (manifests / "release_core_v1.json").write_text(
                '{"archive":"old"}\n', encoding="utf-8"
            )

            inventory = build_release_inventory(
                root,
                ("manifests",),
                {"manifests": ("data/manifests/**/*.json",)},
            )

            self.assertEqual(
                [record["path"] for record in inventory["files"]],
                ["data/manifests/experiment.json"],
            )


if __name__ == "__main__":
    unittest.main()
