import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scip_cut_trace_v2.causal_context import context_sha256
from scip_cut_trace_v2.causal_dataset import build_dataset, build_records


ACTIONS = ("boundary-swap", "boundary-swap-2", "efficacy-promote")


def _experiment(root, context):
    digest = context_sha256(context)
    action_results = {}
    for action in ACTIONS:
        result_path = root / f"{action}.json"
        raw = {
            "selector": {
                "context_records": [
                    {
                        "context_sha256": digest,
                        "decision_context": context,
                    }
                ]
            }
        }
        result_path.write_text(json.dumps(raw), encoding="utf-8")
        action_results[action] = {
            "result_path": str(result_path),
            "initial_context_sha256": digest,
            "outcome": {"status": "optimal"},
            "selector": {"interventions": 1},
            "comparison": {
                "safe": True,
                "eligible": True,
                "valid": True,
                "metrics": {
                    "lp_iterations": {
                        "native": 10.0,
                        "treatment": 8.0,
                        "delta_treatment_minus_native": -2.0,
                        "relative_saving": 0.2,
                    }
                },
            },
        }
    return {
        "intervention_scope": "first-run-only",
        "actions": list(ACTIONS),
        "per_instance": [
            {
                "instance_id": "tiny",
                "instance": "/tmp/tiny.mps",
                "instance_sha256": "instance-hash",
                "pairs": [
                    {
                        "seed": 0,
                        "initial_context": {"matching_across_actions": True},
                        "native_outcome": {"status": "optimal"},
                        "actions": action_results,
                        "oracle": {"selected_action": "boundary-swap"},
                    }
                ],
            }
        ],
    }


class CausalDatasetTest(unittest.TestCase):
    def test_builds_one_shared_context_with_multi_action_labels(self):
        context = {
            "schema_version": 1,
            "solver_state": {"node_depth": 0},
            "candidates": [{"native_rank": 0}],
            "forced_candidates": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "experiment.json"
            source.write_text(json.dumps(_experiment(root, context)), encoding="utf-8")
            output = root / "dataset.jsonl.gz"
            manifest_path = root / "manifest.json"

            manifest = build_dataset(source, output, manifest_path, "train")

            with gzip.open(output, "rt", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(manifest["records"], 1)
            self.assertEqual(manifest["skipped_no_context_pairs"], 0)
            self.assertEqual(manifest["skipped_native_incomplete_pairs"], 0)
            self.assertEqual(records[0]["context"], context)
            self.assertEqual(set(records[0]["action_labels"]), set(ACTIONS))
            self.assertEqual(
                manifest["action_label_counts"]["boundary-swap"][
                    "positive_lp_saving"
                ],
                1,
            )

    def test_rejects_post_outcome_field_in_context(self):
        context = {
            "solver_state": {"node_depth": 0},
            "candidates": [],
            "forced_candidates": [],
            "solving_time": 1.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            with self.assertRaisesRegex(ValueError, "Post-outcome"):
                build_records(_experiment(root, context), "train")

    def test_skips_seed_with_no_cut_selector_context(self):
        context = {
            "solver_state": {"node_depth": 0},
            "candidates": [],
            "forced_candidates": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = _experiment(root, context)
            experiment["per_instance"][0]["pairs"][0]["initial_context"] = {
                "matching_across_actions": False,
                "no_action_observed": True,
                "partial_actions_observed": False,
            }

            self.assertEqual(build_records(experiment, "train"), [])

    def test_skips_seed_when_native_did_not_complete(self):
        context = {
            "solver_state": {"node_depth": 0},
            "candidates": [],
            "forced_candidates": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = _experiment(root, context)
            experiment["per_instance"][0]["pairs"][0]["native_outcome"][
                "status"
            ] = "timelimit"

            self.assertEqual(build_records(experiment, "train"), [])

    def test_combines_disjoint_source_manifests(self):
        context = {
            "solver_state": {"node_depth": 0},
            "candidates": [],
            "forced_candidates": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / "first"
            second_root = root / "second"
            first_root.mkdir()
            second_root.mkdir()
            first_experiment = _experiment(first_root, context)
            second_experiment = _experiment(second_root, context)
            second_experiment["per_instance"][0]["instance_id"] = "other"
            second_experiment["per_instance"][0]["instance"] = "/tmp/other.mps"
            first_source = root / "first.json"
            second_source = root / "second.json"
            first_source.write_text(json.dumps(first_experiment), encoding="utf-8")
            second_source.write_text(json.dumps(second_experiment), encoding="utf-8")

            manifest = build_dataset(
                [first_source, second_source],
                root / "combined.jsonl.gz",
                root / "combined-manifest.json",
                "train",
            )

            self.assertEqual(manifest["records"], 2)
            self.assertEqual(len(manifest["source_manifests"]), 2)


if __name__ == "__main__":
    unittest.main()
