import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scip_cut_trace_v2.learned_policy_pilot import (
    ACTIVE_ARM,
    ARMS,
    SHADOW_ARM,
    _schedule_position_counts,
    _download_one_instance,
    _parse_content_range,
    analyze_result,
    build_balanced_schedule,
    build_download_plan,
    build_plan,
    select_pilot_candidates,
    PilotCandidate,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LearnedPolicyPilotTest(unittest.TestCase):
    def test_content_range_parser_rejects_non_range_response(self):
        self.assertEqual(_parse_content_range("bytes 10-19/100"), (10, 19, 100))
        with self.assertRaises(ValueError):
            _parse_content_range("100")

    def test_range_downloader_verifies_and_reuses_complete_gzip(self):
        payload = gzip.compress(b"NAME test\nROWS\nENDATA\n" * 100)

        class Response:
            def __init__(self, data, headers, status):
                self._data = data
                self.headers = headers
                self.status = status

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return self._data

            def getcode(self):
                return self.status

        calls = []

        def open_http(request, _timeout):
            calls.append(request)
            if request.get_method() == "HEAD":
                return Response(
                    b"",
                    {
                        "Content-Length": str(len(payload)),
                        "ETag": '"test"',
                        "Last-Modified": "today",
                    },
                    200,
                )
            start, end = (
                int(part)
                for part in request.headers["Range"].removeprefix("bytes=").split("-")
            )
            return Response(
                payload[start : end + 1],
                {"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
                206,
            )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "instance.mps.gz"
            first = _download_one_instance(
                "https://example.test/instance.mps.gz",
                destination,
                chunk_bytes=37,
                retries=2,
                timeout=1.0,
                open_http=open_http,
            )
            range_calls = len(calls) - 1
            second = _download_one_instance(
                "https://example.test/instance.mps.gz",
                destination,
                chunk_bytes=37,
                retries=2,
                timeout=1.0,
                open_http=open_http,
            )

            self.assertGreater(range_calls, 1)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(len(calls), range_calls + 2)

    def test_selection_is_deterministic_and_group_unique(self):
        candidates = [
            PilotCandidate(
                instance_id=f"i{index}",
                instance_name=f"i{index}.mps.gz",
                instance=Path(f"i{index}.mps.gz"),
                status="easy",
                official_group="shared" if index < 2 else f"g{index}",
                tags=(),
            )
            for index in range(20)
        ]

        first = select_pilot_candidates(candidates, count=18, key="test")
        second = select_pilot_candidates(reversed(candidates), count=18, key="test")

        self.assertEqual(first, second)
        self.assertEqual(len({item.group_key for item in first}), 18)

    def test_three_arm_schedule_is_position_balanced(self):
        instances = [{"instance_id": f"i{index}"} for index in range(18)]

        schedule = build_balanced_schedule(instances, [0, 1, 2], key="test")
        counts = _schedule_position_counts(schedule)

        self.assertEqual(len(schedule), 54)
        self.assertEqual(set(counts), set(ARMS))
        self.assertTrue(
            all(count == 18 for positions in counts.values() for count in positions.values())
        )

    def test_plan_excludes_every_existing_development_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instances = root / "instances"
            instances.mkdir()
            collection = root / "collection.csv"
            development = root / "development.csv"
            split = root / "train.test"
            model = root / "model.ubj"
            model.write_bytes(b"model")
            manifest = root / "model.json"
            manifest.write_text(
                json.dumps(
                    {
                        "model": {"path": str(model), "sha256": _sha256(model)},
                        "stage_gate": {"passed": False},
                    }
                ),
                encoding="utf-8",
            )
            development.write_text(
                "instance_name,status,group,tags\nold.mps.gz,easy,used,binary\n",
                encoding="utf-8",
            )
            split.write_text("old.mps.gz\n", encoding="utf-8")
            rows = ["instance_name,status,group,tags"]
            rows.append("same-family.mps.gz,easy,used,binary")
            for index in range(20):
                name = f"new-{index}.mps.gz"
                (instances / name).write_bytes(
                    gzip.compress(f"instance-{index}".encode())
                )
                rows.append(f"{name},easy,new-group-{index},binary")
            collection.write_text("\n".join(rows) + "\n", encoding="utf-8")

            plan = build_plan(
                collection,
                development,
                [split],
                instances,
                manifest,
                root / "output",
                root / "result.json",
                root / "statistics.json",
            )

            self.assertTrue(all(plan["checks"].values()))
            self.assertEqual(len(plan["experiment"]["instances"]), 18)
            self.assertNotIn(
                "used", {item["group_key"] for item in plan["experiment"]["instances"]}
            )

            mismatched_download_plan = root / "download-plan.json"
            mismatched_download_plan.write_text(
                json.dumps(
                    {
                        "instances": list(
                            reversed(
                                [
                                    {"instance_id": item["instance_id"]}
                                    for item in plan["experiment"]["instances"]
                                ]
                            )
                        )
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "download-plan selection"):
                build_plan(
                    collection,
                    development,
                    [split],
                    instances,
                    manifest,
                    root / "output",
                    root / "result.json",
                    root / "statistics.json",
                    download_plan_path=mismatched_download_plan,
                )

    def test_download_plan_does_not_require_local_instance_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = root / "collection.csv"
            development = root / "development.csv"
            split = root / "train.test"
            development.write_text(
                "instance_name,status,group,tags\nold.mps.gz,easy,used,binary\n",
                encoding="utf-8",
            )
            split.write_text("old.mps.gz\n", encoding="utf-8")
            rows = ["instance_name,status,group,tags"] + [
                f"new-{index}.mps.gz,easy,new-group-{index},binary"
                for index in range(20)
            ]
            collection.write_text("\n".join(rows) + "\n", encoding="utf-8")

            plan = build_download_plan(
                collection, development, [split], root / "missing-instances"
            )

            self.assertTrue(all(plan["checks"].values()))
            self.assertEqual(len(plan["instances"]), 18)
            self.assertTrue(
                all(record["url"].endswith(".mps.gz") for record in plan["instances"])
            )

    def test_analysis_passes_only_the_frozen_joint_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            result_path = root / "result.json"
            model_manifest = root / "model.json"
            model_manifest.write_text("{}\n", encoding="utf-8")
            instances = [
                {"instance_id": f"i{index}", "group_key": f"g{index}"}
                for index in range(18)
            ]
            plan = {
                "status": "pre_registered_before_any_new_group_online_outcomes",
                "experiment": {
                    "instances": instances,
                    "seeds": [0, 1, 2],
                    "time_limit_seconds": 300.0,
                    "intervention_scope": "first-run-only",
                },
                "source_artifacts": {
                    "learned_model_manifest": {"sha256": _sha256(model_manifest)}
                },
                "analysis_contract": {
                    "primary_estimand": "test",
                    "bootstrap_replicates": 100,
                    "bootstrap_seed": 7,
                    "go_no_go_thresholds": {
                        "minimum_group_keys_with_active_intervention": 12,
                        "maximum_active_correctness_failure_group_keys": 0,
                        "maximum_native_complete_active_incomplete_group_keys": 0,
                        "required_shadow_structural_matches": 54,
                        "maximum_primary_point_ratio": 0.95,
                        "maximum_one_sided_80_percent_upper_ratio": 1.0,
                    },
                },
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            common_checks = {
                "arm_order": True,
                "same_instance_sha256": True,
                "same_seed": True,
                "same_parameters": True,
                "same_runtime_versions": True,
                "same_objective_sense": True,
                "known_intervention_scope": True,
                "run_budget_respected": True,
                "one_record_per_intervention": True,
                "context_budget_respected": True,
                "interventions_have_context": True,
                "same_status": True,
                "same_primal_bound": True,
                "same_dual_bound": True,
            }
            per_instance = []
            for index in range(18):
                pairs = []
                for seed in range(3):
                    native = {
                        "status": "optimal",
                        "arm_wall_time_seconds": 100.0,
                        "solving_time": 99.0,
                    }
                    action_outcome = {
                        "status": "optimal",
                        "arm_wall_time_seconds": 90.0,
                        "solving_time": 89.0,
                    }
                    pairs.append(
                        {
                            "native_outcome": native,
                            "initial_context": {
                                "all_actions_observed": True,
                                "matching_across_actions": True,
                            },
                            "actions": {
                                SHADOW_ARM: {
                                    "comparison": {"safe": True},
                                    "selector": {"interventions": 0},
                                    "outcome": native,
                                },
                                ACTIVE_ARM: {
                                    "comparison": {"safety_checks": common_checks},
                                    "selector": {"interventions": 1},
                                    "outcome": action_outcome,
                                },
                            },
                        }
                    )
                per_instance.append({"instance_id": f"i{index}", "pairs": pairs})
            result = {
                "actions": [SHADOW_ARM, ACTIVE_ARM],
                "seeds": [0, 1, 2],
                "time_limit": 300.0,
                "intervention_scope": "first-run-only",
                "learned_model_manifest_sha256": _sha256(model_manifest),
                "per_instance": per_instance,
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")

            decision = analyze_result(result_path, plan_path, bootstrap_replicates=100)

            self.assertTrue(decision["passed"])
            self.assertAlmostEqual(
                decision["primary_full_process_wall_time"]["active_native_par2_ratio"],
                0.9,
            )


if __name__ == "__main__":
    unittest.main()
