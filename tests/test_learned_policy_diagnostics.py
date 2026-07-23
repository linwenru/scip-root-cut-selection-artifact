import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scip_cut_trace_v2.learned_policy_diagnostics import build_diagnostics
from scip_cut_trace_v2.learned_policy_pilot import ACTIVE_ARM, SHADOW_ARM


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LearnedPolicyDiagnosticsTest(unittest.TestCase):
    def test_post_hoc_diagnostic_does_not_reverse_frozen_no_go(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            plan_path = root / "plan.json"
            statistics_path = root / "statistics.json"
            collection_html = root / "collection.html"
            collection_html.write_text(
                """
                <table id="miplibtable"><tbody><tr>
                <td>a</td><td>1</td><td>1</td><td>0</td><td>0</td>
                <td>1</td><td>1</td><td>Alice</td><td>group-a</td>
                <td>easy</td><td>2</td>
                </tr></tbody></table>
                """,
                encoding="utf-8",
            )
            plan_path.write_text(
                json.dumps(
                    {
                        "experiment": {
                            "time_limit_seconds": 10.0,
                            "instances": [{"instance_id": "a", "group_key": "group-a"}],
                        }
                    }
                ),
                encoding="utf-8",
            )
            native = {
                "status": "optimal",
                "arm_wall_time_seconds": 1.0,
                "solving_time": 0.8,
                "primal_bound": 1.0,
                "dual_bound": 1.0,
            }
            shadow = dict(native)
            active = {
                **native,
                "arm_wall_time_seconds": 2.0,
                "solving_time": 1.0,
                "primal_bound": 2.0,
                "dual_bound": 2.0,
            }
            mechanical = {
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
                "same_primal_bound": False,
                "same_dual_bound": False,
            }
            timing = {
                "model_load": 0.2,
                "policy_compute": 0.01,
            }
            result_path.write_text(
                json.dumps(
                    {
                        "per_instance": [
                            {
                                "instance_id": "a",
                                "pairs": [
                                    {
                                        "seed": 0,
                                        "native_outcome": native,
                                        "initial_context": {
                                            "all_actions_observed": True,
                                            "matching_across_actions": True,
                                        },
                                        "actions": {
                                            SHADOW_ARM: {
                                                "outcome": shadow,
                                                "comparison": {
                                                    "safe": True,
                                                    "safety_checks": {
                                                        "native_complete": True
                                                    },
                                                },
                                                "selector": {
                                                    "interventions": 0,
                                                    "timing_seconds": timing,
                                                },
                                            },
                                            ACTIVE_ARM: {
                                                "outcome": active,
                                                "comparison": {
                                                    "safety_checks": mechanical
                                                },
                                                "selector": {
                                                    "interventions": 1,
                                                    "timing_seconds": timing,
                                                },
                                            },
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            statistics_path.write_text(
                json.dumps(
                    {
                        "source_result_sha256": _sha256(result_path),
                        "source_plan_sha256": _sha256(plan_path),
                        "passed": False,
                        "decision": "stop",
                        "gate_checks": {"primary": False},
                        "primary_full_process_wall_time": {"ratio": 2.0},
                        "secondary_scip_solving_time": {"ratio": 1.25},
                    }
                ),
                encoding="utf-8",
            )

            diagnostics = build_diagnostics(
                result_path, plan_path, statistics_path, collection_html
            )

            self.assertFalse(diagnostics["frozen_decision"]["passed"])
            mismatch = diagnostics["active_safety_diagnostic"][
                "objective_mismatch_pairs"
            ][0]
            self.assertEqual(mismatch["closer_to_official"], "active")
            self.assertEqual(
                diagnostics["shadow_diagnostic"][
                    "completed_pairs_with_full_safety_match"
                ],
                1,
            )
            self.assertEqual(
                diagnostics["coverage"]["active_intervention_pairs"], 1
            )


if __name__ == "__main__":
    unittest.main()
