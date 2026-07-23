import copy
import unittest

from scip_cut_trace_v2.causal_confirmation import evaluate_confirmation


def _plan():
    return {
        "experiment_contract": {
            "actions": ["efficacy-promote"],
            "seeds": [0, 1],
            "time_limit_seconds": 120,
            "node_limit": None,
            "intervention_scope": "first-run-only",
        },
        "confirmation_gate": {
            "minimum_attributable_pairs": 3,
            "passing_consequence": "advance",
            "failure_consequence": "stop",
        },
        "instances": [{"instance_id": "a"}, {"instance_id": "b"}],
    }


def _pair(seed, saving, native_status="optimal", treatment_status="optimal", eligible=True):
    valid = eligible and native_status == "optimal" and treatment_status == "optimal"
    return {
        "seed": seed,
        "native_outcome": {"status": native_status, "lp_iterations": 100},
        "initial_context": {"matching_across_actions": True},
        "actions": {
            "efficacy-promote": {
                "outcome": {"status": treatment_status, "lp_iterations": 100},
                "selector": {"interventions": int(eligible), "context_records": [{}]},
                "comparison": {
                    "eligible": eligible,
                    "valid": valid,
                    "metrics": {"lp_iterations": {"relative_saving": saving}},
                },
            }
        },
    }


def _experiment():
    return {
        "actions": ["efficacy-promote"],
        "seeds": [0, 1],
        "time_limit": 120,
        "node_limit": None,
        "intervention_scope": "first-run-only",
        "per_instance": [
            {"instance_id": "a", "pairs": [_pair(0, 0.2), _pair(1, 0.1)]},
            {
                "instance_id": "b",
                "pairs": [_pair(0, 0.0, eligible=False), _pair(1, 0.0, eligible=False)],
            },
        ],
    }


class CausalConfirmationTest(unittest.TestCase):
    def test_passes_safe_positive_fixed_action(self):
        result = evaluate_confirmation(_plan(), _experiment())

        self.assertTrue(result["confirmation_gate"]["passed"])
        self.assertEqual(result["data"]["attributable_pairs"], 4)
        self.assertEqual(result["data"]["interventions"], 2)
        self.assertEqual(result["fixed_action_summary"]["instance_wins"], 1)
        self.assertEqual(result["fixed_action_summary"]["instance_ties"], 1)

    def test_only_completed_native_arm_can_create_attributable_risk(self):
        experiment = _experiment()
        experiment["per_instance"][0]["pairs"][0] = _pair(
            0, 0.0, native_status="timelimit", treatment_status="timelimit"
        )
        experiment["per_instance"][0]["pairs"][1] = _pair(
            1, 0.0, native_status="optimal", treatment_status="timelimit"
        )

        result = evaluate_confirmation(_plan(), experiment)

        self.assertEqual(result["data"]["native_incomplete_pairs"], 1)
        self.assertEqual(result["data"]["attributable_unsafe_pairs"], 1)
        self.assertFalse(result["confirmation_gate"]["passed"])

    def test_rejects_experiment_that_differs_from_plan(self):
        experiment = copy.deepcopy(_experiment())
        experiment["seeds"] = [1, 0]

        with self.assertRaisesRegex(ValueError, "seeds"):
            evaluate_confirmation(_plan(), experiment)


if __name__ == "__main__":
    unittest.main()
