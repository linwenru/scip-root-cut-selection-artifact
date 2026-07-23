import copy
import unittest

from scip_cut_trace_v2.fixed_baseline_pilot import evaluate_pilot_gate


def _artifacts(*, schema_version=1, instance_count=5):
    actions = ["random-rank", "efficacy-rank", "adaptive-score"]
    instances = [{"instance_id": f"i{index}"} for index in range(instance_count)]
    plan = {
        "schema_version": schema_version,
        "status": "pre_registered_before_active_outcomes",
        "experiment_contract": {
            "actions": actions,
            "seeds": [0, 1, 2],
            "time_limit_seconds": 30,
            "node_limit": None,
            "intervention_scope": "first-run-only",
        },
        "instances": instances,
    }
    causal = {
        "actions": actions,
        "seeds": [0, 1, 2],
        "time_limit": 30,
        "node_limit": None,
        "intervention_scope": "first-run-only",
        "per_instance": instances,
    }
    statistics = {"actions": {}}
    for action in actions:
        statistics["actions"][action] = {
            "native_complete_pairs": instance_count * 3,
            "safety_failure_instances": 0,
            "per_instance": [
                {"instance_id": f"i{index}", "intervention_pairs": 1}
                for index in range(5)
            ],
            "penalized_time": {
                "geometric_mean_ratio_treatment_over_native": 0.98,
                "instance_wins": instance_count - 2,
                "instance_losses": 2,
                "cluster_bootstrap_interval": {
                    "lower": 0.94,
                    "upper": 0.99,
                },
            },
        }
    return plan, causal, statistics


class FixedBaselinePilotTest(unittest.TestCase):
    def test_selects_lowest_ratio_then_frozen_tie_order(self):
        plan, causal, statistics = _artifacts()
        statistics["actions"]["efficacy-rank"]["penalized_time"][
            "geometric_mean_ratio_treatment_over_native"
        ] = 0.95
        statistics["actions"]["adaptive-score"]["penalized_time"][
            "geometric_mean_ratio_treatment_over_native"
        ] = 0.95

        decision = evaluate_pilot_gate(plan, causal, statistics)

        self.assertTrue(decision["passed"])
        self.assertEqual(
            decision["selected_action_for_larger_train_cohort"], "efficacy-rank"
        )

    def test_no_action_advances_after_a_gating_failure(self):
        plan, causal, statistics = _artifacts()
        for result in statistics["actions"].values():
            result["native_complete_pairs"] = 14

        decision = evaluate_pilot_gate(plan, causal, statistics)

        self.assertFalse(decision["passed"])
        self.assertIsNone(decision["selected_action_for_larger_train_cohort"])

    def test_rejects_artifact_that_does_not_match_plan(self):
        plan, causal, statistics = _artifacts()
        mismatched = copy.deepcopy(causal)
        mismatched["time_limit"] = 60

        with self.assertRaises(ValueError):
            evaluate_pilot_gate(plan, mismatched, statistics)

    def test_v2_requires_bootstrap_upper_bound_below_one(self):
        plan, causal, statistics = _artifacts(schema_version=2, instance_count=8)
        for result in statistics["actions"].values():
            result["penalized_time"]["cluster_bootstrap_interval"]["upper"] = 1.01

        decision = evaluate_pilot_gate(plan, causal, statistics)

        self.assertFalse(decision["passed"])
        self.assertEqual(decision["minimum_intervention_instances"], 5)
        self.assertFalse(
            decision["actions"]["random-rank"]["checks"][
                "penalized_time_ci95_upper_below_one"
            ]
        )

    def test_v2_requires_selected_set_changes_on_five_instances(self):
        plan, causal, statistics = _artifacts(schema_version=2, instance_count=8)
        for result in statistics["actions"].values():
            for instance in result["per_instance"][4:]:
                instance["intervention_pairs"] = 0

        decision = evaluate_pilot_gate(plan, causal, statistics)

        self.assertFalse(decision["passed"])
        self.assertFalse(
            decision["actions"]["random-rank"]["checks"][
                "selected_set_changes_on_at_least_5_instances"
            ]
        )


if __name__ == "__main__":
    unittest.main()
