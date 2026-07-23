import math
import unittest

from scip_cut_trace_v2.paper_statistics import (
    analyze_manifest,
    cluster_bootstrap_geometric_ratio,
    exact_binomial_upper_bound,
)


def _outcome(solving_time, status="optimal"):
    return {
        "status": status,
        "objective_sense": "minimize",
        "primal_bound": 10.0,
        "dual_bound": 10.0,
        "solving_time": solving_time,
        "nodes": 10,
        "total_nodes": 10,
        "lp_iterations": 100,
        "lp_count": 20,
        "cuts_applied": 5,
        "primal_dual_integral": 2.0,
    }


def _instance(instance_id, native_time, treatment_time, treatment_status="optimal"):
    intervention = treatment_status == "optimal"
    return {
        "instance_id": instance_id,
        "instance_sha256": f"sha-{instance_id}",
        "pairs": [
            {
                "seed": 0,
                "native_outcome": _outcome(native_time),
                "actions": {
                    "efficacy-rank": {
                        "outcome": _outcome(treatment_time, treatment_status),
                        "selector": {"interventions": int(intervention)},
                        "comparison": {"policy_fallback": not intervention},
                    }
                },
            }
        ],
    }


def _status_instance(
    instance_id,
    native_time,
    native_status,
    treatment_time,
    treatment_status,
    *,
    intervention=True,
):
    return {
        "instance_id": instance_id,
        "instance_sha256": f"sha-{instance_id}",
        "pairs": [
            {
                "seed": 0,
                "native_outcome": _outcome(native_time, native_status),
                "actions": {
                    "efficacy-rank": {
                        "outcome": _outcome(treatment_time, treatment_status),
                        "selector": {"interventions": int(intervention)},
                        "comparison": {"policy_fallback": not intervention},
                    }
                },
            }
        ],
    }


class PaperStatisticsTest(unittest.TestCase):
    def test_zero_failure_exact_upper_bound_uses_instance_count(self):
        upper = exact_binomial_upper_bound(0, 59, confidence=0.95)

        self.assertLess(upper, 0.05)
        self.assertAlmostEqual(upper, 1.0 - 0.05 ** (1.0 / 59.0), places=12)

    def test_bootstrap_is_deterministic_and_on_ratio_scale(self):
        values = [math.log(0.8), math.log(1.0), math.log(1.2)]

        first = cluster_bootstrap_geometric_ratio(values, 500, seed=17)
        repeated = cluster_bootstrap_geometric_ratio(values, 500, seed=17)

        self.assertEqual(first, repeated)
        self.assertGreater(first["lower"], 0.0)
        self.assertGreater(first["upper"], first["lower"])

    def test_analysis_averages_seeds_within_instance_and_instances_equally(self):
        manifest = {
            "time_limit": 100.0,
            "actions": ["efficacy-rank"],
            "per_instance": [
                _instance("a", 10.0, 8.0),
                _instance("b", 10.0, 9.0),
                _instance("c", 10.0, 11.0),
            ],
        }

        analysis = analyze_manifest(
            manifest, bootstrap_replicates=500, bootstrap_seed=3
        )["actions"]["efficacy-rank"]

        expected_ratio = (0.8 * 0.9 * 1.1) ** (1.0 / 3.0)
        self.assertAlmostEqual(
            analysis["penalized_time"][
                "geometric_mean_ratio_treatment_over_native"
            ],
            expected_ratio,
        )
        self.assertEqual(analysis["penalized_time"]["instance_wins"], 2)
        self.assertEqual(analysis["penalized_time"]["instance_losses"], 1)
        self.assertEqual(analysis["safety_failure_instances"], 0)

    def test_treatment_timeout_is_par2_and_fails_safety(self):
        manifest = {
            "time_limit": 100.0,
            "actions": ["efficacy-rank"],
            "per_instance": [
                _instance(
                    "timeout", 10.0, 100.0, treatment_status="timelimit"
                )
            ],
        }

        analysis = analyze_manifest(
            manifest, bootstrap_replicates=200, bootstrap_seed=3
        )["actions"]["efficacy-rank"]

        instance = analysis["per_instance"][0]
        self.assertAlmostEqual(
            instance["geometric_mean_penalized_time_ratio"], 20.0
        )
        self.assertEqual(analysis["safety_failure_pairs"], 1)
        self.assertEqual(analysis["safety_failure_instances"], 1)
        self.assertFalse(analysis["gate"]["passed"])

    def test_primary_itt_keeps_native_timeout_treatment_completion(self):
        manifest = {
            "time_limit": 100.0,
            "actions": ["efficacy-rank"],
            "per_instance": [
                _status_instance(
                    "native-timeout",
                    100.0,
                    "timelimit",
                    50.0,
                    "optimal",
                )
            ],
        }

        analysis = analyze_manifest(
            manifest, bootstrap_replicates=200, bootstrap_seed=3
        )["actions"]["efficacy-rank"]

        self.assertAlmostEqual(
            analysis["penalized_time"][
                "geometric_mean_ratio_treatment_over_native"
            ],
            0.25,
        )
        self.assertEqual(analysis["penalized_time"]["pairs"], 1)
        self.assertEqual(analysis["penalized_time"]["instances"], 1)
        self.assertEqual(
            analysis["outcome_pair_counts"][
                "native_incomplete_treatment_complete"
            ],
            1,
        )
        self.assertEqual(analysis["native_complete_secondary"]["pairs"], 0)
        self.assertIsNone(
            analysis["native_complete_secondary"][
                "geometric_mean_ratio_treatment_over_native"
            ]
        )

    def test_primary_itt_keeps_both_timeouts_and_policy_fallbacks(self):
        manifest = {
            "time_limit": 100.0,
            "actions": ["efficacy-rank"],
            "per_instance": [
                _status_instance(
                    "both-timeout",
                    100.0,
                    "timelimit",
                    100.0,
                    "timelimit",
                    intervention=False,
                )
            ],
        }

        analysis = analyze_manifest(
            manifest, bootstrap_replicates=200, bootstrap_seed=3
        )["actions"]["efficacy-rank"]

        self.assertAlmostEqual(
            analysis["penalized_time"][
                "geometric_mean_ratio_treatment_over_native"
            ],
            1.0,
        )
        self.assertEqual(analysis["penalized_time"]["instance_ties"], 1)
        self.assertEqual(analysis["policy_fallback_pairs"], 1)
        self.assertEqual(analysis["outcome_pair_counts"]["both_incomplete"], 1)

    def test_conditional_policy_safety_includes_fallbacks_without_attribution(self):
        manifest = {
            "time_limit": 100.0,
            "actions": ["efficacy-rank"],
            "per_instance": [
                _status_instance(
                    "fallback",
                    10.0,
                    "optimal",
                    10.0,
                    "optimal",
                    intervention=False,
                )
            ],
        }

        full = analyze_manifest(
            manifest, bootstrap_replicates=200, bootstrap_seed=3
        )
        analysis = full["actions"]["efficacy-rank"]
        safety = analysis["conditional_policy_safety"]

        self.assertTrue(safety["includes_policy_fallbacks"])
        self.assertFalse(safety["causal_attribution_to_selection_change"])
        self.assertEqual(safety["instances"], 1)
        self.assertEqual(safety["policy_fallback_pairs_in_population_instances"], 1)
        self.assertEqual(safety["failure_instances"], 0)
        self.assertFalse(
            full["analysis_contract"][
                "full_arm_wall_time_sensitivity_available"
            ]
        )

    def test_secondary_solver_metrics_are_instance_equal_and_descriptive(self):
        first = _instance("a", 10.0, 8.0)
        second = _instance("b", 10.0, 9.0)
        first["pairs"][0]["native_outcome"]["lp_iterations"] = 100
        first["pairs"][0]["actions"]["efficacy-rank"]["outcome"][
            "lp_iterations"
        ] = 80
        second["pairs"][0]["native_outcome"]["lp_iterations"] = 200
        second["pairs"][0]["actions"]["efficacy-rank"]["outcome"][
            "lp_iterations"
        ] = 260
        manifest = {
            "time_limit": 100.0,
            "actions": ["efficacy-rank"],
            "per_instance": [first, second],
        }

        analysis = analyze_manifest(
            manifest, bootstrap_replicates=200, bootstrap_seed=3
        )["actions"]["efficacy-rank"]
        metric = analysis["descriptive_solver_metrics"]["lp_iterations"]

        self.assertEqual(metric["instances"], 2)
        self.assertEqual(metric["pairs"], 2)
        self.assertAlmostEqual(metric["native_instance_equal_mean"], 150.0)
        self.assertAlmostEqual(metric["treatment_instance_equal_mean"], 170.0)
        self.assertAlmostEqual(
            metric["ratio_of_instance_equal_means_treatment_over_native"],
            170.0 / 150.0,
        )
        self.assertAlmostEqual(metric["mean_delta_treatment_minus_native"], 20.0)
        self.assertEqual(metric["instances_treatment_lower"], 1)
        self.assertEqual(metric["instances_treatment_higher"], 1)
        self.assertIn("no censoring correction", metric["censoring_note"])


if __name__ == "__main__":
    unittest.main()
