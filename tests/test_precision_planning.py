import unittest

from scip_cut_trace_v2.precision_planning import (
    build_precision_plan,
    minimum_detectable_ratio,
    required_paired_instances,
    required_paired_instances_for_log_ratio_test,
    required_zero_failure_instances,
)


class PrecisionPlanningTest(unittest.TestCase):
    def test_zero_failure_requirement_matches_exact_upper_bound(self):
        self.assertEqual(required_zero_failure_instances(0.05), 59)
        self.assertEqual(required_zero_failure_instances(0.10), 29)

    def test_more_instances_detect_smaller_effects(self):
        small = minimum_detectable_ratio(0.37, 35)
        large = minimum_detectable_ratio(0.37, 100)

        self.assertLess(small, large)
        self.assertGreater(
            required_paired_instances(0.37, 0.95),
            required_paired_instances(0.37, 0.90),
        )

    def test_superiority_and_noninferiority_are_distinct_designs(self):
        sigma = 0.3741327367245724

        self.assertEqual(required_paired_instances(sigma, 0.90), 78)
        self.assertEqual(
            required_paired_instances_for_log_ratio_test(
                sigma,
                null_ratio=1.05,
                alternative_ratio=1.00,
            ),
            364,
        )

    def test_current_test_is_not_adequate_for_proposed_targets(self):
        plan = build_precision_plan()

        self.assertEqual(plan["available_sealed_test"]["instances"], 35)
        self.assertEqual(
            plan["available_sealed_test"]["distinct_group_keys"], 34
        )
        self.assertEqual(
            plan["available_sealed_test"]["repeated_test_group_keys"],
            {"square": ["square41.mps.gz", "square47.mps.gz"]},
        )
        self.assertEqual(
            plan["available_sealed_test"]["group_keys_seen_in_training"],
            10,
        )
        self.assertEqual(
            plan["available_sealed_test"]["group_keys_unseen_in_training"],
            24,
        )
        instance_sensitivity = plan["available_sealed_test"][
            "instance_independence_sensitivity"
        ]
        group_sensitivity = plan["available_sealed_test"][
            "official_group_cluster_sensitivity"
        ]
        self.assertAlmostEqual(
            instance_sensitivity["minimum_detectable_relative_improvement"],
            0.1455049827724948,
        )
        self.assertAlmostEqual(
            group_sensitivity["minimum_detectable_relative_improvement"],
            0.14746436559267406,
        )
        self.assertAlmostEqual(
            group_sensitivity["zero_failure_upper_bound"],
            0.08433964335062538,
        )
        self.assertFalse(
            plan["available_sealed_test"][
                "adequate_for_10_percent_improvement_and_5_percent_failure_cap"
            ]
        )
        self.assertGreater(
            plan["recommended_future_design"]["minimum_independent_instances"],
            35,
        )
        self.assertEqual(
            plan["performance_precision"]["superiority_test"][
                "required_instances"
            ],
            78,
        )
        self.assertEqual(
            plan["performance_precision"]["noninferiority_test"][
                "required_instances"
            ],
            364,
        )
        self.assertEqual(
            plan["safety_precision"][
                "required_instances_for_five_percent_cap"
            ],
            59,
        )
        self.assertEqual(
            plan["recommended_future_design"]["minimum_independent_instances"],
            364,
        )
        self.assertEqual(
            plan["recommended_future_design"]
            ["additional_independent_group_keys_beyond_sealed_test"],
            330,
        )
        self.assertEqual(
            plan["recommended_future_design"]
            ["additional_group_ood_keys_beyond_current_unseen_test"],
            340,
        )
        self.assertEqual(
            plan["recommended_future_design"]["two_arm_main_comparison_runs"],
            2184,
        )
        self.assertEqual(
            plan["recommended_future_design"]
            ["maximum_acceptable_group_key_failure_probability"],
            0.05,
        )
        self.assertEqual(
            plan["recommended_future_design"]
            ["three_arm_including_shadow_runs"],
            3276,
        )
        self.assertEqual(
            plan["recommended_future_design"]
            ["superiority_and_safety_two_arm_runs"],
            468,
        )
        self.assertEqual(
            plan["recommended_future_design"]
            ["superiority_and_safety_three_arm_including_shadow_runs"],
            702,
        )


if __name__ == "__main__":
    unittest.main()
