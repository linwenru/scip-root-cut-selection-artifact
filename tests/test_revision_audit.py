from collections import Counter
import unittest

from scip_cut_trace_v2.revision_audit import build_revision_audit


class RevisionAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit, cls.cohort_rows = build_revision_audit()

    def test_root_row_difference_is_exactly_forced_rows(self):
        scope = self.audit["observational_scope"]

        self.assertTrue(scope["root_row_accounting_closes"])
        self.assertEqual(scope["source_root_rows"], 8_455_696)
        self.assertEqual(scope["optional_root_candidate_occurrences"], 8_454_995)
        self.assertEqual(scope["forced_root_rows"], 701)

    def test_actionability_is_separate_from_post_action_informativeness(self):
        scope = self.audit["observational_scope"]

        self.assertEqual(scope["online_actionable_decisions"], 318)
        self.assertEqual(scope["train_actionable_queries"], 215)
        self.assertEqual(scope["train_pairwise_informative_queries"], 204)
        self.assertEqual(
            scope["train_all_applied_queries_excluded_from_pairwise_fit"], 11
        )

    def test_parity_failure_and_exceptions_are_not_hidden(self):
        parity = self.audit["parity"]

        self.assertFalse(parity["predeclared_strict_all_pair_gate_passed"])
        self.assertTrue(parity["conditional_complete_callback_evidence_passed"])
        self.assertEqual(len(parity["exceptions"]), 4)

    def test_publication_cohort_overlap_is_explicit(self):
        overlap = self.audit["publication_cohort_overlap"]
        grouping = self.audit["publication_cohort_grouping"]

        self.assertEqual(overlap["active_instances"], 40)
        self.assertEqual(overlap["unique_instances_previously_evaluated"], 29)
        self.assertEqual(overlap["previously_unevaluated_instances"], 11)
        self.assertEqual(grouping["distinct_group_keys"], 40)
        self.assertEqual(grouping["distinct_nonempty_official_groups"], 30)
        self.assertEqual(grouping["officially_ungrouped_instances"], 10)
        self.assertEqual(
            Counter(row["sampling_stratum"] for row in self.cohort_rows),
            {"completion_enriched": 30, "hardness_coverage": 10},
        )
        self.assertEqual(len(self.cohort_rows), 40)

    def test_itt_reanalysis_keeps_every_planned_block(self):
        itt = self.audit["fixed_policy_intention_to_treat"]

        self.assertTrue(itt["all_predeclared_blocks_retained"])
        for action in itt["actions"].values():
            self.assertEqual(action["instances"], 40)
            self.assertEqual(action["pairs"], 120)
            self.assertFalse(action["gate_passed"])

    def test_learned_policy_has_no_complete_solve_efficacy_evidence(self):
        learned = self.audit["learned_policy_online_evidence"]

        self.assertFalse(learned["learned_action_in_publication_experiment"])
        self.assertEqual(learned["smoke_test_instances"], 1)
        self.assertEqual(learned["smoke_test_eligible_pairs"], 0)
        self.assertFalse(learned["complete_solve_effect_established"])

    def test_post_review_model_selection_does_not_use_external_validation(self):
        selection = self.audit["post_review_model_selection"]

        self.assertEqual(selection["boosting_rounds"], 42)
        self.assertFalse(selection["round_selection_external_validation_used"])
        self.assertEqual(
            selection["official_group_ood_selection_overlap"]["clusters"], 18
        )
        self.assertFalse(selection["stage_gate"]["passed"])

    def test_precision_plan_does_not_overstate_the_sealed_test(self):
        precision = self.audit["prospective_precision"]

        self.assertEqual(precision["available_test"]["instances"], 35)
        self.assertEqual(
            precision["available_test"]["distinct_group_keys"], 34
        )
        self.assertEqual(
            precision["available_test"]["group_keys_seen_in_training"], 10
        )
        self.assertEqual(
            precision["available_test"]["group_keys_unseen_in_training"], 24
        )
        self.assertEqual(
            precision["recommended_future_design"]["minimum_independent_instances"],
            364,
        )
        self.assertEqual(
            precision["recommended_future_design"]
            ["additional_independent_group_keys_beyond_sealed_test"],
            330,
        )

    def test_same_path_shadow_overhead_is_structurally_neutral(self):
        overhead = self.audit["same_path_shadow_overhead"]

        self.assertEqual(overhead["pairs"], 36)
        self.assertTrue(overhead["structural_gate_passed"])
        self.assertEqual(overhead["shadow_evaluation_pairs"], 35)
        self.assertEqual(
            overhead["full_path_exposure_sensitivity"]["solving_time"]["pairs"],
            35,
        )
        self.assertGreater(
            overhead["itt_par2"]["arm_wall_time_seconds"][
                "geometric_mean_ratio_shadow_over_native"
            ],
            1.0,
        )

    def test_fixed_policy_safety_and_time_scope_are_explicit(self):
        itt = self.audit["fixed_policy_intention_to_treat"]

        self.assertEqual(itt["primary_time_field"], "SCIP-reported solving_time")
        self.assertFalse(itt["full_arm_wall_time_sensitivity_available"])
        for action in itt["actions"].values():
            safety = action["conditional_policy_safety"]
            self.assertTrue(safety["includes_policy_fallbacks"])
            self.assertFalse(safety["causal_attribution_to_selection_change"])


if __name__ == "__main__":
    unittest.main()
