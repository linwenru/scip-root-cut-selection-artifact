import unittest
from collections import Counter

from scip_cut_trace_v2.revision_overhead import (
    DEFAULT_PLAN,
    DEFAULT_RESULT,
    analyze_result,
    build_plan,
)


class RevisionOverheadPlanTest(unittest.TestCase):
    def test_plan_is_balanced_and_keeps_test_sealed(self):
        plan = build_plan()
        schedule = plan["experiment"]["execution_schedule"]

        self.assertTrue(all(plan["checks"].values()))
        self.assertEqual(len(plan["experiment"]["instances"]), 12)
        self.assertEqual(len(schedule), 36)
        self.assertEqual(
            Counter(record["arm_order"][0] for record in schedule),
            {"native": 18, "xgb-imitation-shadow": 18},
        )
        self.assertEqual(plan["data_boundary"]["test_split"], "sealed")

    def test_completed_result_passes_structural_gate(self):
        analysis = analyze_result(
            DEFAULT_RESULT,
            DEFAULT_PLAN,
            bootstrap_replicates=200,
        )

        self.assertEqual(analysis["pairs"], 36)
        self.assertEqual(analysis["outcome_pair_counts"], {"both_complete": 36})
        self.assertTrue(analysis["structural_gate_passed"])
        self.assertEqual(analysis["shadow_evaluation_pairs"], 35)
        self.assertEqual(analysis["full_path_structural_matching_pairs"], 35)
        self.assertEqual(analysis["proposed_selected_set_change_pairs"], 32)
        self.assertEqual(
            analysis["full_path_exposure_sensitivity"]["solving_time"]["pairs"],
            35,
        )
        self.assertEqual(
            analysis["analysis_contract"]["bootstrap_unit"], "12 MPS instances"
        )
        self.assertEqual(
            analysis["analysis_contract"]["bootstrap_replicates"], 200
        )


if __name__ == "__main__":
    unittest.main()
