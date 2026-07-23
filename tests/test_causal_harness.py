import copy
import inspect
import unittest
from unittest.mock import patch

from scip_cut_trace_v2.causal_harness import (
    LEARNED_SHADOW_ARMS,
    TREATMENT_ARMS,
    RunInterventionState,
    _build_root_treatment_cut_selector,
    _execution_arm_order,
    boundary_swap,
    classify_parity_pair,
    compare_causal_pair,
    compare_pair,
    efficacy_promote,
    evaluate_leave_one_seed_out,
    main,
    select_oracle_action,
)


def _result(arm):
    return {
        "arm": arm,
        "seed": 7,
        "instance_sha256": "abc",
        "runtime": {"python": "3.13", "pyscipopt": "6.2.1", "scip": "10.0.2"},
        "parameters": {"randomization/randomseedshift": 7},
        "outcome": {
            "status": "optimal",
            "objective_sense": "minimize",
            "primal_bound": 10.0,
            "dual_bound": 10.0,
            "gap": 0.0,
            "nodes": 12,
            "total_nodes": 12,
            "lp_iterations": 33,
            "lp_count": 8,
            "cuts_applied": 5,
            "solving_time": 1.0,
            "primal_dual_integral": 2.0,
        },
        "selector": {
            "mode": arm,
            "calls": 0 if arm == "native" else 3,
            "root_calls": 0 if arm == "native" else 2,
            "candidate_cuts": 0 if arm == "native" else 20,
            "forced_cuts": 0,
            "selected_cuts": 0,
        },
    }


class CausalHarnessParityTest(unittest.TestCase):
    def test_passes_structural_parity_and_ignores_time_overhead(self):
        native = _result("native")
        noop = _result("noop")
        noop["outcome"]["solving_time"] = 1.4
        noop["outcome"]["primal_dual_integral"] = 2.3

        comparison = compare_pair(native, noop)

        self.assertTrue(comparison["passed"])
        self.assertEqual(
            comparison["non_gating_measurements"][
                "solving_time_ratio_candidate_over_native"
            ],
            1.4,
        )

    def test_fails_when_callback_was_not_exercised(self):
        native = _result("native")
        noop = _result("noop")
        noop["selector"]["calls"] = 0

        comparison = compare_pair(native, noop)

        self.assertFalse(comparison["passed"])
        self.assertFalse(comparison["checks"]["candidate_callback_exercised"])

    def test_accepts_direct_hybrid_as_parity_candidate(self):
        native = _result("native")
        direct = _result("direct-hybrid")

        comparison = compare_pair(native, direct, "direct-hybrid")

        self.assertTrue(comparison["passed"])

    def test_fails_on_structural_trajectory_difference(self):
        native = _result("native")
        noop = copy.deepcopy(_result("noop"))
        noop["outcome"]["lp_iterations"] += 1

        comparison = compare_pair(native, noop)

        self.assertFalse(comparison["passed"])
        self.assertFalse(comparison["checks"]["same_lp_iterations"])

    def test_parity_classification_separates_ineligible_and_censored_pairs(self):
        native = _result("native")
        noop = _result("noop")
        noop["selector"]["calls"] = 0
        comparison = compare_pair(native, noop)
        self.assertEqual(
            classify_parity_pair(native, noop, comparison),
            "callback_not_exercised",
        )

        noop["selector"]["calls"] = 1
        native["outcome"]["status"] = "timelimit"
        noop["outcome"]["status"] = "timelimit"
        noop["outcome"]["nodes"] += 1
        comparison = compare_pair(native, noop)
        self.assertEqual(
            classify_parity_pair(native, noop, comparison),
            "both_arms_same_incomplete_limit_status",
        )

    def test_boundary_swap_preserves_count_and_changes_one_selected_cut(self):
        cuts = ["a", "b", "c", "d"]

        swapped, removed, added = boundary_swap(cuts, 2)

        self.assertEqual(swapped, ["a", "c", "b", "d"])
        self.assertEqual((removed, added), ("b", "c"))
        self.assertEqual(cuts, ["a", "b", "c", "d"])

    def test_boundary_swap_can_target_second_unselected_cut(self):
        swapped, removed, added = boundary_swap(
            ["a", "b", "c", "d"], 2, unselected_offset=1
        )

        self.assertEqual(swapped, ["a", "d", "c", "b"])
        self.assertEqual((removed, added), ("b", "d"))

    def test_efficacy_promote_uses_strict_improvement_and_stable_ties(self):
        promoted, removed, added = efficacy_promote(
            ["a", "b", "c", "d"], 2, [0.2, 0.2, 0.4, 0.4]
        )

        self.assertEqual(promoted, ["a", "c", "b", "d"])
        self.assertEqual((removed, added), ("b", "c"))
        self.assertIsNone(
            efficacy_promote(["a", "b", "c"], 2, [0.3, 0.2, 0.2])
        )

    def test_run_state_allows_at_most_one_intervention_per_run(self):
        state = RunInterventionState()
        state.mark_root_focused()
        self.assertTrue(state.can_intervene())
        self.assertTrue(state.can_intervene("first-run-only"))
        state.mark_intervened()
        self.assertFalse(state.can_intervene())
        state.mark_root_focused()
        self.assertTrue(state.can_intervene())
        self.assertFalse(state.can_intervene("first-run-only"))

    def test_execution_arm_order_is_deterministic_and_keeps_all_arms(self):
        arms = ("native", "random-rank", "efficacy-rank", "adaptive-score")

        first = _execution_arm_order(arms, "instance", 2, "frozen-key")
        second = _execution_arm_order(arms, "instance", 2, "frozen-key")

        self.assertEqual(first, second)
        self.assertEqual(set(first), set(arms))
        self.assertEqual(_execution_arm_order(arms, "instance", 2, None), arms)

    def test_root_selector_closes_over_context_capture_functions(self):
        selector, _ = _build_root_treatment_cut_selector(
            RunInterventionState(), "boundary-swap", "first-run-only"
        )

        closure = inspect.getclosurevars(selector._select)

        self.assertIn("capture_decision_context", closure.nonlocals)
        self.assertIn("context_sha256", closure.nonlocals)

    def test_learned_shadow_is_available_as_a_model_backed_neutral_arm(self):
        self.assertIn("xgb-imitation-shadow", LEARNED_SHADOW_ARMS)
        self.assertIn("xgb-imitation-shadow", TREATMENT_ARMS)

        native = _result("native")
        shadow = _result("xgb-imitation-shadow")
        shadow["selector"].update(
            {
                "run_count": 1,
                "decisions": 1,
                "interventions": 0,
                "intervention_records": [],
                "shadow_evaluations": 1,
                "shadow_records": [{}],
                "context_records": [{"context_sha256": "shadow"}],
            }
        )

        comparison = compare_causal_pair(native, shadow)

        self.assertTrue(comparison["neutral_shadow"])
        self.assertTrue(comparison["safe"])
        self.assertTrue(comparison["policy_evaluable"])
        self.assertTrue(comparison["valid"])

        shadow["outcome"]["lp_iterations"] += 1
        mismatch = compare_causal_pair(native, shadow)
        self.assertFalse(mismatch["safe"])
        self.assertFalse(
            mismatch["safety_checks"]["shadow_same_lp_iterations"]
        )

    def test_causal_comparison_checks_run_budget(self):
        native = _result("native")
        treatment = _result("boundary-swap")
        treatment["selector"].update(
            {
                "run_count": 2,
                "interventions": 2,
                "intervention_records": [{"run": 1}, {"run": 2}],
            }
        )

        comparison = compare_causal_pair(native, treatment)

        self.assertTrue(comparison["safe"])
        self.assertTrue(comparison["eligible"])
        self.assertTrue(comparison["valid"])

    def test_causal_comparison_accepts_predeclared_treatment_arm(self):
        native = _result("native")
        treatment = _result("efficacy-promote")
        treatment["selector"].update(
            {
                "run_count": 1,
                "interventions": 1,
                "intervention_records": [{"run": 1}],
            }
        )

        self.assertTrue(compare_causal_pair(native, treatment)["valid"])

    def test_fixed_baseline_same_set_is_an_evaluable_policy_fallback(self):
        native = _result("native")
        treatment = _result("efficacy-rank")
        treatment["selector"].update(
            {
                "run_count": 1,
                "decisions": 1,
                "interventions": 0,
                "intervention_records": [],
                "context_records": [{"context_sha256": "same-set"}],
            }
        )

        comparison = compare_causal_pair(native, treatment)

        self.assertTrue(comparison["safe"])
        self.assertFalse(comparison["eligible"])
        self.assertTrue(comparison["policy_fallback"])
        self.assertTrue(comparison["policy_evaluable"])
        self.assertTrue(comparison["valid"])

    def test_fixed_baseline_without_eligible_callback_is_still_policy_evaluable(self):
        native = _result("native")
        treatment = _result("adaptive-score")
        treatment["selector"].update(
            {
                "run_count": 1,
                "decisions": 0,
                "interventions": 0,
                "intervention_records": [],
                "context_records": [],
            }
        )

        comparison = compare_causal_pair(native, treatment)

        self.assertTrue(comparison["policy_fallback"])
        self.assertTrue(comparison["valid"])

    def test_oracle_uses_only_valid_actions_and_native_wins_ties(self):
        native = _result("native")
        boundary = _result("boundary-swap")
        boundary["selector"].update(
            {"run_count": 1, "interventions": 1, "intervention_records": [{}]}
        )
        boundary_comparison = compare_causal_pair(native, boundary)
        efficacy = _result("efficacy-promote")
        efficacy["selector"].update(
            {"run_count": 1, "interventions": 1, "intervention_records": [{}]}
        )
        efficacy["outcome"]["lp_iterations"] = 20
        efficacy_comparison = compare_causal_pair(native, efficacy)

        oracle = select_oracle_action(
            {
                "boundary-swap": boundary_comparison,
                "efficacy-promote": efficacy_comparison,
            }
        )

        self.assertEqual(oracle["selected_action"], "efficacy-promote")
        self.assertEqual(oracle["delta_selected_minus_native"], -13.0)
        self.assertAlmostEqual(oracle["relative_saving"], 13.0 / 33.0)

        efficacy_comparison["valid"] = False
        tied = select_oracle_action(
            {
                "boundary-swap": boundary_comparison,
                "efficacy-promote": efficacy_comparison,
            }
        )
        self.assertEqual(tied["selected_action"], "native")

    def test_oracle_cli_passes_reuse_existing_to_runner(self):
        manifest = {
            "all_actions_safe": True,
            "oracle_summary": {},
        }
        with patch(
            "scip_cut_trace_v2.causal_harness.run_action_oracle_suite",
            return_value=manifest,
        ) as runner, patch("builtins.print"):
            return_code = main(
                [
                    "action-oracle-suite",
                    "instance.mps",
                    "--reuse-existing",
                    "--intervention-scope",
                    "first-run-only",
                ]
            )

        self.assertEqual(return_code, 0)
        self.assertTrue(runner.call_args.args[-6])
        self.assertEqual(runner.call_args.args[-5], "first-run-only")
        self.assertIsNone(runner.call_args.args[-4])
        self.assertEqual(runner.call_args.args[-3], 1)
        self.assertIsNone(runner.call_args.args[-2])
        self.assertIsNone(runner.call_args.args[-1])

    def test_learned_ranker_same_set_is_an_evaluable_policy_fallback(self):
        native = _result("native")
        treatment = _result("xgb-imitation-rank")
        treatment["selector"].update(
            {
                "run_count": 1,
                "decisions": 1,
                "interventions": 0,
                "intervention_records": [],
                "context_records": [{"context_sha256": "same-set"}],
            }
        )

        comparison = compare_causal_pair(native, treatment)

        self.assertTrue(comparison["policy_fallback"])
        self.assertTrue(comparison["valid"])

    def test_leave_one_seed_out_does_not_use_held_out_outcome_for_selection(self):
        pairs = []
        for seed, treatment_lp_iterations in enumerate((20, 20, 40)):
            native = _result("native")
            native["seed"] = seed
            treatment = _result("boundary-swap")
            treatment["seed"] = seed
            treatment["outcome"]["lp_iterations"] = treatment_lp_iterations
            treatment["selector"].update(
                {"run_count": 1, "interventions": 1, "intervention_records": [{}]}
            )
            pairs.append(
                {
                    "seed": seed,
                    "native_outcome": native["outcome"],
                    "actions": {
                        "boundary-swap": {
                            "comparison": compare_causal_pair(native, treatment)
                        }
                    },
                }
            )

        stability = evaluate_leave_one_seed_out(pairs, ["boundary-swap"])

        self.assertTrue(stability["safe"])
        self.assertEqual(stability["wins"], 2)
        self.assertEqual(stability["losses"], 1)
        self.assertEqual(
            stability["evaluations"][2]["selected_action"], "boundary-swap"
        )


if __name__ == "__main__":
    unittest.main()
