import unittest

from scip_cut_trace_v2.causal_context import (
    capture_decision_context,
    context_sha256,
)


class FakeNode:
    def getNumber(self):
        return 1

    def getDepth(self):
        return 0


class FakeRow:
    def __init__(self, name, values, efficacy):
        self.name = name
        self.values = values
        self.efficacy = efficacy

    def getVals(self):
        return self.values

    def getRhs(self):
        return 3.0

    def getLhs(self):
        return float("-inf")

    def getConstant(self):
        return 0.0

    def getOrigintype(self):
        return 3

    def isLocal(self):
        return False

    def isModifiable(self):
        return False

    def isRemovable(self):
        return True

    def isIntegral(self):
        return False

    def isInGlobalCutpool(self):
        return False


class FakeModel:
    def getBestSol(self):
        return None

    def getCutEfficacy(self, row):
        return row.efficacy

    def getRowObjParallelism(self, row):
        return row.efficacy / 2.0

    def getRowNumIntCols(self, row):
        return 1

    def getCurrentNode(self):
        return FakeNode()

    def getLPSolstat(self):
        return "optimal"

    def getObjectiveSense(self):
        return "minimize"

    def getLPObjVal(self):
        return 12.0

    def getNLPRows(self):
        return 8

    def getNLPCols(self):
        return 6

    def getNLPIterations(self):
        return 21

    def getNNodeLPIterations(self):
        return 9

    def getNLPs(self):
        return 2

    def getNSepaRounds(self):
        return 1

    def getDualbound(self):
        return 10.0

    def getPrimalbound(self):
        return float("inf")

    def getGap(self):
        return float("inf")

    def getNNodes(self):
        return 1

    def getNTotalNodes(self):
        return 1

    def getNCutsApplied(self):
        return 0


class CausalContextTest(unittest.TestCase):
    def test_captures_pre_intervention_rows_and_stable_fingerprint(self):
        rows = [FakeRow("selected", [1.0, -2.0], 0.4), FakeRow("next", [3.0], 0.2)]

        context = capture_decision_context(
            FakeModel(), rows, [], 1, run_number=1, selector_call=2
        )

        self.assertEqual(context["solver_state"]["node_depth"], 0)
        self.assertEqual(context["solver_state"]["primal_bound"], "inf")
        self.assertNotIn("solving_time", context["solver_state"])
        self.assertTrue(context["candidates"][0]["native_selected"])
        self.assertFalse(context["candidates"][1]["native_selected"])
        self.assertEqual(context["candidates"][0]["coeff_norm_l1"], 3.0)
        self.assertEqual(context["candidates"][0]["coeff_std_abs"], 0.5)
        self.assertEqual(context_sha256(context), context_sha256(dict(context)))


if __name__ == "__main__":
    unittest.main()
