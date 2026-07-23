import unittest
from unittest.mock import Mock, patch

from pyscipopt import Model

from scip_cut_trace_v2._scip_pointer import model_pointer
from scip_cut_trace_v2.native_hybrid import (
    SUPPORTED_SCIP_VERSION,
    _validate_version,
    select_cuts_hybrid,
)


class NativeHybridBridgeTest(unittest.TestCase):
    def test_bridge_is_guarded_to_exact_runtime(self):
        self.assertEqual(SUPPORTED_SCIP_VERSION, (10, 0, 2))

    def test_model_pointer_and_supported_runtime(self):
        model = Model()
        try:
            self.assertGreater(model_pointer(model), 0)
            _validate_version(model)
        finally:
            model.freeProb()

    def test_duplicate_row_pointer_occurrences_are_preserved(self):
        duplicate = object()
        other = object()
        model = Mock()
        model.getMajorVersion.return_value = 10
        model.getMinorVersion.return_value = 0
        model.getTechVersion.return_value = 2
        with patch(
            "scip_cut_trace_v2.native_hybrid.row_pointer",
            side_effect=[11, 11, 22],
        ), patch(
            "scip_cut_trace_v2.native_hybrid.select_hybrid_pointers",
            return_value=([11, 22, 11], 2),
        ):
            selection = select_cuts_hybrid(
                model, [duplicate, duplicate, other], [], True, 3
            )

        self.assertEqual(selection.cuts, [duplicate, other, duplicate])
        self.assertEqual(selection.nselectedcuts, 2)


if __name__ == "__main__":
    unittest.main()
