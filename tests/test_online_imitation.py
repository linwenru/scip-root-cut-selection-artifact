import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scip_cut_trace_v2.online_imitation import (
    EXCLUDED_NON_EQUIVALENT_FEATURES,
    ONLINE_ENCODED_FEATURES,
    OnlineImitationRanker,
    derive_online_dataset,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Row:
    def __init__(self, name, efficacy, origin=3):
        self.name = name
        self.efficacy = efficacy
        self.origin = origin

    def getVals(self):
        return [1.0, -2.0]

    def getNNonz(self):
        return 2

    def getRhs(self):
        return 1.0

    def getLhs(self):
        return -1e20

    def getConstant(self):
        return 0.0

    def getOrigintype(self):
        return self.origin

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


class _Model:
    def getParam(self, name):
        return {
            "cutselection/hybrid/efficacyweight": 1.0,
            "cutselection/hybrid/dircutoffdistweight": 0.0,
            "cutselection/hybrid/objparalweight": 0.1,
            "cutselection/hybrid/intsupportweight": 0.1,
        }[name]

    def getBestSol(self):
        return None

    def getCutEfficacy(self, row):
        return row.efficacy

    def getRowObjParallelism(self, row):
        return 0.0

    def getRowNumIntCols(self, row):
        return 1

    def getNLPRows(self):
        return 10

    def getNLPCols(self):
        return 5

    def getLPObjVal(self):
        return 3.0

    def getNLPIterations(self):
        return 7

    def getNNodeLPIterations(self):
        return 7

    def getDualbound(self):
        return 2.0

    def getPrimalbound(self):
        return 4.0

    def getGap(self):
        return 0.5

    def getNCutsApplied(self):
        return 0


class _Booster:
    def predict(self, matrix):
        del matrix
        return np.asarray([0.0, 1.0, 2.0], dtype=np.float32)


class OnlineImitationTest(unittest.TestCase):
    def test_derives_only_live_equivalent_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            features = ONLINE_ENCODED_FEATURES + EXCLUDED_NON_EQUIVALENT_FEATURES
            source_path = source_dir / "train.npz"
            np.savez_compressed(
                source_path,
                X=np.arange(2 * len(features), dtype=np.float32).reshape(2, -1),
                feature_names=np.asarray(features, dtype=str),
            )
            source_manifest_path = root / "source.json"
            source_manifest_path.write_text(
                json.dumps(
                    {
                        "checks": {"valid": True},
                        "feature_contract": {"encoded_feature_names": list(features)},
                        "matrices": {
                            "train": {
                                "path": str(source_path),
                                "sha256": _sha256(source_path),
                                "bytes": source_path.stat().st_size,
                                "features": len(features),
                                "rows": 2,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = derive_online_dataset(
                source_dir, source_manifest_path, output_dir, root / "online.json"
            )

            self.assertTrue(all(manifest["checks"].values()))
            with np.load(output_dir / "train.npz") as matrix:
                self.assertEqual(matrix["X"].shape, (2, 37))
                self.assertEqual(
                    tuple(matrix["feature_names"].astype(str)),
                    ONLINE_ENCODED_FEATURES,
                )

    def test_live_ranker_anchors_native_top_and_reranks_tail(self):
        rows = [_Row("a", 3.0), _Row("b", 2.0), _Row("c", 1.0)]
        ranker = OnlineImitationRanker(
            booster=_Booster(),
            feature_names=ONLINE_ENCODED_FEATURES,
            model_manifest_path=Path("model-manifest.json"),
            model_manifest_sha256="manifest",
            model_path=Path("model.ubj"),
            model_sha256="model",
            dataset_manifest_path=Path("dataset.json"),
            dataset_manifest_sha256="dataset",
            offline_stage_gate_passed=False,
        )

        selection = ranker.rank(_Model(), rows, rows, 2, run_number=1)

        self.assertEqual([row.name for row in selection.cuts], ["a", "c", "b"])
        self.assertEqual(selection.nselectedcuts, 2)
        self.assertEqual(selection.metadata["features"], 37)
        self.assertFalse(selection.metadata["offline_stage_gate_passed"])


if __name__ == "__main__":
    unittest.main()
