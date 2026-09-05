"""Debug-only I/O/regression unit fixtures; no medical data or model execution."""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping
import unittest

import numpy as np
from tools.online_eval_provenance import (
    EvaluationProvenanceError, build_evaluation_contract,
    contract_comparability, prepare_new_output,
)


def isolated_functions():
    """Load actual pure I/O functions without importing unavailable NIfTI/SciPy."""
    source = Path(__file__).resolve().parents[1] / "tools" / "online_eval_v2.py"
    wanted = {
        "EvaluationError", "load_json", "_json_safe", "atomic_text", "atomic_json",
        "atomic_csv", "file_sha256", "verify_prediction_inventory", "outer_split",
        "_regression_check",
    }
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in wanted]
    assert {node.name for node in nodes} == wanted
    prefix = ast.parse("from __future__ import annotations").body
    namespace = dict(globals())
    exec(compile(ast.Module(body=prefix + nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


class EvaluationIORegressionDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = isolated_functions()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="debug-eval-regression-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cohort = ["debug_case_1", "debug_case_2"]
        self.files = {}
        for role in ("ground_truth", "basic", "hier"):
            folder = self.root / role
            folder.mkdir()
            self.files[role] = {}
            for case_id in self.cohort:
                path = folder / f"{case_id}.nii.gz"
                # Byte-only hashing fixtures, deliberately NOT NIfTI/model output.
                path.write_bytes(f"debug hash fixture {role} {case_id}".encode())
                self.files[role][case_id] = path
        self.metrics = {
            side: {"mean_case_tumor_dice": 0.5, "gt_lesions": 2,
                   "detected": 1, "false_positive_lesions": 1}
            for side in ("basic", "hier")
        }
        self.reference = self.root / "old_summary.json"

    def contract(self, hier_trainer="DebugHier"):
        return build_evaluation_contract(
            cohort=self.cohort, ground_truth_files=self.files["ground_truth"],
            prediction_files={side: self.files[side] for side in ("basic", "hier")},
            trainers={"basic": "DebugBasic", "hier": hier_trainer},
            evaluation_definition={"profile": "debug_io_only"},
        )

    def write_reference(self, contract=None, metrics=None):
        payload = {"legacy_metrics": metrics if metrics is not None else self.metrics}
        if contract is not None:
            payload["evaluation_contract"] = contract
        self.reference.write_text(json.dumps(payload), encoding="utf-8")

    def check(self, allow=False):
        return self.api["_regression_check"](
            self.metrics, self.reference, 1e-9, allow,
            evaluation_contract=self.contract(),
        )

    def test_same_inputs_pass(self):
        self.write_reference(self.contract())
        self.assertEqual(self.check()["status"], "pass")

    def test_same_inputs_drift_cannot_be_bypassed(self):
        old = json.loads(json.dumps(self.metrics))
        old["hier"]["detected"] = 0
        self.write_reference(self.contract(), old)
        for allow in (False, True):
            with self.subTest(allow=allow), self.assertRaises(self.api["EvaluationError"]):
                self.check(allow)

    def test_missing_provenance_is_not_pass(self):
        self.write_reference()
        self.assertEqual(self.check()["status"], "unverified_provenance")

    def test_different_trainer_is_not_comparable(self):
        self.write_reference(self.contract("DebugLegacyTopK"))
        self.assertEqual(self.check()["status"], "not_comparable")

    def test_old_v2_shape_remains_explicitly_unverified(self):
        self.reference.write_text(json.dumps({
            "basic_cp": self.metrics["basic"], "hiercp": self.metrics["hier"],
        }), encoding="utf-8")
        self.assertEqual(self.check()["status"], "unverified_provenance")

    def test_bad_contract_is_not_downgraded_to_legacy(self):
        contract = self.contract()
        contract["methods"]["hier"]["trainer"] = "tampered"
        self.write_reference(contract)
        with self.assertRaises(EvaluationProvenanceError):
            self.check()

    def test_atomic_publish_does_not_clobber(self):
        target = self.root / "output.json"
        self.api["atomic_json"](target, {"complete": True})
        self.assertIs(json.loads(target.read_text())["complete"], True)
        before = target.read_bytes()
        with self.assertRaises(FileExistsError):
            self.api["atomic_json"](target, {"complete": False})
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_prediction_inventory_rejects_extra_case(self):
        folder = self.root / "hier"
        self.api["verify_prediction_inventory"](folder, self.cohort)
        (folder / "unexpected.nii.gz").write_bytes(b"debug extra")
        with self.assertRaises(self.api["EvaluationError"]):
            self.api["verify_prediction_inventory"](folder, self.cohort)

    def test_csv_streams_all_rows_and_refuses_replacement(self):
        target = self.root / "rows.csv"
        seen = []
        def rows():
            for value in range(7):  # Explicit debug fixture, not an evaluation subset.
                seen.append(value)
                yield {"value": value, "diagnostics": {"debug": True}}
        self.api["atomic_csv"](target, rows(), ["value"])
        self.assertEqual(seen, list(range(7)))
        with target.open(newline="", encoding="utf-8") as stream:
            self.assertEqual(list(csv.DictReader(stream)), [{"value": str(i)} for i in range(7)])
        before = target.read_bytes()
        with self.assertRaises(FileExistsError):
            self.api["atomic_csv"](target, [{"value": 999}], ["value"])
        self.assertEqual(target.read_bytes(), before)

    def test_evaluation_contract_wiring_static(self):
        # Guards against a helper existing but not being connected to evaluation.
        source = Path(__file__).resolve().parents[1] / "tools" / "online_eval_v2.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        evaluate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
        calls = [node for node in ast.walk(evaluate) if isinstance(node, ast.Call)]
        positions = {}
        for name in ("build_evaluation_contract", "prepare_new_output", "load_label", "verify_evaluation_contract"):
            positions[name] = min(node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == name)
        completion = next(node for node in calls
                          if isinstance(node.func, ast.Name) and node.func.id == "atomic_json"
                          and node.args and isinstance(node.args[0], ast.BinOp)
                          and isinstance(node.args[0].right, ast.Constant)
                          and node.args[0].right.value == "completion.json")
        self.assertLess(positions["build_evaluation_contract"], positions["load_label"])
        self.assertLess(positions["prepare_new_output"], positions["load_label"])
        self.assertGreater(positions["verify_evaluation_contract"], positions["load_label"])
        self.assertGreater(completion.lineno, positions["verify_evaluation_contract"])
        regression = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "_regression_check")
        self.assertIn("evaluation_contract", [item.arg for item in regression.keywords])

    def test_duplicate_validation_patient_rejected(self):
        target = self.root / "split.json"
        target.write_text(json.dumps({"splits": [{"train": ["train"], "val": ["val", "val"]}]}))
        with self.assertRaises(self.api["EvaluationError"]):
            self.api["outer_split"](target, 0)


if __name__ == "__main__":
    unittest.main()
