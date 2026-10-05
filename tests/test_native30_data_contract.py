"""UNIT data-operator parity only; no CT/GPU quality or trained-weight claim."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from hiercp_v1x import native30_data_contract as contract


def metadata():
    identities = {case: {"patient_group": "UNIT_group_" + case} for case in ("train_a", "train_b", "val_c")}
    rows = [dict(id=case + ":" + str(target), case_id=case, patient_group=identities[case]["patient_group"],
                 component=1 if target else None, center=[target, 2, 3], target=target)
            for case in identities for target in (1, 0)]
    return {"donor_pool": [{"case_id": "train_a", "component_id": 1},
                           {"case_id": "train_b", "component_id": 2}],
            "identities": {"cases": identities}, "split": {"inner_train": ["train_a", "train_b"]},
            "records": rows, "UNIT_metadata_only": True}


def case_fixture():
    label = np.ones((8, 9, 10), dtype=np.uint8)
    label[2:4, 3:5, 4:6] = 2
    label[6, 7, 8] = 2
    return SimpleNamespace(label=label, image=np.arange(label.size, dtype=np.float32).reshape(label.shape),
                           spacing=np.array([1., 1.2, 1.4]))


class Native30DataContractUnit(unittest.TestCase):
    def test_assignment_exact_normal_module_parity(self):
        from l0_regions.donor_data import assignment
        value = metadata()
        before = copy.deepcopy(value)
        self.assertEqual(contract.assignment(value, 42), assignment(value, 42))
        self.assertEqual(value, before)
        self.assertEqual(contract.source_receipt()["policy"], "same_donor_live_v1")

    def test_source_and_spacing_exact_normal_module_parity(self):
        from hiercp_v22.data import sources, donor_in_target_spacing
        case = case_fixture()
        left, right = contract.sources(case, 1, 30), sources(case, 1, 30)
        self.assertEqual(left.entries, right.entries)
        for (actual, diameter), (expected, other_diameter) in zip(left, right):
            self.assertEqual(diameter, other_diameter)
            self.assertEqual(actual.anchor_center, expected.anchor_center)
            self.assertEqual(actual.patch_slices, expected.patch_slices)
            np.testing.assert_array_equal(actual.full_mask, expected.full_mask)
            np.testing.assert_array_equal(actual.patch_image, expected.patch_image)
            spacing = np.array([.8, 1., 1.1])
            converted, mask = contract.donor_in_target_spacing(actual, case.spacing, spacing)
            old_converted, old_mask = donor_in_target_spacing(expected, case.spacing, spacing)
            self.assertEqual(converted.anchor_center, old_converted.anchor_center)
            self.assertEqual(converted.patch_slices, old_converted.patch_slices)
            self.assertEqual(converted.voxel_count, old_converted.voxel_count)
            np.testing.assert_array_equal(mask, old_mask)
            np.testing.assert_array_equal(converted.patch_image, old_converted.patch_image)

    def test_current_graph_imports_are_not_needed_in_fresh_process(self):
        # The forbidden imports are unavailable, just as in a legacy worker.
        code = """import builtins, json, sys
from hiercp_v1x import native30_data_contract as c
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name in ('hiercp_v22.data','l0_regions.donor_data','hiercp.preparation_runtime'):
        raise ImportError('UNIT deliberately unavailable: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
value=json.loads(sys.argv[1])
rows=c.assignment(value,42)
r=c.source_receipt()
assert len(rows)==len(value['records'])
assert 'hiercp_v22.data' not in sys.modules
assert 'l0_regions.donor_data' not in sys.modules
assert not r['original_file_imports_executed']
print('exact AST unavailable-import UNIT PASS')
"""
        result = subprocess.run([sys.executable, "-B", "-c", code, json.dumps(metadata())],
                                cwd=contract._ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("UNIT PASS", result.stdout)

    def test_module_level_side_effects_are_not_executed_and_source_changes_rejected(self):
        with tempfile.TemporaryDirectory(prefix="native30_AST_UNIT_") as directory:
            root = Path(directory)
            for name in contract._DEFINITIONS:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(contract._ROOT / name, target)
                with target.open("a", encoding="utf8") as stream:
                    stream.write("\nraise RuntimeError('UNIT top-level must not execute')\n")
            self.assertEqual(contract.assignment(metadata(), 42, source_root=root), contract.assignment(metadata(), 42))
            receipt = contract.source_receipt(source_root=root)
            self.assertFalse(receipt["algorithm_rewritten"])
            self.assertEqual(set(receipt["exact_AST_definition_sha256"]["hiercp_v22/data.py"]),
                             {"SourceCollection", "sources", "donor_in_target_spacing"})
            with (root / "hiercp_v22/data.py").open("a", encoding="utf8") as stream:
                stream.write("# UNIT bytes changed after binding\n")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                contract.source_receipt(source_root=root)

    def test_missing_exact_definition_is_not_replaced_with_an_alternative(self):
        with tempfile.TemporaryDirectory(prefix="native30_missing_AST_UNIT_") as directory:
            path = Path(directory) / "data.py"
            path.write_text("def different_function(): return 1\n", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Exactly one original AST"):
                contract._definitions(path, ("sources",))


if __name__ == "__main__":
    unittest.main()
