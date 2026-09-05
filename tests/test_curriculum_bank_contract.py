"""DEBUG-only provenance/partition tests; no medical training or evaluation."""
import copy
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from custom_trainers.onlinecp_curriculum_contract import (
    FORMAT, ARCHITECTURE, GEOMETRY, file_sha256, verify_curriculum_bank_contract,
    _compare_unpacked_cache,
)
from hiercp.contracts import validate_nested_cohorts, require_current_checkpoint
from hiercp.split import load_case_split, SPLIT_FORMAT
from tools.online_cp_benchmark import _training_only_planning_env


class CurriculumBankContractDebugTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="debug_curriculum_contract_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pre = self.root / "pre" / "Dataset900_Debug"
        self.pre.mkdir(parents=True)
        self.bank = self.root / "bank"
        self.bank.mkdir()
        self.train, self.val = ["a", "b"], ["c"]
        self.index = self.bank / "index.json"
        self.write(self.index, {"dataset_name": "Dataset900_Debug", "candidate_count": 128,
                                "entries_by_case": {"a": ["a.npz"]}})
        (self.bank / "a.npz").write_bytes(b"explicit debug hash-only artifact")
        split = self.pre / "splits_final.json"
        self.write(split, [{"train": self.train, "val": self.val}])
        gnn = self.bank / "gnn_split.json"
        self.write(gnn, {"train": ["a"], "val": ["b"], "outer_validation_excluded": self.val})
        marker = self.pre / "preprocess.json"
        self.data_root = self.pre / "DebugPlans_3d_fullres"
        self.data_root.mkdir()
        self.arrays = {}
        for offset, case_id in enumerate(self.train + self.val):
            data = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4) + offset
            seg = np.zeros((1, 2, 3, 4), dtype=np.int16)
            self.arrays[case_id] = data, seg
            np.savez_compressed(self.data_root / (case_id + ".npz"), data=data, seg=seg)
            (self.data_root / (case_id + ".pkl")).write_bytes(b"debug hash-only properties; never unpickled")
        self.marker = {"input_contract": {"planning_cohort": "outer_train_only_v1"},
                       "outputs": {"data_identifier": self.data_root.name, "storage_format": "npz",
                           "cases": [{"case_id": case_id,
                                      "data_sha256": file_sha256(self.data_root / (case_id + ".npz")),
                                      "properties_sha256": file_sha256(self.data_root / (case_id + ".pkl"))}
                                     for case_id in self.train + self.val]}}
        self.write(marker, self.marker)
        plans = self.pre / "DebugPlans.json"
        self.write(plans, {"configurations": {"3d_fullres": {"data_identifier": self.data_root.name}}})
        dataset = self.pre / "dataset.json"
        self.write(dataset, {"debug_only": True})
        names = {"index", "config", "manifest", "complete", "gnn_checkpoint", "gnn_split",
                 "prototype", "graph_complete", "outer_splits", "preprocessed_split",
                 "preprocess_marker", "plans", "dataset", "train_config", "nnunet_config"}
        paths = {"index": self.index, "preprocessed_split": split,
                 "gnn_split": gnn, "preprocess_marker": marker, "plans": plans, "dataset": dataset}
        for name in names - paths.keys():
            paths[name] = self.bank / f"debug_{name}.json"
            self.write(paths[name], {"debug_only": True})
        self.write(paths["nnunet_config"], {"dataset": {"configuration": "3d_fullres", "plans": "DebugPlans"}})
        self.paths = paths
        self.contract = {"format": FORMAT, "architecture_version": ARCHITECTURE,
                         "geometry_contract": GEOMETRY, "dataset_name": "Dataset900_Debug",
                         "nnunet_fold": 0, "curriculum_sha256": "debug-policy",
                         "candidate_count": 128, "train_case_ids": self.train,
                         "validation_case_ids": self.val, "gnn_train_case_ids": ["a"],
                         "gnn_validation_case_ids": ["b"], "prototype_training_case_ids": ["a"],
                         "entry_sha256": {"a.npz": file_sha256(self.bank / "a.npz")},
                         "files": {k: {"path": str(v), "sha256": file_sha256(v)} for k, v in paths.items()}}
        self.env = patch.dict(os.environ, {"nnUNet_preprocessed": str(self.pre.parent)})
        self.env.start()
        self.addCleanup(self.env.stop)

    @staticmethod
    def write(path, obj):
        path.write_text(json.dumps(obj), encoding="utf-8")

    def verify(self):
        self.write(self.bank / "curriculum_contract.json", self.contract)
        return verify_curriculum_bank_contract(self.index, curriculum_sha256="debug-policy",
                    expected_candidate_count=128, dataset_name="Dataset900_Debug", nnunet_fold=0)

    def refresh_marker(self):
        self.write(self.paths["preprocess_marker"], self.marker)
        self.contract["files"]["preprocess_marker"]["sha256"] = file_sha256(self.paths["preprocess_marker"])

    def test_exact_contract(self):
        verified = self.verify()
        self.assertEqual(verified["train_case_ids"], ["a", "b"])
        self.assertEqual(verified["verified_configuration"], "3d_fullres")

    def test_live_preprocessed_arrays_and_properties_cannot_change_after_publication(self):
        for suffix in (".npz", ".pkl"):
            path = self.data_root / ("a" + suffix)
            original = path.read_bytes()
            path.write_bytes(original + b"debug changed bytes")
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "bytes changed"):
                self.verify()
            path.write_bytes(original)

    def test_mutation_immediately_after_a_successful_case_hash_is_rejected(self):
        target = self.data_root / "a.pkl"
        original_hash = file_sha256
        changed = []
        def hash_then_change(path):
            result = original_hash(path)
            if Path(path) == target and not changed:
                target.write_bytes(target.read_bytes() + b"debug concurrent change")
                changed.append(True)
            return result
        with patch("custom_trainers.onlinecp_curriculum_contract.file_sha256", side_effect=hash_then_change):
            with self.assertRaisesRegex(ValueError, "changed during verification"):
                self.verify()
        self.assertEqual(changed, [True])

    def test_missing_extra_and_mixed_backend_are_rejected(self):
        canonical = self.data_root / "a.npz"
        moved = self.data_root / "a.debug-moved"
        canonical.rename(moved)
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.verify()
        moved.rename(canonical)
        for name in ("heldout.npz", "a.b2nd", "heldout.npy", "unrecognized.txt"):
            path = self.data_root / name
            path.write_bytes(b"debug-only extra file")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "inventory"):
                self.verify()
            path.unlink()

    def test_ordinary_automatic_unpacked_cache_is_allowed_without_identity_change(self):
        before = self.verify()
        for case_id, (data, seg) in self.arrays.items():
            np.save(self.data_root / (case_id + ".npy"), data)
            np.save(self.data_root / (case_id + "_seg.npy"), seg)
        self.assertEqual(before, self.verify())

    def test_unpacked_cache_value_shape_dtype_order_and_length_are_checked(self):
        data, seg = self.arrays["a"]
        path = self.data_root / "a.npy"
        variations = [data + 1, data.reshape(1, 3, 2, 4), data.astype(np.float64),
                      np.asfortranarray(data)]
        for value in variations:
            np.save(path, value)
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaisesRegex(ValueError, "cache"):
                self.verify()
        np.save(path, data)
        original = path.read_bytes()
        path.write_bytes(original[:-1])
        with self.assertRaisesRegex(ValueError, "length"):
            self.verify()
        path.write_bytes(original)
        np.save(self.data_root / "a_seg.npy", seg + 1)
        with self.assertRaisesRegex(ValueError, "cache bytes"):
            self.verify()

    def test_cache_nan_payload_and_signed_zero_require_exact_bytes(self):
        data, seg = self.arrays["a"]
        data = data.copy()
        data.ravel()[0] = -0.0
        data.ravel()[1] = np.nan
        np.savez_compressed(self.data_root / "a.npz", data=data, seg=seg)
        self.marker["outputs"]["cases"][0]["data_sha256"] = file_sha256(self.data_root / "a.npz")
        self.refresh_marker()
        np.save(self.data_root / "a.npy", data)
        self.verify()
        data.ravel()[0] = 0.0
        np.save(self.data_root / "a.npy", data)
        with self.assertRaisesRegex(ValueError, "cache bytes"):
            self.verify()

    def test_matching_fortran_npz_and_cache_are_supported(self):
        data, seg = (np.asfortranarray(array) for array in self.arrays["a"])
        np.savez_compressed(self.data_root / "a.npz", data=data, seg=seg)
        self.marker["outputs"]["cases"][0]["data_sha256"] = file_sha256(self.data_root / "a.npz")
        self.refresh_marker()
        np.save(self.data_root / "a.npy", data)
        np.save(self.data_root / "a_seg.npy", seg)
        self.verify()

    def test_object_cache_is_rejected_without_unpickling(self):
        np.save(self.data_root / "a.npy", np.array([{"debug": True}], dtype=object))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.verify()

    def test_marker_cannot_omit_case_or_repoint_configuration_data(self):
        original = copy.deepcopy(self.marker)
        self.marker["outputs"]["cases"].pop()
        self.refresh_marker()
        with self.assertRaisesRegex(ValueError, "cohort"):
            self.verify()
        self.marker = original
        self.marker["outputs"]["data_identifier"] = "other_configuration"
        self.refresh_marker()
        with self.assertRaisesRegex(ValueError, "data_identifier"):
            self.verify()

    def test_plans_must_be_the_live_dataset_plans(self):
        substitute = self.bank / "DebugPlans.json"
        substitute.write_bytes(self.paths["plans"].read_bytes())
        self.contract["files"]["plans"]["path"] = str(substitute)
        with self.assertRaisesRegex(ValueError, "Live nnU-Net plans"):
            self.verify()

    @unittest.skipUnless(importlib.util.find_spec("blosc2"), "Actual Blosc2 encoding requires optional installed blosc2")
    def test_actual_blosc2_inventory_and_data_segmentation_properties_mutation(self):
        import blosc2
        for case_id, (data, seg) in self.arrays.items():
            (self.data_root / (case_id + ".npz")).unlink()
            blosc2.asarray(data, urlpath=str(self.data_root / (case_id + ".b2nd")))
            blosc2.asarray(seg, urlpath=str(self.data_root / (case_id + "_seg.b2nd")))
        self.marker["outputs"]["storage_format"] = "blosc2"
        for row in self.marker["outputs"]["cases"]:
            row["data_sha256"] = file_sha256(self.data_root / (row["case_id"] + ".b2nd"))
            row["segmentation_sha256"] = file_sha256(self.data_root / (row["case_id"] + "_seg.b2nd"))
        self.refresh_marker()
        self.verify()
        for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
            path = self.data_root / ("a" + suffix)
            original = path.read_bytes()
            path.write_bytes(original + b"debug corruption")
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "bytes changed"):
                self.verify()
            path.write_bytes(original)
        (self.data_root / "a_seg.b2nd").unlink()
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.verify()

    def test_cache_comparison_uses_mmap_and_bounded_reads(self):
        class TrackedStream(io.BytesIO):
            largest_read = 0
            def read(self, count=-1):
                self.largest_read = max(self.largest_read, count)
                if count < 0:
                    raise AssertionError("A complete volume read was requested")
                return super().read(count)
        data = self.arrays["a"][0]
        path = self.data_root / "a.npy"
        np.save(path, data)
        stream = TrackedStream(data.tobytes())
        with patch("custom_trainers.onlinecp_curriculum_contract.STREAM_BYTES", 7), patch(
                "custom_trainers.onlinecp_curriculum_contract.np.load", wraps=np.load) as reader:
            _compare_unpacked_cache(stream, (data.shape, False, data.dtype, data.nbytes), path)
        self.assertLessEqual(stream.largest_read, 7)
        reader.assert_called_once_with(path, mmap_mode="r", allow_pickle=False)

    def test_old_architecture_rejected(self):
        self.contract["architecture_version"] = "legacy"
        with self.assertRaisesRegex(ValueError, "architecture"):
            self.verify()

    def test_outer_validation_in_gnn_rejected(self):
        self.contract["gnn_validation_case_ids"] = ["c"]
        with self.assertRaisesRegex(ValueError, "boundary"):
            self.verify()

    def test_prototype_fit_on_validation_rejected(self):
        self.contract["prototype_training_case_ids"] = ["b"]
        with self.assertRaisesRegex(ValueError, "boundary"):
            self.verify()

    def test_live_split_mutation_rejected(self):
        self.write(self.pre / "splits_final.json", [{"train": ["a", "c"], "val": ["b"]}])
        with self.assertRaisesRegex(ValueError, "changed"):
            self.verify()

    def test_entry_mutation_rejected(self):
        (self.bank / "a.npz").write_bytes(b"changed debug artifact")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.verify()

    def test_duplicate_ids_rejected(self):
        self.contract["train_case_ids"] = ["a", "a"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.verify()

    def test_nested_cohorts_and_legacy_checkpoint(self):
        validate_nested_cohorts(["a", "b"], ["c"], ["a"], ["b"], ["a"])
        with self.assertRaises(ValueError):
            validate_nested_cohorts(["a", "b"], ["b"], ["a"], ["b"], ["a"])
        with self.assertRaisesRegex(ValueError, "older/incompatible"):
            require_current_checkpoint({"model_kwargs": {}})

    def test_split_loader_rejects_outer_exclusion_overlap(self):
        path = self.root / "split.json"
        self.write(path, {"format": SPLIT_FORMAT, "train": ["a"], "val": ["b"],
                          "outer_validation_excluded": ["b"]})
        with self.assertRaisesRegex(ValueError, "Outer validation"):
            load_case_split(path)


class PlanningCohortDebugTests(unittest.TestCase):
    def test_planning_view_contains_no_validation_images_or_labels(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(prefix="debug_train_only_plan_") as tmp:
            root = Path(tmp)
            raw = root / "raw" / "Dataset900_LiverOnlineCP_OF1"
            (raw / "imagesTr").mkdir(parents=True)
            (raw / "labelsTr").mkdir()
            for case in ("train", "heldout"):
                (raw / "imagesTr" / f"{case}_0000.nii.gz").write_bytes(b"debug-not-nifti")
                (raw / "labelsTr" / f"{case}.nii.gz").write_bytes(b"debug-not-nifti")
            (raw / "dataset.json").write_text(json.dumps({"numTraining": 2,
                "file_ending": ".nii.gz", "channel_names": {"0": "CT"}}))
            outer = root / "outer.json"
            outer.write_text("{}")
            layout = SimpleNamespace(raw=root / "raw", outer_splits=outer,
                                     online_fold=lambda _: root / "fold")
            env = _training_only_planning_env(layout, 1, 900,
                {"train": ["train"], "val": ["heldout"]}, {"nnUNet_raw": str(raw.parent)})
            view = Path(env["nnUNet_raw"]) / raw.name
            self.assertEqual([p.name for p in (view / "labelsTr").iterdir()], ["train.nii.gz"])
            self.assertEqual([p.name for p in (view / "imagesTr").iterdir()], ["train_0000.nii.gz"])
            self.assertEqual(json.loads((view / "dataset.json").read_text())["numTraining"], 1)
            self.assertTrue((raw / "labelsTr" / "heldout.nii.gz").exists())


if __name__ == "__main__":
    unittest.main()
