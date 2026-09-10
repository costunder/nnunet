"""DEBUG trainer/payload boundary tests, not native-resampling certification.

The canonical resampling engine has separate parity tests. Here an explicitly
labelled analytic engine double isolates lazy selected-candidate loading, crop
padding, full-crop replacement, event RNG and zero-support feedback plumbing.
No medical image, final model, training checkpoint or installed trainer is used.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from test_online_no_placement_debug import production_classes
from custom_trainers.onlinecp_curriculum_policy import CurriculumError
from custom_trainers.onlinecp_feedback_metrics import compute_feedback_metrics


CONTRACT = "onlinecp_raw_target_paste_v1"


class RawCPTrainerDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = production_classes()
        cls.namespace["CurriculumError"] = CurriculumError
        cls.namespace["compute_feedback_metrics"] = compute_feedback_metrics
        source_root = Path(__file__).resolve().parents[1] / "custom_trainers"
        for filename, names in (("nnUNetTrainer_OnlinePairedCP.py", {"_bbox_around_paste"}),
                                ("nnUNetTrainer_OnlineCPFeedback.py", {"FeedbackLossObserver"})):
            tree = ast.parse((source_root / filename).read_text(encoding="utf-8"))
            nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
            module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), filename, "exec"), cls.namespace)
        tree = ast.parse((source_root / "nnUNetTrainer_OnlinePairedCP.py").read_text(encoding="utf-8"))
        trainer = copy.deepcopy(next(node for node in tree.body if isinstance(node, ast.ClassDef)
                                     and node.name == "_nnUNetTrainer_250epochs_OnlineCP"))
        trainer.body = [node for node in trainer.body if isinstance(node, ast.FunctionDef)
                        and node.name in {"__init__", "train_step", "_consume_native_transport_audit"}]
        cls.namespace["nnUNetTrainer"] = type("DEBUGOrdinaryTrainerBoundary", (), {
            "__init__": lambda self, *args, **kwargs: None,
            "train_step": lambda self, batch: batch,
        })
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), trainer], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "DEBUG_native_audit_trainer", "exec"), cls.namespace)
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def bank(self, root, *, duplicate_raw=False):
        root = Path(root)
        centers_raw = np.zeros((128, 3), dtype=np.int64)
        centers_raw[:, 0] = np.arange(128)
        if duplicate_raw:
            centers_raw[1] = centers_raw[0]
        entry = dict(
            paste_contract=np.asarray([CONTRACT]),
            case_id=np.asarray(["DEBUG_case"]),
            candidate_centers=np.full((128, 3), 4, dtype=np.int64),
            candidate_raw_centers=centers_raw, scores=np.linspace(0, 1, 128, dtype=np.float32),
            source_component=np.asarray([1]), source_diameter_mm=np.asarray([6.]),
            candidate_payloads=np.asarray([f"payloads/c{index}.json" for index in range(128)]),
            candidate_payload_sha256=np.asarray(["a" * 64] * 128),
            raw_case_reference=np.asarray(["cases/DEBUG_case.json"]),
            raw_case_reference_sha256=np.asarray(["b" * 64]),
        )
        np.savez(root / "entry.npz", **entry)
        metadata = dict(format="hiercp_online_bank_v2", paste_contract=CONTRACT,
                        entries_by_case={"DEBUG_case": ["entry.npz"]}, candidate_count=128,
                        hier_top_k=8, tumor_label=2, liver_label=1, cp_probability=.5,
                        intensity_scale_range=[.8, 1.2], intensity_shift_range_hu=[-10., 10.],
                        normalization={"mean": 50., "std": 20.})
        (root / "index.json").write_text(json.dumps(metadata), encoding="utf-8")
        return self.namespace["OnlineCPBank"](root / "index.json")

    def loader(self, bank, *, feedback=False, support=False):
        cls = self.namespace["nnUNetDataLoaderOnlineCPFeedback" if feedback else "nnUNetDataLoaderOnlineCP"]
        loader = cls.__new__(cls)
        loader.online_bank = bank
        loader.online_policy, loader.online_epoch = "hier_argmax", 4
        loader.patch_size, loader.need_to_pad = np.asarray([7, 7, 7]), np.asarray([0, 0, 0])
        draws, consumed = iter([.1, .8, .3, .4, .5]), []

        def draw():
            value = next(draws)
            consumed.append(value)
            return value

        loader._rng = lambda: SimpleNamespace(random=draw)
        loader.curriculum_sha256, loader.basic_control = "DEBUG_policy", False
        loader.feedback_state = SimpleNamespace(select=lambda *args, **kwargs: 127)
        loader._entry_ids, loader._candidate_indices, loader._choice_tokens = [], [], []
        case = {"metadata": {"preprocessed_shape": [9, 9, 9], "case_id": "DEBUG_case"},
                "DEBUG_baseline": np.ones((1, 9, 9, 9), dtype=np.int16)}
        case["baseline_seg"] = case["DEBUG_baseline"]
        candidate = {"output_bbox": [[2, 6], [2, 6], [2, 6]],
                     "case_id": "DEBUG_case", "source_component": 1,
                     "raw_target_center": [127, 0, 0], "case_reference_sha256": "b" * 64,
                     "source_mask": np.ones((2, 2, 2), dtype=bool),
                     "source_ct": np.full((1, 2, 2, 2), 50., dtype=np.float32),
                     "pasted_support": np.zeros((4, 4, 4), dtype=bool),
                     "seg_patch": np.ones((1, 4, 4, 4), dtype=np.int16),
                     "target_input_origin": np.asarray([2, 2, 2])}
        if support:
            candidate["pasted_support"][1, 1, 1] = True
            candidate["seg_patch"][0, 1, 1, 1] = 2
        loads = []
        bank._raw_store = SimpleNamespace(
            load_candidate=lambda path, sha: (loads.append(("candidate", path, sha)), candidate)[1],
            load_case=lambda path, sha: (loads.append(("case", path, sha)), case)[1],
        )
        calls = []

        def debug_engine(raw_case, selected, bbox, scale, shift_hu):
            calls.append((copy.deepcopy(bbox), scale, shift_hu))
            slices = tuple(slice(lo, hi) for lo, hi in bbox)
            segmentation = raw_case["DEBUG_baseline"].copy()
            if selected["pasted_support"].any():
                segmentation[0, 3, 3, 3] = 2
            region = segmentation[(slice(None), *slices)].copy()
            baseline = raw_case["DEBUG_baseline"][(slice(None), *slices)]
            return {"data": np.full(region.shape, 10 * scale + shift_hu, dtype=np.float32),
                    "seg": region, "pasted_support": (region[0] == 2) & (baseline[0] != 2),
                    "audit": {"format": "DEBUG_analytic_engine_boundary_not_native_reference"}}

        bank.raw_apply_function = lambda: debug_engine
        return loader, consumed, loads, calls, candidate

    def test_raw_candidates_remain_128_even_when_native_centers_collide(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_bank_") as root:
            bank = self.bank(root)
            entry = bank.load_for_case("DEBUG_case", 0)
            self.assertEqual(entry["candidate_centers"].shape, (128, 3))
            self.assertIsNone(bank._raw_store)
            self.assertNotIn("source_mask", entry)

    def test_duplicate_raw_candidates_are_still_rejected(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_duplicate_") as root:
            bank = self.bank(root, duplicate_raw=True)
            with self.assertRaisesRegex(self.namespace["OnlineCPError"], "Duplicate raw"):
                bank.load_for_case("DEBUG_case", 0)

    def test_base_and_feedback_keep_five_draws_and_load_only_selected_candidate(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_schedule_") as root:
            bank = self.bank(root)
            for feedback in (False, True):
                loader, consumed, loads, _, _ = self.loader(bank, feedback=feedback)
                plan, token = loader._sample_paste_plan("DEBUG_case")
                self.assertEqual(consumed, [.1, .8, .3, .4, .5])
                self.assertEqual(plan["candidate_index"], 127)
                self.assertEqual(plan["shift_hu"], 0.)
                self.assertAlmostEqual(plan["scale"], .96)
                self.assertNotIn("normalized_offset", plan)
                self.assertEqual([row[1] for row in loads], ["payloads/c127.json", "cases/DEBUG_case.json"])
                self.assertIsNotNone(token)

    def test_zero_support_uses_output_bbox_and_replaces_full_crop_without_skipping(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_zero_") as root:
            loader, _, _, calls, _ = self.loader(self.bank(root), feedback=True)
            plan, _ = loader._sample_paste_plan("DEBUG_case")
            source_shape, anchor = loader._paste_crop_geometry(plan, (9, 9, 9), "DEBUG_case")
            lower, upper = self.namespace["_bbox_around_paste"](loader, (9, 9, 9), plan["center"], source_shape, anchor)
            data = np.zeros((1, 7, 7, 7), dtype=np.float32)
            segmentation = np.ones((1, 7, 7, 7), dtype=np.int16)
            loader._apply_paste_to_crop(data, segmentation, lower, plan, "DEBUG_case")
            self.assertTrue(np.all(data != 0))
            self.assertEqual(int(loader._crop_pasted_mask.sum()), 0)
            self.assertEqual(loader._last_raw_paste_audit,
                             dict(raw_source_voxels=8, native_support_voxels=0, crop_support_voxels=0,
                                  native_zero_support=1, crop_zero_support=1))
            self.assertEqual(loader._entry_ids, ["entry.npz"])
            self.assertEqual(loader._candidate_indices, [127])
            self.assertEqual(calls[0][0], [[lo, hi] for lo, hi in zip(lower, upper)])
            self.assertEqual(len(calls), 1)

    def test_engine_replaces_only_valid_region_and_preserves_external_padding(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_padding_") as root:
            loader, _, _, calls, candidate = self.loader(self.bank(root))
            candidate["output_bbox"] = [[1, 3], [1, 3], [1, 3]]
            candidate["pasted_support"] = np.zeros((2, 2, 2), dtype=bool)
            candidate["seg_patch"] = np.ones((1, 2, 2, 2), dtype=np.int16)
            plan, _ = loader._sample_paste_plan("DEBUG_case")
            data = np.zeros((1, 7, 7, 7), dtype=np.float32)
            segmentation = np.full((1, 7, 7, 7), -1, dtype=np.int16)
            segmentation[:, 2:, 1:, 1:] = 1
            loader._apply_paste_to_crop(data, segmentation, (-2, -1, -1), plan, "DEBUG_case")
            self.assertEqual(calls[0][0], [[0, 5], [0, 6], [0, 6]])
            self.assertTrue(np.all(data[:, :2] == 0))
            self.assertTrue(np.all(segmentation[:, :2] == -1))
            self.assertTrue(np.all(data[:, 2:, 1:, 1:] != 0))

    def test_nonzero_support_is_engine_attribution_not_translated_raw_mask(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_support_") as root:
            loader, _, _, _, _ = self.loader(self.bank(root), feedback=True, support=True)
            plan, _ = loader._sample_paste_plan("DEBUG_case")
            data = np.zeros((1, 7, 7, 7), dtype=np.float32)
            segmentation = np.ones((1, 7, 7, 7), dtype=np.int16)
            loader._apply_paste_to_crop(data, segmentation, (1, 1, 1), plan, "DEBUG_case")
            self.assertEqual(int(loader._crop_pasted_mask.sum()), 1)
            self.assertTrue(loader._crop_pasted_mask[2, 2, 2])
            self.assertTrue(np.all(data != 0))

    def test_invalid_engine_attribution_fails_before_mutating_crop(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_invalid_") as root:
            loader, _, _, _, _ = self.loader(self.bank(root), feedback=True, support=True)
            original = loader.online_bank.raw_apply_function()

            def invalid(*args, **kwargs):
                result = original(*args, **kwargs)
                result["pasted_support"][:] = False
                return result

            loader.online_bank.raw_apply_function = lambda: invalid
            plan, _ = loader._sample_paste_plan("DEBUG_case")
            data = np.zeros((1, 7, 7, 7), dtype=np.float32)
            segmentation = np.ones((1, 7, 7, 7), dtype=np.int16)
            with self.assertRaisesRegex(self.namespace["OnlineCPError"], "attribution"):
                loader._apply_paste_to_crop(data, segmentation, (1, 1, 1), plan, "DEBUG_case")
            self.assertTrue(np.all(data == 0))
            self.assertTrue(np.all(segmentation == 1))

    def test_wrong_baseline_fails_before_engine_call_or_crop_write(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_baseline_") as root:
            loader, _, _, calls, _ = self.loader(self.bank(root))
            plan, _ = loader._sample_paste_plan("DEBUG_case")
            data = np.zeros((1, 7, 7, 7), dtype=np.float32)
            segmentation = np.ones((1, 7, 7, 7), dtype=np.int16)
            segmentation[0, 0, 0, 0] = 0
            before = segmentation.copy()
            with self.assertRaisesRegex(self.namespace["OnlineCPError"], "bound native baseline"):
                loader._apply_paste_to_crop(data, segmentation, (1, 1, 1), plan, "DEBUG_case")
            self.assertEqual(calls, [])
            np.testing.assert_array_equal(segmentation, before)
            self.assertTrue(np.all(data == 0))

    def test_oversized_raw_output_bbox_selects_fixed_patch_without_mutating_source(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_partial_geometry_") as root:
            loader, _, _, _, candidate = self.loader(self.bank(root))
            candidate["output_bbox"] = [[1, 8], [1, 8], [1, 8]]
            candidate["pasted_support"] = np.zeros((7, 7, 7), dtype=bool)
            candidate["seg_patch"] = np.ones((1, 7, 7, 7), dtype=np.int16)
            loader.patch_size = np.asarray([3, 5, 5])
            plan, _ = loader._sample_paste_plan("DEBUG_case")
            before = copy.deepcopy((plan["center"], candidate["source_mask"], candidate["output_bbox"]))
            lower, upper = loader._raw_candidate_crop_bbox(plan, (9, 9, 9), "DEBUG_case")
            np.testing.assert_array_equal(np.asarray(upper) - lower, [3, 5, 5])
            self.assertTrue(np.all(np.asarray(lower) >= 0))
            self.assertTrue(np.all(np.asarray(upper) <= 9))
            self.assertEqual(plan["center"], before[0])
            np.testing.assert_array_equal(candidate["source_mask"], before[1])
            self.assertEqual(candidate["output_bbox"], before[2])

    def test_applied_zero_support_is_unobserved_but_regular_loss_backward_runs(self):
        image = torch.linspace(-1, 1, 125).reshape(1, 1, 5, 5, 5)
        labels = torch.ones((1, 1, 5, 5, 5), dtype=torch.long)
        network = torch.nn.Conv3d(1, 3, 1)
        observer = self.namespace["FeedbackLossObserver"](lambda output, target: torch.nn.functional.cross_entropy(output, target[:, 0]))
        observer.context = dict(pasted_mask=torch.zeros_like(labels, dtype=torch.bool),
                                valid_mask=torch.ones_like(labels, dtype=torch.bool),
                                event_applied=torch.ones(1, dtype=torch.bool))
        output = network(image)
        loss = observer(output, labels)
        torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(output, labels[:, 0]))
        loss.backward()
        self.assertGreater(float(network.weight.grad.abs().sum()), 0.)
        self.assertFalse(bool(observer.observation["available"][0]))
        self.assertTrue(bool(torch.isnan(observer.observation["foreground_ce"][0])))

    def audit_trainer_and_batch(self):
        trainer = object.__new__(self.namespace["_nnUNetTrainer_250epochs_OnlineCP"])
        trainer._online_paste_contract = CONTRACT
        trainer._online_cp_events = trainer._online_cp_samples = 0
        trainer._online_schedule_hash = 0xCBF29CE484222325
        trainer._online_native_transport = None
        batch = {"data": torch.ones((3, 1, 1, 1, 1)),
                 "online_cp_applied": np.asarray([1, 1, 0], dtype=np.uint8),
                 "online_cp_schedule_token": np.asarray([1, 2, 3], dtype=np.uint64),
                 "online_cp_raw_source_voxels": np.asarray([8, 8, 0], dtype=np.int64),
                 "online_cp_native_support_voxels": np.asarray([0, 5, 0], dtype=np.int64),
                 "online_cp_crop_support_voxels": np.asarray([0, 5, 0], dtype=np.int64),
                 "online_cp_native_zero_support": np.asarray([1, 0, 0], dtype=np.int64),
                 "online_cp_crop_zero_support": np.asarray([1, 0, 0], dtype=np.int64)}
        return trainer, batch

    def test_native_transport_counts_are_separate_from_augmentation_and_event_counts(self):
        trainer, batch = self.audit_trainer_and_batch()
        result = trainer.train_step(batch)
        self.assertEqual(set(result), {"data"})
        self.assertEqual((trainer._online_cp_events, trainer._online_cp_samples), (2, 3))
        self.assertEqual(trainer._online_native_transport,
                         dict(raw_events=2, raw_source_voxels=16, native_support_voxels=5,
                              crop_support_voxels=5, native_zero_support_events=1, crop_zero_support_events=1))

    def test_native_transport_audit_rejects_missing_and_false_counts(self):
        for mutation in ("missing", "boolean", "zero_flag", "source_event"):
            with self.subTest(mutation=mutation):
                trainer, batch = self.audit_trainer_and_batch()
                if mutation == "missing":
                    del batch["online_cp_native_zero_support"]
                elif mutation == "boolean":
                    batch["online_cp_raw_source_voxels"] = np.asarray([True, True, False])
                elif mutation == "zero_flag":
                    batch["online_cp_native_zero_support"][0] = 0
                else:
                    batch["online_cp_raw_source_voxels"][1] = 0
                with self.assertRaises(self.namespace["OnlineCPError"]):
                    trainer.train_step(batch)
                self.assertEqual(trainer._online_cp_events, 0)

    def test_legacy_trainer_rejects_new_raw_contract_at_startup(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_trainer_gate_") as root:
            self.bank(root)
            cls = self.namespace["_nnUNetTrainer_250epochs_OnlineCP"]
            with mock.patch.dict(self.namespace["os"].environ, {"ONLINE_CP_BANK": str(Path(root) / "index.json")}):
                with self.assertRaisesRegex(self.namespace["OnlineCPError"], "Full/Basic OnlineCPFeedback"):
                    cls({}, "DEBUG", 0, {}, torch.device("cpu"))
                allowed = type("DEBUGVerifiedFeedbackContract", (cls,), {"required_paste_contract": CONTRACT})
                trainer = allowed({}, "DEBUG", 0, {}, torch.device("cpu"))
                self.assertEqual(trainer.num_epochs, 250)
                self.assertEqual(trainer._online_paste_contract, CONTRACT)


if __name__ == "__main__":
    unittest.main()
