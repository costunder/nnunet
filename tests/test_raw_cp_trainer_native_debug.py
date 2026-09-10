"""DEBUG persisted-bank -> real native CP -> real feedback transform integration.

All 128 tiny raw-distinct candidate payloads are genuinely prepared and hashed.
Only framework initialization, the in-memory DEBUG case reader and score-policy
state are isolated; neither resampling nor its outputs are mocked. Identity
augmentation deliberately isolates this boundary (spatial augmentation has its
own tests). No final-model configuration, medical data or checkpoint is used.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

from custom_trainers.onlinecp_curriculum_policy import CurriculumError
from custom_trainers.onlinecp_feedback_metrics import compute_feedback_metrics, transform_with_feedback
from custom_trainers.onlinecp_raw_bank import RawBankStore, audit_source_entry, save_case, save_candidate
from custom_trainers.onlinecp_raw_resampling import prepare_candidate
from test_online_no_placement_debug import production_classes
from test_raw_cp_resampling_debug import debug_native


CONTRACT = "onlinecp_raw_target_paste_v1"
CASE_ID = "DEBUG_native_case"


class RawCPTrainerNativeDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = production_classes()
        cls.namespace.update(CurriculumError=CurriculumError,
                             compute_feedback_metrics=compute_feedback_metrics,
                             transform_with_feedback=transform_with_feedback)
        source = Path(__file__).resolve().parents[1] / "custom_trainers"
        for filename, names in (("nnUNetTrainer_OnlinePairedCP.py", {"_bbox_around_paste"}),
                                ("nnUNetTrainer_OnlineCPFeedback.py", {"_FeedbackTransform", "FeedbackLossObserver"})):
            tree = ast.parse((source / filename).read_text(encoding="utf-8"))
            nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
            module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), filename, "exec"), cls.namespace)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def persisted_bank(self, root, *, elongated=False):
        # Keep this TestCase import local: unittest must not collect the entire
        # independent engine suite a second time from this integration module.
        from test_raw_cp_resampling_debug import RawCPResamplingDebugTests
        ct, seg, spacing, plans, case = RawCPResamplingDebugTests().fixture(shape=(31, 31, 31))
        case["metadata"]["case_id"] = CASE_ID
        if elongated:
            source = np.arange(525, dtype=np.float32).reshape(5, 5, 21) * 3 - 650
            mask = np.zeros(source.shape, dtype=bool)
            mask[2, 2, 1:20] = True
            zero_center, visible_center = (15, 15, 15), (15, 15, 19)
            positions = itertools.product((3, 7, 11, 15, 19, 23, 27), (3, 7, 11, 15, 19, 23, 27), (11, 15, 19))
        else:
            source = np.arange(125, dtype=np.float32).reshape(5, 5, 5) * 12 - 650
            mask = np.zeros(source.shape, dtype=bool)
            mask[2, 2, 2] = True
            # Two phases of the same raw source: zero and nonzero native support.
            zero_center, visible_center = (15, 15, 15), (15, 15, 3)
            positions = itertools.product((3, 7, 11, 15, 19, 23, 27), repeat=3)
        anchor = np.asarray(source.shape) // 2
        other = [center for center in positions if center not in {zero_center, visible_center}]
        centers = np.asarray([zero_center, *other[:126], visible_center], dtype=np.int64)
        self.assertEqual(len(np.unique(centers, axis=0)), 128)
        case_path = "raw_cases/DEBUG_native.json"
        case_sha = save_case(root, case_path, {name: value for name, value in case.items() if name != "preparation"})
        paths, hashes, native_centers, support_counts = [], [], [], []
        metadata = case["metadata"]
        for index, center in enumerate(centers):
            candidate = prepare_candidate(case, source, mask, anchor, center)
            candidate.update(case_id=CASE_ID, source_component=1,
                             raw_target_center=center.tolist(), case_reference_sha256=case_sha)
            path = f"raw_candidates/DEBUG_native/c{index:04d}.json"
            paths.append(path)
            hashes.append(save_candidate(root, path, candidate))
            support_counts.append(int(candidate["pasted_support"].sum()))
            mapped = center[::-1][metadata["transpose_forward"]] - np.asarray(metadata["crop_bbox"])[:, 0]
            native = np.rint((mapped + .5) * np.asarray(metadata["preprocessed_shape"])
                             / np.asarray(metadata["cropped_shape"]) - .5).astype(np.int64)
            native_centers.append(native)
        if not elongated:
            self.assertEqual(support_counts[0], 0)
        self.assertGreater(support_counts[127], 0)
        payload = dict(paste_contract=np.asarray([CONTRACT]), case_id=np.asarray([CASE_ID]),
                       candidate_centers=np.asarray(native_centers), candidate_raw_centers=centers,
                       scores=np.linspace(0, 1, 128, dtype=np.float32),
                       source_component=np.asarray([1], dtype=np.int64),
                       source_diameter_mm=np.asarray([(6 * np.prod(spacing) * int(mask.sum()) / np.pi) ** (1 / 3)], dtype=np.float32),
                       candidate_payloads=np.asarray(paths), candidate_payload_sha256=np.asarray(hashes),
                       raw_case_reference=np.asarray([case_path]), raw_case_reference_sha256=np.asarray([case_sha]))
        np.savez(root / "entry.npz", **payload)
        index = dict(format="hiercp_online_bank_v2", paste_contract=CONTRACT,
                     entries_by_case={CASE_ID: ["entry.npz"]}, candidate_count=128, hier_top_k=8,
                     cp_probability=.5, liver_label=1, tumor_label=2,
                     intensity_scale_range=[.8, 1.2], intensity_shift_range_hu=[-10., 10.],
                     normalization={"mean": 100., "std": 75.})
        (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
        verifier = RawBankStore(root)
        try:
            verified = audit_source_entry(root, payload, 128, store=verifier)
            self.assertEqual(verified["metadata"]["case_id"], CASE_ID)
            receipt = {"root": str(root.resolve()),
                       "index_sha256": hashlib.sha256((root / "index.json").read_bytes()).hexdigest(),
                       "witnesses": dict(verifier._witnesses)}
        finally:
            verifier.close()
        bank = self.namespace["OnlineCPBank"](root / "index.json")
        self.assertIsNone(bank._raw_store)
        bank.adopt_raw_verification(receipt)
        self.verification_receipt = receipt
        return bank, (ct, seg, spacing, plans), source, mask, centers, support_counts, anchor

    def test_startup_witness_handoff_avoids_rehash_but_preserves_stat_guards(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_raw_witness_") as directory:
            bank, *_ = self.persisted_bank(Path(directory))
            try:
                receipt = self.verification_receipt
                self.assertIsNot(bank._get_raw_store()._witnesses, receipt["witnesses"])
                with self.assertRaisesRegex(self.namespace["OnlineCPError"], "root/index"):
                    bank.adopt_raw_verification({**receipt, "root": str(Path(directory).parent)})
                entry = bank._load("entry.npz")
                with mock.patch("custom_trainers.onlinecp_raw_bank._sha",
                                side_effect=AssertionError("A startup-verified payload was rehashed in a worker")):
                    case, candidate = bank.load_raw_candidate(entry, 0)
                    self.assertEqual(case["metadata"]["case_id"], CASE_ID)
                    self.assertEqual(candidate["source_component"], 1)
                    # Only a DEBUG temporary manifest is changed; real payloads
                    # and the original Medical tree are never touched.
                    manifest = Path(directory) / str(entry["raw_case_reference"][0])
                    manifest.write_bytes(manifest.read_bytes() + b" ")
                    with self.assertRaises(ValueError):
                        bank.load_raw_candidate(entry, 0)
                index = Path(directory) / "index.json"
                index.write_bytes(index.read_bytes() + b" ")
                with self.assertRaisesRegex(self.namespace["OnlineCPError"], "root/index"):
                    bank.adopt_raw_verification(receipt)
            finally:
                bank._get_raw_store().close()

    def loader(self, bank, baseline_data, baseline_seg, *, basic, patch_size=None):
        loader = object.__new__(self.namespace["nnUNetDataLoaderOnlineCPFeedback"])
        loader.online_bank, loader.online_epoch = bank, 4
        loader.online_policy = "basic" if basic else "hier_argmax"
        loader.curriculum_sha256, loader.basic_control = "DEBUG_fixed_policy", basic
        loader.snapshot_sha256 = "DEBUG_frozen_snapshot"
        loader.batch_size = 1
        loader.patch_size = np.asarray(baseline_seg.shape[1:] if patch_size is None else patch_size, dtype=np.int64)
        loader.need_to_pad = np.zeros(3, dtype=np.int64)
        loader.patch_size_was_2d = False
        loader.get_indices = lambda: [CASE_ID]
        loader.get_do_oversample = lambda _: False
        loader._data = SimpleNamespace(load_case=lambda _: (baseline_data, baseline_seg, None, {"class_locations": {}}))
        loader.transforms = self.namespace["_FeedbackTransform"](ComposeTransforms([]), loader)
        draws, consumed = iter([.1, .8, .001, .4, .5]), []

        def draw():
            value = next(draws)
            consumed.append(value)
            return value

        loader._rng = lambda: SimpleNamespace(random=draw)

        def debug_selection(entry, scores, epoch, candidate_u, *, basic_control):
            # Explicit DEBUG score-state boundary, not a learned GNN prediction.
            self.assertEqual((entry, epoch, basic_control), ("entry.npz", 4, basic))
            return int(np.floor(candidate_u * len(scores))) if basic_control else int(np.argmax(scores))

        loader.feedback_state = SimpleNamespace(select=debug_selection)
        return loader, consumed

    def test_persisted_128_candidate_bank_native_oracle_and_feedback_batch(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_native_trainer_") as directory:
            bank, fixture, source, source_mask, centers, support_counts, anchor = self.persisted_bank(Path(directory))
            try:
                ct, seg, spacing, plans = fixture
                untouched_ct, untouched_seg = ct.copy(), seg.copy()
                baseline_data, baseline_seg, _ = debug_native(ct, seg, spacing, plans)
                baseline_data.flags.writeable = baseline_seg.flags.writeable = False
                store = bank._get_raw_store()
                loaded_candidates = []
                original_load = store.load_candidate

                def counted_load(path, digest):
                    loaded_candidates.append(path)
                    return original_load(path, digest)

                store.load_candidate = counted_load
                batches = []
                for basic, selected in ((True, 0), (False, 127)):
                    loader, consumed = self.loader(bank, baseline_data, baseline_seg, basic=basic)
                    batch = loader.generate_train_batch()
                    self.assertEqual(consumed, [.1, .8, .001, .4, .5])
                    self.assertEqual(batch["feedback_candidate_indices"], [selected])
                    self.assertEqual(batch["feedback_entry_ids"], ["entry.npz"])
                    np.testing.assert_array_equal(batch["online_cp_applied"], [1])
                    np.testing.assert_array_equal(batch["online_cp_raw_source_voxels"], [1])
                    np.testing.assert_array_equal(batch["online_cp_native_support_voxels"], [support_counts[selected]])
                    np.testing.assert_array_equal(batch["online_cp_crop_support_voxels"], [support_counts[selected]])
                    np.testing.assert_array_equal(batch["online_cp_native_zero_support"], [int(selected == 0)])
                    np.testing.assert_array_equal(batch["online_cp_crop_zero_support"], [int(selected == 0)])
                    pasted_ct, pasted_seg = ct.copy(), seg.copy()
                    origin = centers[selected] - anchor
                    slices = tuple(slice(int(start), int(start + length)) for start, length in zip(origin, source.shape))
                    pasted_ct[slices][source_mask] = (source * .96)[source_mask]
                    pasted_seg[slices][source_mask] = 2
                    expected_data, expected_seg, _ = debug_native(pasted_ct, pasted_seg, spacing, plans)
                    np.testing.assert_allclose(batch["data"][0].numpy(), expected_data, atol=3e-6, rtol=3e-6)
                    np.testing.assert_array_equal(batch["target"][0].numpy(), expected_seg)
                    support = (expected_seg == 2) & (baseline_seg != 2)
                    np.testing.assert_array_equal(batch["feedback_pasted_mask"][0].numpy(), support)
                    self.assertGreater(float(np.abs(expected_data - baseline_data).max()), 0.)
                    network = torch.nn.Conv3d(1, 3, 1)
                    observer = self.namespace["FeedbackLossObserver"](
                        lambda logits, labels: torch.nn.functional.cross_entropy(logits, labels[:, 0].long()))
                    observer.context = dict(pasted_mask=batch["feedback_pasted_mask"],
                                            valid_mask=batch["feedback_valid_mask"],
                                            event_applied=torch.as_tensor(batch["online_cp_applied"], dtype=torch.bool))
                    loss = observer(network(batch["data"]), batch["target"])
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertGreater(float(network.weight.grad.abs().sum()), 0.)
                    self.assertEqual(bool(observer.observation["available"][0]), selected == 127)
                    if selected == 0:
                        self.assertTrue(bool(torch.isnan(observer.observation["foreground_ce"][0])))
                    batches.append(batch)
                np.testing.assert_array_equal(batches[0]["online_cp_schedule_token"], batches[1]["online_cp_schedule_token"])
                self.assertNotEqual(int(batches[0]["online_cp_choice_token"][0]), int(batches[1]["online_cp_choice_token"][0]))
                self.assertEqual(loaded_candidates, ["raw_candidates/DEBUG_native/c0000.json",
                                                     "raw_candidates/DEBUG_native/c0127.json"])
                self.assertEqual(set(next(iter(bank._cache.values()))), {
                    "paste_contract", "case_id", "candidate_centers", "candidate_raw_centers", "scores",
                    "source_component", "source_diameter_mm", "candidate_payloads", "candidate_payload_sha256",
                    "raw_case_reference", "raw_case_reference_sha256",
                })
                # Drop only our DEBUG instrumentation before testing the real
                # spawn state. The case volume must not enter a worker pickle.
                del store.load_candidate

                def contains_array(value):
                    if isinstance(value, np.ndarray):
                        return True
                    if isinstance(value, dict):
                        return any(contains_array(item) for item in value.values())
                    if isinstance(value, (tuple, list)):
                        return any(contains_array(item) for item in value)
                    return False

                self.assertFalse(contains_array(store.__getstate__()))
                np.testing.assert_array_equal(ct, untouched_ct)
                np.testing.assert_array_equal(seg, untouched_seg)
            finally:
                if bank._raw_store is not None:
                    bank._raw_store.close()

    def test_elongated_raw_source_keeps_all_voxels_but_native_training_crop_is_partial(self):
        with tempfile.TemporaryDirectory(prefix="DEBUG_native_partial_") as directory:
            bank, fixture, source, mask, centers, support_counts, anchor = self.persisted_bank(Path(directory), elongated=True)
            try:
                ct, seg, spacing, plans = fixture
                baseline_data, baseline_seg, _ = debug_native(ct, seg, spacing, plans)
                loader, consumed = self.loader(bank, baseline_data, baseline_seg, basic=False, patch_size=(5, 11, 11))
                entry = bank.load_for_case(CASE_ID, 0)
                plan = loader._make_paste_plan(entry, 127, .96, 0., CASE_ID)
                complete_mask = plan["raw_candidate"]["source_mask"].copy()
                complete_bbox = np.asarray(plan["raw_candidate"]["output_bbox"]).copy()
                self.assertTrue(np.any(complete_bbox[:, 1] - complete_bbox[:, 0] > loader.patch_size))
                lower, upper = loader._raw_candidate_crop_bbox(plan, baseline_seg.shape[1:], CASE_ID)
                batch = loader.generate_train_batch()
                self.assertEqual(consumed, [.1, .8, .001, .4, .5])
                self.assertEqual(tuple(batch["data"].shape), (1, 1, 5, 11, 11))
                self.assertEqual(batch["feedback_candidate_indices"], [127])
                np.testing.assert_array_equal(batch["online_cp_applied"], [1])
                np.testing.assert_array_equal(batch["online_cp_raw_source_voxels"], [int(mask.sum())])
                pasted_ct, pasted_seg = ct.copy(), seg.copy()
                origin = centers[127] - anchor
                region = tuple(slice(int(start), int(start + length)) for start, length in zip(origin, source.shape))
                pasted_ct[region][mask] = (source * .96)[mask]
                pasted_seg[region][mask] = 2
                full_data, full_seg, _ = debug_native(pasted_ct, pasted_seg, spacing, plans)
                crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
                expected_data, expected_seg = full_data[(slice(None), *crop)], full_seg[(slice(None), *crop)]
                cropped_support = (expected_seg == 2) & (baseline_seg[(slice(None), *crop)] != 2)
                count = int(cropped_support.sum())
                self.assertGreater(count, 0)
                self.assertLess(count, support_counts[127])
                np.testing.assert_array_equal(batch["online_cp_native_support_voxels"], [support_counts[127]])
                np.testing.assert_array_equal(batch["online_cp_crop_support_voxels"], [count])
                np.testing.assert_allclose(batch["data"][0].numpy(), expected_data, atol=3e-6, rtol=3e-6)
                np.testing.assert_array_equal(batch["target"][0].numpy(), expected_seg)
                np.testing.assert_array_equal(batch["feedback_pasted_mask"][0].numpy(), cropped_support)
                np.testing.assert_array_equal(plan["raw_candidate"]["source_mask"], complete_mask)
                np.testing.assert_array_equal(plan["raw_candidate"]["output_bbox"], complete_bbox)
                self.assertEqual(entry["candidate_centers"].shape, (128, 3))
            finally:
                if bank._raw_store is not None:
                    bank._raw_store.close()


if __name__ == "__main__":
    unittest.main()
