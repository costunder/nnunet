"""DEBUG original-retention/source-schedule integration, not medical training.

Uses the real exhaustive CP predicate, real 128-entry NPZ files and production
bank/sampler/paste code. Framework initialization and GNN scoring are not run.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from scipy import ndimage as ndi
from threadpoolctl import threadpool_limits

from hiercp.common import build_candidate_pool
from tools import online_cp_benchmark as online


def production_classes():
    """Load actual source definitions without changing an installed nnU-Net."""
    root = Path(__file__).resolve().parents[1] / "custom_trainers"
    namespace = dict(np=np, torch=torch, hashlib=hashlib, json=json, os=os,
                     Path=Path, OrderedDict=OrderedDict, nnUNetDataLoader=object,
                     BANK_FORMAT=online.BANK_FORMAT, TRAINER_FORMAT="hiercp_online_trainer_v2",
                     threadpool_limits=threadpool_limits)
    requested = {"OnlineCPError", "OnlineCPBank", "_stable_u64", "_stable_seed",
                 "_anchored_slices", "_select_candidate_index", "nnUNetDataLoaderOnlineCP"}
    tree = ast.parse((root / "nnUNetTrainer_OnlinePairedCP.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "crop_and_pad_nd":
                    namespace["crop_and_pad_nd"] = getattr(importlib.import_module(node.module), alias.name)
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in requested]
    if {node.name for node in nodes} != requested:
        raise AssertionError("A production definition disappeared")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(root), "exec"), namespace)
    from custom_trainers.onlinecp_curriculum_policy import schedule_token
    namespace.update(schedule_token=schedule_token)
    tree = ast.parse((root / "nnUNetTrainer_OnlineCPFeedback.py").read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "nnUNetDataLoaderOnlineCPFeedback")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(root), "exec"), namespace)
    return namespace


class OnlineNoPlacementDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classes = production_classes()
        shape = (9, 9, 9)
        label = np.ones(shape, dtype=np.int16)
        label[2:7, 2:7, 2:7] = 2
        raw = SimpleNamespace(image=np.zeros(shape, dtype=np.float32), label=label,
                              shape=shape, spacing=np.ones(3, dtype=np.float32))
        source = online._source_from_component(raw, (label == 2).astype(np.int16), 1, 2)
        cls.proof = {}
        pool, _ = build_candidate_pool(
            raw, source, placement_mask=label == 1, full_organ_mask=label > 0,
            occupied_mask=label == 2, organ_distance=np.ones(shape, dtype=np.float32),
            rng=np.random.default_rng(42), num_candidates=128, max_draws=4000,
            min_liver_coverage=.85, occupied_clearance_vox=2,
            min_center_separation_mm=0.0, min_center_separation_vox=12.0,
            diagnostics=cls.proof,
        )
        if pool or cls.proof.get("fullsearch_exhausted") is not True:
            raise AssertionError("DEBUG impossible-placement fixture did not prove zero")

    def no_cp_row(self, case="DEBUG_MIX", component=1):
        proof = dict(self.proof, source_component=component)
        return dict(case_id=case, source_component=component, status="no_placement",
                    reason="DEBUG actual exhaustive zero", candidate_count=0,
                    rejected_geometry=0, rejected_preprocessed=0, entry="",
                    candidate_search_json=json.dumps(proof))

    def bank_fixture(self, root):
        entries = root / "entries"
        entries.mkdir()
        names, rows = {}, [self.no_cp_row("DEBUG_ZERO"), self.no_cp_row()]
        inventory = {"DEBUG_ZERO": [1], "DEBUG_MIX": [1, 2], "DEBUG_NEXT": [1]}
        centers = np.column_stack((np.arange(128) % 8 + 2, np.arange(128) // 8 + 2, np.full(128, 2))).astype(np.int32)
        scores = np.linspace(0, 1, 128, dtype=np.float32)
        for case, component in (("DEBUG_MIX", 2), ("DEBUG_NEXT", 1)):
            relative = f"entries/{case}__component_{component:03d}.npz"
            np.savez(root / relative, source_data=np.ones((1, 1, 1, 1), np.float32),
                     source_mask=np.ones((1, 1, 1), np.uint8), anchor_offset=np.zeros(3, np.int16),
                     candidate_centers=centers, candidate_raw_centers=centers, scores=scores,
                     source_component=np.asarray([component], np.int16), source_diameter_mm=np.asarray([1.], np.float32))
            names[case] = [relative]
            rows.append(dict(case_id=case, source_component=component, status="ok",
                             candidate_count=128, entry=relative,
                             entry_sha256=online.file_sha256(root / relative),
                             candidate_pool_sha256=online.candidate_pool_hash(centers, centers, scores)))
        metadata = dict(format=online.BANK_FORMAT, candidate_count=128, hier_top_k=8,
                        cp_probability=.5, tumor_label=2, liver_label=1,
                        intensity_scale_range=[.95, 1.05], intensity_shift_range_hu=[-5., 5.],
                        normalization={"mean": 0., "std": 1.}, no_placement_policy="retain_original",
                        entries_by_case=names, eligible_sources_by_case=inventory,
                        no_eligible_cases=[], eligible_cases=3, source_entries=2, total_candidates=256,
                        **online._source_slot_metadata(rows, inventory, 128))
        path = root / "index.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return self.classes["OnlineCPBank"](path), metadata, rows, inventory

    def sampler(self, bank, draws, *, feedback=False, basic=False):
        name = "nnUNetDataLoaderOnlineCPFeedback" if feedback else "nnUNetDataLoaderOnlineCP"
        loader = object.__new__(self.classes[name])
        loader.online_bank, loader.online_epoch = bank, 7
        loader.online_policy = "basic" if basic else "hier_argmax"
        remaining, consumed, selected = list(draws), [], []
        def random():
            value = remaining.pop(0)
            consumed.append(value)
            return value
        loader._rng = lambda: SimpleNamespace(random=random)
        if feedback:
            loader.curriculum_sha256, loader.basic_control = "DEBUG_policy", basic
            loader._entry_ids, loader._candidate_indices, loader._choice_tokens = [], [], []
            def choose(entry, scores, epoch, u, *, basic_control):
                selected.append(entry)
                return int(np.floor(u * len(scores))) if basic_control else int(np.argmax(scores))
            loader.feedback_state = SimpleNamespace(select=choose)
        return loader, consumed, selected

    def test_real_zero_proof_is_not_geometry_or_partial_pool_fallback(self):
        row = self.no_cp_row()
        online._validate_no_placement_row(row, 128)
        for change in ({"fullsearch_exhausted": False}, {"accepted": 1},
                       {"excluded_center_count": 1}, {"source_component": 99}):
            altered = dict(row, candidate_search_json=json.dumps({**self.proof, **change}))
            with self.assertRaises(online.OnlineBenchmarkError):
                online._validate_no_placement_row(altered, 128)
        for change in ({"status": "source_patch_too_large"}, {"status": "insufficient_candidates", "candidate_count": 127},
                       {"rejected_geometry": 1}, {"rejected_preprocessed": 1}, {"entry": "fake.npz"}):
            with self.assertRaises(online.OnlineBenchmarkError):
                online._validate_no_placement_row({**row, **change}, 128)

    def test_zero_case_and_source_do_not_prevent_next_real_bank_entry(self):
        with tempfile.TemporaryDirectory(prefix="debug_online_no_cp_") as directory:
            root = Path(directory)
            bank, metadata, rows, inventory = self.bank_fixture(root)
            online._audit_completed_bank(root, metadata, rows, inventory, 128)
            self.assertEqual(metadata["eligible_source_slots"], 4)
            self.assertEqual(metadata["no_placement_sources"], 2)
            self.assertEqual(len(bank.source_slots_by_case["DEBUG_MIX"]), 2)
            self.assertEqual(bank.load_for_case("DEBUG_NEXT", 0)["scores"].shape, (128,))
            altered = copy.deepcopy(metadata)
            altered["source_slots_by_case"]["DEBUG_MIX"].pop(0)
            with self.assertRaises(online.OnlineBenchmarkError):
                online._audit_completed_bank(root, altered, rows, inventory, 128)
            (root / "index.json").write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(self.classes["OnlineCPError"]):
                self.classes["OnlineCPBank"](root / "index.json")

    def test_paired_and_feedback_keep_failed_source_probability_and_five_draws(self):
        with tempfile.TemporaryDirectory(prefix="debug_online_no_cp_") as directory:
            bank, _, _, _ = self.bank_fixture(Path(directory))
            for feedback in (False, True):
                for source_u in (.1, .49, .5, .9):
                    draws = [.1, source_u, .25, .4, .6]
                    full, consumed, selected = self.sampler(bank, draws, feedback=feedback)
                    basic, basic_consumed, _ = self.sampler(bank, draws, feedback=feedback, basic=True)
                    full_plan, full_event = full._sample_paste_plan("DEBUG_MIX")
                    basic_plan, basic_event = basic._sample_paste_plan("DEBUG_MIX")
                    self.assertEqual(consumed, draws)
                    self.assertEqual(basic_consumed, draws)
                    self.assertEqual(full_event, basic_event)
                    self.assertEqual(full_plan is None, source_u < .5)
                    self.assertEqual(basic_plan is None, source_u < .5)
                    if source_u < .5:
                        self.assertEqual(selected, [])
                        if feedback:
                            self.assertEqual(full._entry_ids, [""])
                            self.assertEqual(full._candidate_indices, [-1])
                    else:
                        self.assertEqual(full_plan["scale"], basic_plan["scale"])
                        self.assertEqual(full_plan["normalized_offset"], basic_plan["normalized_offset"])
                        self.assertEqual(int(full_plan["entry"]["source_component"][0]), 2)

    def test_non_cp_then_next_case_keeps_rng_stream_and_original_arrays(self):
        with tempfile.TemporaryDirectory(prefix="debug_online_no_cp_") as directory:
            bank, _, _, _ = self.bank_fixture(Path(directory))
            draws = [.1, .8, .25, .4, .6, .1, .8, .25, .4, .6]
            full, consumed, _ = self.sampler(bank, draws)
            basic, basic_consumed, _ = self.sampler(bank, draws, basic=True)
            original_data = np.arange(512, dtype=np.float32).reshape(1, 8, 8, 8)
            original_seg = np.ones((1, 8, 8, 8), dtype=np.int16)
            for loader in (full, basic):
                data, seg = original_data.copy(), original_seg.copy()
                plan, _ = loader._sample_paste_plan("DEBUG_ZERO")
                self.assertIsNone(plan)
                np.testing.assert_array_equal(data, original_data)
                np.testing.assert_array_equal(seg, original_seg)
                plan, _ = loader._sample_paste_plan("DEBUG_NEXT")
                self.assertIsNotNone(plan)
                self.assertEqual(int(plan["entry"]["source_component"][0]), 1)
            self.assertEqual(consumed, draws)
            self.assertEqual(basic_consumed, draws)

    def test_unknown_and_nonzero_failure_status_cannot_be_a_no_cp_slot(self):
        with tempfile.TemporaryDirectory(prefix="debug_online_no_cp_") as directory:
            root = Path(directory)
            _, metadata, _, _ = self.bank_fixture(root)
            for status in ("source_patch_too_large", "error", "unrepresentable_source", "insufficient_candidates"):
                altered = copy.deepcopy(metadata)
                altered["source_slots_by_case"]["DEBUG_ZERO"][0]["status"] = status
                (root / "index.json").write_text(json.dumps(altered), encoding="utf-8")
                with self.assertRaises(self.classes["OnlineCPError"]):
                    self.classes["OnlineCPBank"](root / "index.json")

    def test_actual_batch_keeps_non_cp_patient_and_pastes_following_patient(self):
        """Actual loader crop/paste/tensor assembly; only crop selection is fixed."""
        with tempfile.TemporaryDirectory(prefix="debug_online_no_cp_") as directory:
            bank, _, _, _ = self.bank_fixture(Path(directory))
            draws = [.1, .8, .25, .4, .6] * 2
            initial = np.full((1, 32, 32, 32), 100., np.float32)
            labels = np.ones_like(initial, dtype=np.int16)
            batches = []
            self.classes["_bbox_around_paste"] = lambda loader, shape, *args: ([0, 0, 0], list(shape))
            for basic in (False, True):
                loader, consumed, _ = self.sampler(bank, draws, basic=basic)
                loader.batch_size, loader.patch_size_was_2d, loader.transforms = 2, False, None
                loader.get_indices = lambda: ["DEBUG_ZERO", "DEBUG_NEXT"]
                loader.get_do_oversample = lambda index: False
                loader.get_bbox = lambda shape, force, locations: ([0, 0, 0], list(shape))
                loader._data = SimpleNamespace(load_case=lambda case: (initial, labels, None, {"class_locations": {}}))
                batches.append(loader.generate_train_batch())
                self.assertEqual(consumed, draws)
            for batch in batches:
                self.assertEqual(batch["keys"], ["DEBUG_ZERO", "DEBUG_NEXT"])
                np.testing.assert_array_equal(batch["online_cp_applied"], [0, 1])
                np.testing.assert_array_equal(batch["data"][0].numpy(), initial)
                np.testing.assert_array_equal(batch["target"][0].numpy(), labels)
                self.assertEqual(int((batch["target"][1] == 2).sum()), 1)
            np.testing.assert_array_equal(batches[0]["online_cp_schedule_token"], batches[1]["online_cp_schedule_token"])
            np.testing.assert_array_equal(initial, np.full_like(initial, 100.))
            np.testing.assert_array_equal(labels, np.ones_like(labels))

    def test_actual_bank_case_loop_continues_after_exhaustive_zero(self):
        """Production outer/source loops and publication; GNN is a DEBUG boundary."""
        from hiercp.common import CasePaths
        stable_seed = online.stable_seed
        from hiercp.spatial import AdaptiveRoiBudgetError
        path = Path(online.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_online_bank")
        loop = next(node for node in function.body if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Tuple)
                    and [part.id for part in node.target.elts] == ["case_index", "case_id"])
        cases = {}
        for case_id, shape in (("DEBUG_ZERO", (9, 9, 9)), ("DEBUG_NEXT", (32, 32, 32))):
            label = np.ones(shape, dtype=np.int16)
            if case_id == "DEBUG_ZERO":
                label[2:7, 2:7, 2:7] = 2
            else:
                label[5, 5, 5] = 2
            cases[case_id] = SimpleNamespace(image=np.zeros(shape, np.float32), label=label, shape=shape,
                                           spacing=np.ones(3, np.float32), image_header=SimpleNamespace(get_xyzt_units=lambda: ("mm", "sec")))
        class DebugScorer:
            def submit(self, samples, callback):
                callback([np.linspace(0, 1, 128, dtype=np.float32)])
            def flush_ready(self):
                return None
        with tempfile.TemporaryDirectory(prefix="debug_online_bank_loop_") as directory:
            root = Path(directory)
            (root / "entries").mkdir()
            rows, mapping_calls = [], []
            def mapped_source(*args, **kwargs):
                mapping_calls.append(args[4].component_id)
                return np.ones((1, 1, 1, 1), np.float32), np.ones((1, 1, 1), bool), np.zeros(3, np.int64)
            context = dict(vars(online))
            context.update(np=np, ndi=ndi, os=os, build_candidate_pool=build_candidate_pool,
                           CasePaths=CasePaths, stable_seed=stable_seed,
                           train_ids=list(cases), case_map={case: SimpleNamespace(image=case, label=case) for case in cases},
                           load_case=lambda paths: cases[paths.case_id],
                           pre_dataset=SimpleNamespace(load_case=lambda case: (cases[case].image[None], cases[case].label[None], None, {})),
                           tumor_label=2, liver_label=1, min_diameter=0., max_diameter=100., global_seed=42,
                           source_inventory={}, entries_by_case={}, existing_by_source={}, overwrite=False,
                           REGION_CACHE_SEED_SALT="DEBUG", region_cache=root / "regions", graph_config=object(), ct_clip=(-160, 240),
                           load_or_build_patient_regions=lambda raw, **kwargs: SimpleNamespace(full_organ_mask=raw.label > 0,
                                                                                                organ_depth=np.ones(raw.shape, np.float32)),
                           generation=dict(source_pad=2, max_draws=64000, min_liver_coverage=.85,
                                           occupied_clearance_vox=2, min_center_separation_mm=0., min_center_separation_vox=12.),
                           metadata_contract={"no_placement_policy": "retain_original"},
                           candidate_count=128, draw_count=128, attempts=3, bank_root=root, entries_root=root / "entries",
                           commit_row=lambda row: rows.append(dict(row)),
                           _preprocessed_source=mapped_source, plans={"transpose_forward": [0, 1, 2]}, configuration="3d_fullres",
                           network_patch_size=np.array([32, 32, 32]), prepare_local_source=lambda *args, **kwargs: object(),
                           build_generation_specs=lambda pool, *args, **kwargs: [object() for _ in pool], bank=object(),
                           _map_raw_point=lambda point, *args: np.asarray(point, dtype=np.int32),
                           build_local_graph=lambda *args, **kwargs: object(),
                           build_inference_sample=lambda *args, **kwargs: (object(), {}),
                           scorer=DebugScorer(), AdaptiveRoiBudgetError=AdaptiveRoiBudgetError,
                           is_unrepresentable_geometry=lambda exc: False)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(path), "exec"), context)
            self.assertEqual([(row["case_id"], row["status"]) for row in rows],
                             [("DEBUG_ZERO", "no_placement"), ("DEBUG_NEXT", "ok")])
            self.assertEqual(mapping_calls, [1], "The proven-zero source must not need mapping or a graph")
            self.assertEqual(context["source_inventory"], {"DEBUG_ZERO": [1], "DEBUG_NEXT": [1]})
            self.assertEqual(len(context["entries_by_case"]["DEBUG_NEXT"]), 1)
            with np.load(root / rows[1]["entry"], allow_pickle=False) as entry:
                self.assertEqual(entry["scores"].shape, (128,))
            slots = online._source_slot_metadata(rows, context["source_inventory"], 128)
            self.assertEqual(slots["eligible_source_slots"], 2)
            self.assertEqual(slots["no_placement_sources"], 1)
