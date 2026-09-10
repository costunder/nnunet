"""DEBUG caller integration, not GNN/medical performance validation.

Executes both real bank resource/loop/publication/error paths, the real normal
inference assembler, ordered mapper and scoring disk-spool lifecycle. Geometry,
primitive graph work, resource scheduling and GNN output are explicit DEBUG
boundaries. All 128 candidates are retained; production settings are untouched.
"""
from __future__ import annotations

import ast
from contextlib import closing, ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hiercp import cache
from hiercp.common import CandidateInfo, CasePaths, LoadedCase
from hiercp.curriculum import CandidateSpec
from hiercp.local import BuiltLocalGraph, PreparedLocalSource
from hiercp.schema import GraphBuildConfig
from hiercp.spatial import AdaptiveRoiBudgetError, CanonicalGraphUnavailable
from tools import online_bank_preparation as preparation
from tools import online_cp_benchmark as normal
from tools import online_cp_argmax_benchmark as argmax
from tools.online_scoring import PendingBankScorer
from tools.online_raw_bank_preparation import prepare_raw_case, prepare_source_candidates


def debug_native_identity_inputs(image, label):
    """Real native-resampler oracle on an explicit DEBUG identity CT plan."""
    from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
    shape = list(image.shape)
    data_options = dict(is_seg=False, order=3, order_z=0, force_separate_z=None)
    seg_options = dict(is_seg=True, order=1, order_z=0, force_separate_z=None)
    plans = {
        "image_reader_writer": "NibabelIO", "transpose_forward": [2, 1, 0],
        "foreground_intensity_properties_per_channel": {
            "0": dict(mean=0., std=1., percentile_00_5=-1e6, percentile_99_5=1e6)},
        "configurations": {"3d_fullres": {
            "architecture": {}, "spacing": [1., 1., 1.],
            "preprocessor_name": "DefaultPreprocessor", "normalization_schemes": ["CTNormalization"],
            "use_mask_for_norm": [False], "resampling_fn_data": "resample_data_or_seg_to_shape",
            "resampling_fn_data_kwargs": data_options, "resampling_fn_seg": "resample_data_or_seg_to_shape",
            "resampling_fn_seg_kwargs": seg_options}},
    }
    properties = {"spacing": [1., 1., 1.], "shape_before_cropping": shape,
                  "shape_after_cropping_and_before_resampling": shape,
                  "bbox_used_for_cropping": [[0, value] for value in shape]}
    # Reader ZYX followed by the declared reverse transpose restores XYZ.
    # All fixture values lie within the identity CT normalization bounds.
    data = resample_data_or_seg_to_shape(image[None].astype(np.float32), shape,
                                        [1.] * 3, [1.] * 3, **data_options)
    seg = resample_data_or_seg_to_shape(label[None].astype(np.int16), shape,
                                       [1.] * 3, [1.] * 3, **seg_options)
    return properties, plans, data, seg


def debug_prepare_raw_candidates(*args, **kwargs):
    # Only resource scheduling is a DEBUG boundary: every supplied target is
    # processed by the real engine and written with the real storage format.
    return prepare_source_candidates(
        *args, **kwargs, candidate_map=lambda function, tasks: [function(task) for task in tasks])


def caller_work_scope(module):
    """Keep actual resource scopes, case/source loops, publication and guards.

    Environment/checkpoint setup precedes this scope and is tested separately;
    final calibrated report/index publication follows it and is not forged.
    """
    path = Path(module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "build_online_bank")
    helpers = [node for node in function.body if isinstance(node, ast.FunctionDef)
               and node.name in {"is_unrepresentable_geometry", "scoring_sample"}]
    start = next(index for index, node in enumerate(function.body)
                 if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == "BankLocalGraphMapper")
    stop = next(index for index, node in enumerate(function.body[start:], start)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "validate_scoring_report")
    selected = ast.Module(body=[ast.ImportFrom(module="__future__",
                names=[ast.alias(name="annotations")], level=0), *helpers,
                *function.body[start:stop]], type_ignores=[])
    return compile(ast.fix_missing_locations(selected), str(path), "exec")


class DebugBankBoundary:
    """Synthetic framework boundary around actual bank orchestration."""
    def __init__(self, module, root, *, fault=None, existing_rows=()):
        self.module, self.root, self.fault = module, root, fault
        self.rows = [dict(row) for row in existing_rows]
        self.trace, self.scorers, self.progress, self.proposals = [], [], [], []
        self.ordinal = 0
        self.is_argmax = module is argmax
        self.pool_count = 2 if self.is_argmax else 1  # Explicit DEBUG profile.
        (root / "entries").mkdir(exist_ok=True)
        # Region serialization is an existing DEBUG boundary; represent its
        # already-present shared cache instead of bypassing the reuse guard.
        (root / "regions" / "DEBUG_WIRING").mkdir(parents=True, exist_ok=True)
        shape = (32, 32, 32)
        label = np.ones(shape, dtype=np.uint8)
        label[5, 5, 5] = 2
        self.case = LoadedCase(
            CasePaths("DEBUG_WIRING", Path("DEBUG_IMAGE"), Path("DEBUG_LABEL")),
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 100., label,
            np.eye(4), np.eye(4), SimpleNamespace(get_xyzt_units=lambda: ("mm", "sec")),
            None, np.ones(3, np.float32), {"debug": True}, {"debug": True},
        )
        self.centers = np.asarray([(12, 8 + index // 16, 8 + index % 16)
                                   for index in range(128)], dtype=np.int32)
        self.source_nodes = self.node_table("source_context", 0.0)
        self.source_edges = {("source_context", "context_neighbor", "source_context"):
                             torch.empty((2, 0), dtype=torch.int32)}
        self.prepared = PreparedLocalSource(
            np.ones((1, 1, 1), bool), np.ones((5, 16, 16, 16), np.float32),
            self.source_nodes, self.source_edges, {"source_context": 1},
        )
        self.source_local = {"format": "canonical-full-v22",
                             "nodes": self.source_nodes, "edges": self.source_edges}

    @staticmethod
    def node_table(kind, value):
        return {kind: {"x": torch.full((1, 16), value, dtype=torch.float16),
                       "grid": torch.zeros((1, 3), dtype=torch.float16),
                       "pos": torch.zeros((1, 3)), "pos_mm": torch.zeros((1, 3))}}

    def proposal(self, case, source, **kwargs):
        centers = np.roll(self.centers, self.ordinal, axis=0)
        self.ordinal += 1
        self.proposals.append(centers.copy())
        self.trace.append(("proposal", len(centers)))
        return [CandidateInfo(tuple(map(int, center)), (slice(0, 1),) * 3,
                              1.0, 5.0, 5.0, 20.0, 4.0) for center in centers], None

    def specs(self, candidates, *args, **kwargs):
        return [CandidateSpec(candidate.center, 1, 0, 0, 0, 1.0, 5.0, 5.0, 20.0, 4.0)
                for candidate in candidates]

    def prepare_source(self, *args, **kwargs):
        self.trace.append(("prepare_source",))
        return self.prepared

    def geometry(self, case, source, spec, **kwargs):
        self.trace.append(("geometry", spec.center))
        if kwargs["prepared_source"] is not self.prepared:
            raise AssertionError("Caller did not share the prepared source")
        if self.fault == "geometry_budget":
            raise AdaptiveRoiBudgetError("DEBUG measured full ROI budget")
        if self.fault == "all_geometry":
            raise CanonicalGraphUnavailable("DEBUG unavailable target geometry")

    def graph(self, case, source, spec, **kwargs):
        self.trace.append(("graph", spec.center))
        if kwargs["prepared_source"] is not self.prepared:
            raise AssertionError("Inference rebuilt or replaced the prepared source")
        if self.fault == "graph_memory":
            raise MemoryError("DEBUG full graph allocation")
        value = float(spec.center[1] * 32 + spec.center[2])
        target = {"format": "canonical-full-v22", "nodes": self.node_table("target_context", value),
                  "edges": {("target_context", "context_neighbor", "target_context"):
                            torch.empty((2, 0), dtype=torch.int32)}}
        return BuiltLocalGraph(None, self.prepared.source_patch,
                               np.full((5, 16, 16, 16), value, np.float32),
                               self.source_local, target)

    def scheduler(self, *, tasks, function, commit, **kwargs):
        self.trace.append(("map", len(tasks), kwargs["workers"]))
        # Actual mapper must restore input order even when completion reverses.
        for task in reversed(tasks):
            commit(function(task))
        return {"DEBUG_scheduler": True, "completed": len(tasks)}

    def run(self):
        boundary = self
        class DebugProgress:
            def __init__(self, root):
                self.closed = False
                boundary.progress.append(self)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.closed = True
                return False
            def update(self, *args, **kwargs):
                return None
            def counters(self, **kwargs):
                return None

        class DebugScoreBoundary(PendingBankScorer):
            # Real __init__, submit, disk serialization and close are retained.
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                boundary.scorers.append(self)
                if kwargs.get("spool_directory") != boundary.root or self.spool_root is None:
                    raise AssertionError("Caller did not request disk-backed scoring")
            def flush_ready(self):
                return None  # Deliberately defer callbacks to exercise ownership.
            def flush(self):
                if boundary.fault == "score_io":
                    raise OSError("DEBUG spool read failure")
                while self.pending:
                    group, callback = self.pending.pop(0)
                    scores = []
                    for entry in group:
                        if "path" not in entry or "sample" in entry:
                            raise AssertionError("Canonical sample stayed in memory instead of disk spool")
                        sample = torch.load(entry["path"], map_location="cpu", weights_only=False, mmap=True)
                        centers = sample["candidate_centers"].numpy()
                        boundary.trace.append(("score", centers.copy()))
                        scores.append((centers[:, 1] * 32 + centers[:, 2]).astype(np.float32))
                        del sample
                    callback(scores)
            def report(self):
                return {"DEBUG_score_boundary": True}

        def commit(row):
            self.rows.append(dict(row))
        config = GraphBuildConfig(patch_size=16)
        population = SimpleNamespace(fingerprint=lambda: "DEBUG_PROTOTYPE")
        properties, native_plans, native_data, native_seg = debug_native_identity_inputs(
            self.case.image, self.case.label)
        context = dict(vars(self.module))
        context.update(
            np=np, torch=torch, CasePaths=CasePaths, CACHE_FORMAT=cache.CACHE_FORMAT,
            train_ids=[self.case.paths.case_id],
            case_map={self.case.paths.case_id: SimpleNamespace(image="DEBUG_IMAGE", label="DEBUG_LABEL")},
            load_case=lambda paths: self.case,
            pre_dataset=SimpleNamespace(load_case=lambda name: (native_data, native_seg, None, properties)),
            min_diameter=0.0, max_diameter=100.0, tumor_label=2, liver_label=1, global_seed=42,
            rows=self.rows, source_inventory={}, entries_by_case={}, overwrite=False,
            existing_by_source={(row["case_id"], int(row["source_component"])): row for row in self.rows},
            REGION_CACHE_SEED_SALT="DEBUG", region_cache=self.root / "regions",
            graph_config=config, ct_clip=(-160.0, 240.0),
            load_or_build_patient_regions=lambda raw, **kwargs: SimpleNamespace(
                full_organ_mask=raw.label > 0, organ_depth=np.ones(raw.shape, np.float32)),
            generation=dict(source_pad=2, max_draws=64000, min_liver_coverage=.85,
                occupied_clearance_vox=2, min_center_separation_mm=0.0, min_center_separation_vox=12.0,
                scoring_batch_size="auto", scoring_batch_size_candidates=[1, 2],
                scoring_batch_calibration_repeats=3, scoring_batch_max_vram_fraction=.8,
                local_candidate_chunk_size=128, amp=False, pin_memory=False),
            metadata_contract={"no_placement_policy": "retain_original"},
            candidate_count=128, draw_count=128, attempts=1, pools_per_source=self.pool_count,
            bank_root=self.root, entries_root=self.root / "entries", manifest_path=self.root / "DEBUG_manifest.csv",
            commit_row=commit, build_candidate_pool=self.proposal,
            _preprocessed_source=lambda *args, **kwargs: (np.ones((1, 1, 1, 1), np.float32),
                np.ones((1, 1, 1), bool), np.zeros(3, np.int64)),
            plans=native_plans, configuration="3d_fullres",
            nn_cfg={"runtime": {"minimum_free_gb_before_preprocess": 0}},
            prepare_raw_case=prepare_raw_case, prepare_source_candidates=debug_prepare_raw_candidates,
            network_patch_size=np.array([32, 32, 32]), prepare_local_source=self.prepare_source,
            build_generation_specs=self.specs, bank=population, population_bank=population,
            _map_raw_point=lambda point, *args: np.asarray(point, dtype=np.int32),
            validate_local_geometry=self.geometry, build_local_graph=self.graph,
            build_inference_sample=cache.build_inference_sample,
            build_patient_graph=lambda *args, **kwargs: {"DEBUG_patient": torch.ones(1)},
            build_prototype_graph=lambda *args, **kwargs: {"DEBUG_population": torch.ones(1)},
            closing=closing, BankProgress=DebugProgress, PendingBankScorer=DebugScoreBoundary,
            BankLocalGraphMapper=preparation.BankLocalGraphMapper,
            AdaptiveRoiBudgetError=AdaptiveRoiBudgetError, CanonicalGraphUnavailable=CanonicalGraphUnavailable,
            model=object(), device=torch.device("cpu"),
        )
        self.context = context
        with ExitStack() as stack:
            stack.enter_context(patch.object(preparation, "run_case_jobs", self.scheduler))
            stack.enter_context(patch.object(cache, "build_generation_specs", self.specs))
            stack.enter_context(patch.object(cache, "build_local_graph", self.graph))
            stack.enter_context(patch.object(cache, "prepare_local_source",
                                            side_effect=AssertionError("Inference rebuilt source")))
            stack.enter_context(patch.object(cache, "build_patient_graph", context["build_patient_graph"]))
            stack.enter_context(patch.object(cache, "build_prototype_graph", context["build_prototype_graph"]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                exec(caller_work_scope(self.module), context)
            finally:
                # Unlike a returned production function, this AST fixture keeps
                # its locals for assertions. Release only its owned mappings
                # after all consumers finish, before TemporaryDirectory cleanup.
                store = context.get("reused_raw_store")
                if store is not None:
                    store.close()


class OnlineBankWiringDebugTests(unittest.TestCase):
    def assert_closed(self, boundary):
        self.assertEqual(len(boundary.scorers), 1)
        self.assertTrue(boundary.scorers[0]._closed)
        self.assertFalse(boundary.scorers[0].spool_root.exists())
        self.assertTrue(boundary.progress[0].closed)
        self.assertTrue(boundary.root.is_dir())

    def test_debug_both_callers_keep_128_order_single_graph_work_and_real_disk_spool(self):
        for module in (normal, argmax):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory(prefix="debug_bank_wiring_") as directory:
                boundary = DebugBankBoundary(module, Path(directory))
                boundary.run()
                self.assert_closed(boundary)
                kinds = [event[0] for event in boundary.trace]
                self.assertEqual(kinds.count("prepare_source"), 1)
                self.assertEqual(kinds.count("geometry"), boundary.pool_count * 128)
                self.assertEqual(kinds.count("graph"), boundary.pool_count * 128)
                self.assertEqual(kinds.count("map"), boundary.pool_count)
                for index, event in enumerate(boundary.trace):
                    if event[0] == "map":
                        self.assertEqual(event[1:], (128, "auto"))
                        self.assertEqual([item[0] for item in boundary.trace[:index]].count("geometry") % 128, 0)
                self.assertGreater(boundary.scorers[0].spool_io["written_bytes"], 0)
                self.assertEqual(len(boundary.rows), 1)
                row = boundary.rows[0]
                self.assertEqual((row["status"], row["candidate_count"]), ("ok", 128))
                expected = np.stack(boundary.proposals) if boundary.is_argmax else boundary.proposals[0]
                with np.load(boundary.root / row["entry"], allow_pickle=False) as entry:
                    np.testing.assert_array_equal(entry["candidate_raw_centers"], expected)
                    np.testing.assert_array_equal(entry["candidate_centers"], expected)
                    np.testing.assert_array_equal(entry["scores"], (expected[..., 1] * 32 + expected[..., 2]).astype(np.float32))
                    if boundary.is_argmax:
                        np.testing.assert_array_equal(entry["argmax_indices"], np.argmax(entry["scores"], axis=1))
                    else:
                        self.assertEqual(entry["paste_contract"].tolist(), [normal.PASTE_CONTRACT])
                # Actual reader validates every published dimension/content hash.
                args = (boundary.root, boundary.context["entries_by_case"], 128)
                if boundary.is_argmax:
                    args += (boundary.pool_count,)
                expected_contract = {} if boundary.is_argmax else {
                    "expected_paste_contract": normal.PASTE_CONTRACT,
                }
                module._audit_bank_entries(*args, {row["entry"]: row}, **expected_contract)

    def test_debug_partial_bank_reuse_skips_work_and_keeps_native_dimension_guard(self):
        for module in (normal, argmax):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory(prefix="debug_bank_reuse_") as directory:
                root = Path(directory)
                first = DebugBankBoundary(module, root)
                first.run()
                row = first.rows[0]
                before = (root / row["entry"]).read_bytes()
                resumed = DebugBankBoundary(module, root, existing_rows=first.rows)
                resumed.run()
                self.assert_closed(resumed)
                self.assertEqual(resumed.trace, [])
                self.assertEqual(resumed.context["entries_by_case"], first.context["entries_by_case"])
                self.assertEqual((root / row["entry"]).read_bytes(), before)
                changed = DebugBankBoundary(module, root, existing_rows=[dict(row, candidate_count=127)])
                with self.assertRaisesRegex(module.OnlineBenchmarkError, "Cached entry"):
                    changed.run()
                self.assert_closed(changed)
                self.assertEqual((root / row["entry"]).read_bytes(), before)

    def test_debug_errors_keep_native_failure_contract_and_close_owned_resources(self):
        for module in (normal, argmax):
            for fault, error in (("geometry_budget", AdaptiveRoiBudgetError),
                                 ("all_geometry", module.OnlineBenchmarkError),
                                 ("graph_memory", module.OnlineBenchmarkError if module is argmax else MemoryError),
                                 ("score_io", OSError)):
                with self.subTest(module=module.__name__, fault=fault), tempfile.TemporaryDirectory(prefix="debug_bank_failure_") as directory:
                    boundary = DebugBankBoundary(module, Path(directory), fault=fault)
                    with self.assertRaises(error):
                        boundary.run()
                    self.assert_closed(boundary)
                    self.assertEqual(list((boundary.root / "entries").iterdir()), [])
                    self.assertFalse(any(row["status"] == "no_placement" for row in boundary.rows))
                    if fault == "all_geometry":
                        self.assertEqual(boundary.rows[0]["status"], "insufficient_candidates")
                        self.assertEqual(sum(event[0] == "graph" for event in boundary.trace), 0)


if __name__ == "__main__":
    unittest.main()
