"""UNIT contract tests with synthetic metadata and mocked original operators.

These tests run no neural model and make no CT or prediction-quality claim.
"""
from dataclasses import dataclass
from contextlib import contextmanager, redirect_stdout, nullcontext
import copy
import ast
import hashlib
from io import StringIO
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch

from hiercp_v1x.u_bridge_data import (UBridgeData, _sha, _resident_size, _verify_original_core,
                                     _distance_to_other_tumor_mm, P_METADATA_CONTRACT)
from hiercp_v1x import u_bridge_fields


@contextmanager
def unit_directory():
    # Keep every owned synthetic artifact in the authorized workspace.
    with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
        try:
            yield directory
        finally:
            # Close only this UNIT fixture's owned mappings before Windows
            # removes its own temporary files.
            for key, (arrays, _) in list(u_bridge_fields._OPENED.items()):
                if Path(key[0]).is_relative_to(Path(directory)):
                    u_bridge_fields._OPENED.pop(key)
                    for array in arrays:
                        array._mmap.close()


@dataclass
class UnitCandidate:
    center: tuple
    slices: tuple
    liver_coverage: float
    border_distance_mm: float
    occupied_distance_mm: float
    context_mean_hu: float
    context_std_hu: float


@dataclass
class UnitBuilt:
    source_patch: np.ndarray
    target_patch: np.ndarray
    source_local: dict
    target_local: dict


class UnitFixture:
    def __init__(self, root, *, two=False, resident=2 ** 20):
        self.root = Path(root)
        self.original = dict(case_id="UNIT_case", sample_index=0, source_component=1,
            prototype_fingerprint="UNIT_bank",
            split="train", candidate_centers=torch.tensor([(7, 7, 7)] + list(np.ndindex(8, 8, 8))[:7]),
            difficulties=torch.tensor([0, 1, 1, 2, 2, 3, 3, 3]))
        self.centers = list(np.ndindex(8, 8, 8))[:128]
        label = np.ones((8, 8, 8), dtype=np.int16)
        label[7, 7, 7] = 2
        self.case = SimpleNamespace(paths=SimpleNamespace(case_id="UNIT_case"),
            image=np.ones(label.shape, dtype=np.float32), label=label,
            shape=label.shape, spacing=np.ones(3, dtype=np.float32))
        self.raw = []
        for kind in ("image", "label"):
            (self.root / kind).write_bytes(b"explicit UNIT synthetic file")
        self.raw.append(dict(case_id="UNIT_case", image=str(self.root / "image"),
            label=str(self.root / "label"), image_sha256=_sha(self.root / "image"),
            label_sha256=_sha(self.root / "label"), shape=list(label.shape), spacing=[1., 1., 1.],
            comparison=dict(centers=self.centers)))
        self.config = dict(seed=42, labels=dict(liver=1, tumor=2), ct_clip=[-200., 250.],
            cache=dict(total_candidates=8, source_selection="random", source_pad=2),
            graph=dict(adaptive_roi_margin_mm=10., context_outer_radius_mm=10.))
        self.source = SimpleNamespace(component_id=1, anchor_center=(7, 7, 7),
            patch_mask=np.ones((1, 1, 1), dtype=bool), voxel_count=1,
            full_mask=label == 2, patch_slices=(slice(7, 8),) * 3,
            patch_image=self.case.image[7:8, 7:8, 7:8].copy())
        self.build_calls = []
        self.source_calls = 0
        self.view_calls = []
        self.guards = 0
        def guard():
            self.guards += 1
        samples = [self.original]
        if two:
            samples.append(dict(self.original, split="val", sample_index=1))
        self.provider = UBridgeData(samples, self.raw, self.config,
            SimpleNamespace(fingerprint=lambda: "UNIT_bank"), self.root / "output",
            workers=2, resident_bytes=resident, budget=guard)
        self.provider._scope = dict(contract_sha256="UNIT_bounded_m10",
            original_module_sha256={"hiercp/common.py": hashlib.sha256(b"UNIT_original_common").hexdigest()})

        def extract(array, center, shape, pad_value=0):
            # A tiny explicit synthetic fixture operator, never a production path.
            return np.asarray(array[tuple(center)]).reshape(shape)
        def choose(*args, **kwargs):
            return self.source, None, 1
        def prepare(*args, **kwargs):
            self.source_calls += 1
            return SimpleNamespace(source_patch=np.zeros((1, 1, 1), np.float32),
                                   source_local={"UNIT": torch.ones(2)})
        def inference(case, source, candidates, bank, **kwargs):
            specs = [SimpleNamespace(center=x.center, corruption=0,
                rotation_matrix=np.eye(3), scale_array=np.ones(3)) for x in candidates]
            def build(spec):
                self.build_calls.append(tuple(spec.center))
                prepared = kwargs["prepared_source"]
                return UnitBuilt(prepared.source_patch, np.zeros((1, 1, 1), np.float32),
                    prepared.source_local, dict(center=spec.center, value=torch.tensor(spec.center)))
            built = kwargs["local_graph_map"](build, specs,
                case_id=case.paths.case_id, source_component=source.component_id)
            return dict(candidate_centers=torch.tensor([x.center for x in specs]),
                source_local=built[0].source_local, target_locals=[x.target_local for x in built],
                source_patch=torch.zeros(1), target_patches=torch.zeros(len(built), 1),
                patient_graph={"UNIT_joint_count": len(built)},
                prototype_graph={"UNIT_joint_count": len(built)}), specs
        def views(sample, *, training, epoch, global_seed):
            self.view_calls.append((sample["case_id"], sample["sample_index"], training, epoch, global_seed))
            sample["UNIT_views"] = tuple(np.random.default_rng(
                global_seed + (epoch if training else 0) + sample["sample_index"]).integers(0, 1000, 2).tolist())
            return sample
        def collate(samples):
            return SimpleNamespace(counts=tuple(len(s["candidate_centers"]) for s in samples),
                                   UNIT_samples=samples)
        self.runtime = SimpleNamespace(common=SimpleNamespace(
            CandidateInfo=UnitCandidate, CasePaths=lambda *args: args,
            load_case=lambda paths: self.case, organ_depth_mm=lambda organ, spacing: organ.astype(float),
            distance_to_mask_mm=lambda mask, spacing: (~mask).astype(float),
            choose_source_tumor=choose, stable_case_seed=lambda *args: 42,
            extract_centered_patch=extract,
            context_stats_for_local_mask=lambda *args, **kwargs: (1., 0.)),
            schema=SimpleNamespace(graph_config_from_dict=lambda value: value),
            local=SimpleNamespace(prepare_local_source=prepare),
            cache=SimpleNamespace(build_inference_sample=inference),
            sample=SimpleNamespace(materialize_sample_views=views),
            data=SimpleNamespace(collate_samples=collate))
        self.provider._runtime = lambda: self.runtime
        self.provider._regions = lambda case: SimpleNamespace(UNIT=True)
        # Synthetic data-operator tests do not execute the original hierarchy;
        # exact original static-helper persistence has its own UNIT suite.
        self.provider._upper_context = lambda *args: nullcontext()


class BridgeDataTests(unittest.TestCase):
    def test_native_rotation_retains_original_update_work_and_covers128_across40epochs(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            seen = set()
            for epoch in range(1, 41):
                keys = f.provider.candidate_keys(0, "native", epoch)
                self.assertEqual(len(keys), 8)
                self.assertEqual(keys[0], "P")
                self.assertEqual(len(set(keys)), 8)
                seen.update(int(k.split(":")[1]) for k in keys[1:])
            self.assertEqual(seen, set(range(128)))
            self.assertEqual(f.provider.candidate_keys(0, "native", 19),
                             ("P", "U:126", "U:127", "U:0", "U:1", "U:2", "U:3", "U:4"))
            self.assertEqual(f.provider.candidate_keys(0, "selected", 40),
                             ("P", "S:0", "S:1", "S:2", "S:3", "S:4", "S:5", "S:6"))

    def test_missing_or_duplicate128_metadata_fails_without_resampling(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            for mode in ("missing", "short", "duplicate"):
                raw = copy.deepcopy(f.raw)
                if mode == "missing":
                    raw[0].pop("comparison")
                elif mode == "short":
                    raw[0]["comparison"]["centers"] = f.centers[:127]
                else:
                    raw[0]["comparison"]["centers"][1] = f.centers[0]
                with self.assertRaises(ValueError):
                    UBridgeData([f.original], raw, f.config, f.provider.bank,
                        Path(directory) / mode, 2, 1000, lambda: None)

    def test_original_prototype_bank_identity_must_match_and_stay_fixed(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            wrong = dict(f.original, prototype_fingerprint="UNIT_other_bank")
            with self.assertRaisesRegex(ValueError, "prototype fingerprint differs"):
                UBridgeData([wrong], f.raw, f.config, f.provider.bank,
                    Path(directory) / "wrong", 2, 1000, lambda: None)
            f.provider.bank.fingerprint = lambda: "UNIT_changed_bank"
            with self.assertRaisesRegex(ValueError, "bank identity changed"):
                f.provider.sample(0, ("P", "U:0"), 0, False)

    def test_original_source_split_and_sample_identity_are_used_for_views(self):
        with unit_directory() as directory:
            f = UnitFixture(directory, two=True)
            self.assertEqual([x["index"] for x in f.provider.examples("train")], [0])
            self.assertEqual([x["index"] for x in f.provider.examples("inner_val")], [1])
            original = copy.deepcopy(f.original)
            result = f.provider.sample(1, f.provider.candidate_keys(1, "selected", 3), 3, True)
            self.assertEqual(result["split"], "val")
            self.assertEqual(result["sample_index"], 1)
            self.assertEqual(result["source_component"], 1)
            self.assertEqual(f.view_calls[-1], ("UNIT_case", 1, True, 3, 42))
            torch.testing.assert_close(f.original["candidate_centers"], original["candidate_centers"])
            torch.testing.assert_close(f.original["difficulties"], original["difficulties"])
            self.assertEqual(result["difficulties"].tolist(), [0] + [1] * 7)
            self.assertEqual(result["corruptions"].tolist(), [0] * 8)

    def test_full_evaluation_preserves129_joint_candidates_in_both_arms(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            a = f.provider.batch([0], "selected", 0, False, full=True)
            b = f.provider.batch([0], "native", 0, False, full=True)
            self.assertEqual(a.counts, (129,))
            self.assertEqual(a.bridge_candidate_keys, b.bridge_candidate_keys)
            self.assertEqual(a.bridge_source_ids, ("UNIT_case:0",))
            sample = a.UNIT_samples[0]
            self.assertEqual(sample["patient_graph"]["UNIT_joint_count"], 129)
            self.assertEqual(sample["prototype_graph"]["UNIT_joint_count"], 129)
            self.assertEqual(sample["candidate_centers"][0].tolist(), [7, 7, 7])
            self.assertEqual(sample["candidate_centers"][1:].tolist(), [list(x) for x in f.centers])
            self.assertEqual(len(f.build_calls), 129)
            with self.assertRaises(ValueError):
                f.provider.batch([0], "native", 1, True, full=True)

    def test_candidates_are_directly_built_without_CP_admission_or_transforms(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            # UNIT U at volume face would fail a full-footprint in-bounds filter.
            sample = f.provider.sample(0, ("P", "U:0"), 0, False)
            self.assertEqual(sample["candidate_centers"][1].tolist(), [0, 0, 0])
            self.assertFalse(f.provider.report()["CP_candidate_search_used"])
            self.assertFalse(f.provider.report()["U_resampled"])

    def test_frozen_U_actual_label_mismatch_and_raw_hash_change_fail(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            f.case.label[f.centers[0]] = 2
            with self.assertRaisesRegex(ValueError, "label1 center"):
                f.provider.sample(0, ("P", "U:0"), 0, False)
        with unit_directory() as directory:
            f = UnitFixture(directory)
            (Path(directory) / "label").write_bytes(b"changed UNIT raw")
            with self.assertRaisesRegex(ValueError, "SHA256 changed"):
                f.provider.sample(0, ("P", "U:0"), 0, False)

    def test_wrong_source_component_is_rejected(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            f.source.component_id = 2
            with self.assertRaisesRegex(ValueError, "source component/anchor changed"):
                f.provider.sample(0, ("P", "U:0"), 0, False)

    def test_positive_distance_scalar_matches_original_EDT_with_anisotropic_float32_spacing(self):
        from scipy import ndimage
        rng = np.random.default_rng(20261006)
        for _ in range(50):
            shape = tuple(rng.integers(2, 9, 3).tolist())
            label = np.where(rng.random(shape) < .25, 2, 1).astype(np.int16)
            source_mask = rng.random(shape) < .2
            center = tuple(int(rng.integers(n)) for n in shape)
            source_mask[center] = True
            spacing = rng.uniform(.3, 5., 3)
            other = (label == 2) & ~source_mask
            expected = (float(ndimage.distance_transform_edt(~other,
                sampling=np.asarray(spacing, dtype=np.float32)).astype(np.float32)[center])
                if other.any() else np.inf)
            actual = _distance_to_other_tumor_mm(label, source_mask, center, spacing,
                slab_voxels=shape[1] * shape[2], point_chunk=3)
            self.assertEqual(actual, expected)
        label = np.full((3, 4, 5), 2, np.int16)
        self.assertEqual(_distance_to_other_tumor_mm(label, np.ones_like(label, dtype=bool),
            (1, 2, 3), (.7, .7, 2.)), np.inf)
        label[:] = 1
        self.assertEqual(_distance_to_other_tumor_mm(label, np.zeros_like(label, dtype=bool),
            (1, 2, 3), (.7, .7, 2.)), np.inf)

    def test_P_metadata_uses_original_source_patch_and_U_metadata_keeps_all_tumors(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            # Source patch differs from the centered target CT crop in this
            # explicit UNIT case, so using the wrong context input is visible.
            f.source.patch_image = np.full((1, 1, 1), 123., dtype=np.float32)
            f.case.label[5, 7, 7] = 2  # a separate true component outside source
            contexts = []
            def context(image, organ, footprint, ring_width):
                contexts.append(image)
                self.assertEqual(ring_width, 3)
                return float(image.mean()), float(image.std())
            f.runtime.common.context_stats_for_local_mask = context
            case, organ, depth, occupied = f.provider._case("UNIT_case")
            example = f.provider._example(0)
            positive = f.provider._positive_candidate(example, case, f.source, organ, depth)
            self.assertIs(contexts[-1], f.source.patch_image)
            self.assertEqual(positive.liver_coverage, 1.)
            self.assertEqual(positive.context_mean_hu, 123.)
            self.assertEqual(positive.border_distance_mm, float(depth[7, 7, 7]))
            self.assertEqual(positive.occupied_distance_mm, 2.)
            with patch("hiercp_v1x.u_bridge_data._distance_to_other_tumor_mm",
                       side_effect=AssertionError("Cached P scalar recomputed")):
                self.assertIs(f.provider._positive_candidate(example, case, f.source, organ, depth), positive)
            # Comparator extraction remains generic original-negative metadata.
            at_source = f.provider._candidate((7, 7, 7), case, f.source, organ, depth, occupied)
            self.assertEqual(at_source.liver_coverage, 0.)
            self.assertEqual(at_source.occupied_distance_mm, 0.)
            self.assertEqual(at_source.context_mean_hu, 1.)
            sample = f.provider.sample(0, ("P", "U:0"), 0, False)
            self.assertEqual(sample["candidate_metadata_contract"], P_METADATA_CONTRACT)
            self.assertEqual(f.provider.report()["positive_metadata_builds"], 1)

    def test_original_canonical_local_graph_does_not_consume_upper_candidate_metadata(self):
        # Audits the preserved real source, justifying reuse of unchanged local
        # geometry cache entries while upper graphs receive corrected P fields.
        archive = Path(__file__).resolve().parents[1] / "versions/v1/pipeline_v1_source.zip"
        with ZipFile(archive) as bundle:
            tree = ast.parse(bundle.read("hiercp/local.py").decode())
        spec_fields = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                       and isinstance(node.value, ast.Name) and node.value.id == "spec"}
        self.assertEqual(spec_fields, {"rotation_matrix", "scale_array", "center"})
        dynamic_access = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "vars")
            and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "spec"]
        self.assertEqual(dynamic_access, [])

    def test_static_canonical_cache_reused_but_views_follow_current_epoch(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            a = f.provider.sample(0, ("P", "S:0"), 1, True)
            b = f.provider.sample(0, ("P", "S:0"), 2, True)
            self.assertEqual(len(f.build_calls), 2)
            self.assertNotEqual(a["UNIT_views"], b["UNIT_views"])
            with f.provider._lock:
                f.provider._cache.clear()
                f.provider._resident_bytes = 0
            c = f.provider.sample(0, ("P", "S:0"), 3, True)
            self.assertEqual(len(f.build_calls), 2)
            self.assertEqual(f.provider.stats["disk_hits"], 2)
            self.assertNotEqual(c["UNIT_views"], b["UNIT_views"])

    def test_whole_case_fields_preserve_original_math_and_do_not_recompute_after_LRU_or_reopen(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            calls = dict(depth=0, occupied=0)
            expected_organ = (f.case.label == 1) | (f.case.label == 2)
            expected_tumor = f.case.label == 2
            def depth(mask, spacing):
                calls["depth"] += 1
                np.testing.assert_array_equal(mask, expected_organ)
                np.testing.assert_array_equal(spacing, f.case.spacing)
                return mask.astype(np.float64)
            def occupied(mask, spacing):
                calls["occupied"] += 1
                np.testing.assert_array_equal(mask, expected_tumor)
                np.testing.assert_array_equal(spacing, f.case.spacing)
                return (~mask).astype(np.float32)
            f.runtime.common.organ_depth_mm = depth
            f.runtime.common.distance_to_mask_mm = occupied
            stream = StringIO()
            with redirect_stdout(stream):
                first = f.provider._case("UNIT_case")
                self.assertIsInstance(first[2], np.memmap)
                self.assertEqual(first[2].dtype, np.dtype(np.float64))
                np.testing.assert_array_equal(first[3], (~expected_tumor).astype(np.float32))
                f.provider._cache.clear()
                f.provider._resident_bytes = 0
                # Returning to this case after LRU eviction reuses a mapping;
                # hashing the large field files again would reintroduce I/O.
                with patch.object(u_bridge_fields, "_sha", side_effect=AssertionError("Mapped fields rehashed")):
                    second = f.provider._case("UNIT_case")
                self.assertIs(first[2], second[2])
                f.provider._cache.clear()
                f.provider._resident_bytes = 0
                for key, (arrays, _) in list(u_bridge_fields._OPENED.items()):
                    u_bridge_fields._OPENED.pop(key)
                    for array in arrays:
                        array._mmap.close()
                third = f.provider._case("UNIT_case")
                self.assertFalse(third[2].flags.writeable)
            self.assertEqual(calls, {"depth": 1, "occupied": 1})
            report = f.provider.report()
            self.assertEqual((report["field_builds"], report["field_mapping_reuses"], report["field_reopens"]), (1, 1, 1))
            self.assertEqual(report["whole_case_fields"]["cases"], 1)
            self.assertEqual(report["whole_case_fields"]["array_bytes"], 8 ** 3 * (8 + 4))
            self.assertGreater(report["whole_case_fields"]["disk_bytes"], report["whole_case_fields"]["array_bytes"])
            self.assertIn("status=built", stream.getvalue())
            self.assertIn("status=reopened", stream.getvalue())
            self.assertGreaterEqual(report["field_reopen_seconds"], 0.)

    def test_corrupt_cache_binding_fails_and_is_not_replaced(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            f.provider.sample(0, ("P", "S:0"), 1, True)
            path = next(f.provider.graph_dir.glob("*.pt"))
            saved = torch.load(path, weights_only=False)
            saved["binding"]["source_component"] = 99
            torch.save(saved, path)  # Deliberate UNIT corruption of this test's file.
            f.provider._cache.clear()
            f.provider._resident_bytes = 0
            with self.assertRaisesRegex(ValueError, "cache identity changed"):
                f.provider.sample(0, ("P", "S:0"), 2, True)
            self.assertEqual(torch.load(path, weights_only=False)["binding"]["source_component"], 99)

    def test_global_rng_unchanged_by_prefetchable_input_creation(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            self.assertIs(f.provider.global_rng_free, True)
            python_state, numpy_state = random.getstate(), np.random.get_state()
            torch_state = torch.random.get_rng_state().clone()
            f.provider.batch([0], "native", 7, True)
            self.assertEqual(random.getstate(), python_state)
            now = np.random.get_state()
            self.assertEqual(now[0], numpy_state[0])
            np.testing.assert_array_equal(now[1], numpy_state[1])
            self.assertEqual(now[2:], numpy_state[2:])
            torch.testing.assert_close(torch.random.get_rng_state(), torch_state)

    def test_required_m10_scope_fails_without_activated_original_source(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            # Call the actual runtime guard, not this fixture's mocked operators.
            from hiercp_v1x import bounded_scope
            with patch.object(bounded_scope, "_ACTIVE", None):
                with self.assertRaisesRegex(ValueError, "Already activated"):
                    UBridgeData._runtime(f.provider)

    def test_original_core_verification_allows_PyG_generated_modules_but_rejects_wrong_core(self):
        from hiercp_v1x import bounded_scope
        from hiercp_v1x.contracts import V1_ARCHIVE_SHA256
        with unit_directory() as directory:
            root = Path(directory)
            source = root / "UNIT_original_source"
            files, modules = {}, {}
            archive = bounded_scope.ROOT / "versions/v1/pipeline_v1_source.zip"
            with ZipFile(archive) as bundle:
                for name in bundle.namelist():
                    if name.startswith("hiercp/") and name.endswith(".py"):
                        target = source / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        content = bundle.read(name)
                        target.write_bytes(content)
                        files[name] = hashlib.sha256(content).hexdigest()
                        module_name = name[:-3].replace("/", ".")
                        if module_name.endswith(".__init__"):
                            module_name = module_name[:-len(".__init__")]
                        modules[module_name] = ModuleType(module_name)
                        modules[module_name].__file__ = str(target)
            generated_name = "hiercp.model_CompatibilityGatedGATv2Conv_propagate"
            generated = ModuleType(generated_name)
            generated.__file__ = str(root / "UNIT_PyG_generated_propagate.py")
            modules[generated_name] = generated
            scope = dict(source_archive_sha256=V1_ARCHIVE_SHA256, original_module_sha256=files)
            with patch.dict(sys.modules, modules):
                self.assertEqual(_verify_original_core(source, scope), files)
                # Loaded core must point to its exact original file even when
                # generated aliases are present and have external temp paths.
                modules["hiercp.model"].__file__ = generated.__file__
                with self.assertRaisesRegex(ValueError, "core implementation is imported"):
                    _verify_original_core(source, scope)
                modules["hiercp.model"].__file__ = str(source / "hiercp/model.py")
                (source / "hiercp/model.py").write_bytes(b"UNIT deliberate source corruption")
                with self.assertRaisesRegex(ValueError, "byte-exact original core"):
                    _verify_original_core(source, scope)

    def test_resident_budget_evicts_cache_not_samples_or_candidates(self):
        with unit_directory() as directory:
            f = UnitFixture(directory, resident=200)
            result = f.provider.batch([0], "native", 1, True)
            self.assertEqual(result.counts, (8,))
            self.assertLessEqual(f.provider.report()["resident_bytes"], 200)
            self.assertEqual(len(f.build_calls), 8)
            tensor = torch.ones(5)
            self.assertEqual(_resident_size([tensor, tensor]), tensor.untyped_storage().nbytes())

    def test_unknown_or_duplicate_centers_and_invalid_arm_epoch_fail(self):
        with unit_directory() as directory:
            f = UnitFixture(directory)
            for values in (("U:0", "P"), ("P", "P"), ("P", "U:128"), ("P", (9, 9, 9))):
                with self.assertRaises(ValueError):
                    f.provider.sample(0, values, 0, False)
            with self.assertRaises(ValueError):
                f.provider.candidate_keys(0, "native", 0)
            with self.assertRaises(ValueError):
                f.provider.candidate_keys(0, "other", 1)
            with self.assertRaises(ValueError):
                f.provider.batch([0, 0], "selected", 1, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
