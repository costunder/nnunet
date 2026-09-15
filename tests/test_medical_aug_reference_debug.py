"""DEBUG small arrays: archive-source CP predicates, never medical metrics.

Only selected pure helper functions and the paste kernel are compiled from the
preserved reference text. Its loader, batch driver and NIfTI save code are never
executed. The original uint8-inversion defect is demonstrated, not adopted.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import ndimage as ndi

from hiercp.common import build_candidate_pool, choose_source_tumor, paste_source


class MedicalAugReferenceDebugTests(unittest.TestCase):
    def reference(self):
        path = Path(__file__).resolve().parents[1] / "reference/medical_data_aug/3d_copy_paste_tumor.py.txt"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {"LIVER_LABEL", "TUMOR_LABEL", "NUM_COPIES", "BLEND_BORDER",
                 "INTENSITY_SCALE_RANGE", "INTENSITY_SHIFT_RANGE", "MIN_LIVER_COVERAGE",
                 "AVOID_OVERLAP_WITH_ORIG", "OCCUPIED_CLEARANCE_VOX", "MIN_CENTER_SEPARATION_VOX",
                 "RNG_SEED"}
        literals = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names:
                        self.assertNotIn(target.id, literals)
                        literals[target.id] = ast.literal_eval(node.value)
        self.assertEqual(set(literals), names)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        namespace = {"np": np, "ndi": ndi}
        pure_helpers = ast.Module(body=[functions[name] for name in ("bbox_of_mask", "feather_alpha")],
                                  type_ignores=[])
        exec(compile(pure_helpers, str(path), "exec"), namespace)
        return path, functions, literals, namespace

    def fixture(self):
        shape = (35, 11, 11)
        image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        label = np.ones(shape, dtype=np.uint8)
        label[17, 5, 5] = 2
        case = SimpleNamespace(image=image, label=label, shape=shape, spacing=(3.0, 0.7, 0.7))
        source, _, _ = choose_source_tumor(image, label, tumor_label=2,
                                          rng=np.random.default_rng(42), selection="random", pad=2)
        return case, source

    def pool(self, case, source, **overrides):
        kwargs = dict(placement_mask=case.label == 1, full_organ_mask=case.label > 0,
                      occupied_mask=case.label == 2, organ_distance=np.ones(case.shape),
                      rng=np.random.default_rng(42), num_candidates=int(np.prod(case.shape)),
                      max_draws=0, force_exhaustive=True, min_liver_coverage=0.85,
                      occupied_clearance_vox=2, min_center_separation_mm=0.0,
                      min_center_separation_vox=12.0)
        kwargs.update(overrides)
        return build_candidate_pool(case, source, **kwargs)

    def test_shared_default_matches_reference_geometry_without_model_reduction(self):
        _, functions, original, _ = self.reference()
        cfg = json.loads((Path(__file__).resolve().parents[1] / "config/train.json").read_text())
        bbox = functions["bbox_of_mask"]
        defaults = dict(zip([arg.arg for arg in bbox.args.args][-len(bbox.args.defaults):],
                            map(ast.literal_eval, bbox.args.defaults)))
        calls = [node for node in ast.walk(functions["augment_case"]) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "bbox_of_mask"]
        self.assertEqual(len(calls), 1)
        call_pad = next(ast.literal_eval(item.value) for item in calls[0].keywords if item.arg == "pad")
        self.assertEqual(call_pad, defaults["pad"])
        self.assertEqual(cfg["labels"], {"liver": original["LIVER_LABEL"], "tumor": original["TUMOR_LABEL"]})
        for section in ("cache", "generation"):
            self.assertEqual(cfg[section]["source_pad"], call_pad)
            self.assertEqual(cfg[section]["occupied_clearance_vox"], original["OCCUPIED_CLEARANCE_VOX"])
            self.assertEqual(cfg[section]["min_liver_coverage"], original["MIN_LIVER_COVERAGE"])
            self.assertEqual(cfg[section]["min_center_separation_mm"], 0.0)
            self.assertEqual(cfg[section]["min_center_separation_vox"], original["MIN_CENTER_SEPARATION_VOX"])
            self.assertEqual(cfg[section]["no_placement_policy"], "retain_original")
        self.assertEqual(cfg["generation"]["num_copies"], original["NUM_COPIES"])
        self.assertEqual(cfg["generation"]["blend_border"], original["BLEND_BORDER"])
        self.assertEqual(tuple(cfg["generation"]["intensity_scale_range"]), original["INTENSITY_SCALE_RANGE"])
        self.assertEqual(tuple(cfg["generation"]["intensity_shift_range"]), original["INTENSITY_SHIFT_RANGE"])
        self.assertIs(original["AVOID_OVERLAP_WITH_ORIG"], True)
        # The production experiment intentionally has a recorded deterministic
        # seed and an extended candidate search; neither is claimed identical to
        # the original batch's unseeded, 4000-proposal sampler.
        self.assertIsNone(original["RNG_SEED"])
        self.assertEqual(cfg["cache"]["total_candidates"], 8)
        self.assertEqual(cfg["cache"]["candidate_pool_size"], 128)

    def test_hard_and_feather_paste_match_original_helper_and_kernel(self):
        path, functions, original, helpers = self.reference()
        loops = [node for node in ast.walk(functions["augment_case"])
                 if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "center"]
        self.assertEqual(len(loops), 1)
        body = loops[0].body
        start = next(index for index, node in enumerate(body) if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "scale")
        end = next(index for index, node in enumerate(body) if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0], ast.Subscript)
                   and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == "out_lab")
        occupied_updates = [node for node in body if isinstance(node, ast.AugAssign)
                            and isinstance(node.target, ast.Name) and node.target.id == "occupied_mask"]
        self.assertEqual(len(occupied_updates), 1)
        kernel = compile(ast.Module(body=body[start:end + 1] + occupied_updates, type_ignores=[]), str(path), "exec")
        shape = (19, 19, 19)
        image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * np.float32(0.25) - np.float32(500)
        label = np.ones(shape, dtype=np.int16)
        label[8:11, 8:11, 8:11] = original["TUMOR_LABEL"]
        image_before, label_before = image.copy(), label.copy()
        source, _, _ = choose_source_tumor(image, label, tumor_label=original["TUMOR_LABEL"],
                                         rng=np.random.default_rng(42), selection="random", pad=2)
        reference_slices = helpers["bbox_of_mask"](label == original["TUMOR_LABEL"], pad=2)
        self.assertEqual(source.patch_slices, reference_slices)
        np.testing.assert_array_equal(source.patch_image, image[reference_slices])
        np.testing.assert_array_equal(source.patch_mask, (label == original["TUMOR_LABEL"])[reference_slices])
        candidate = SimpleNamespace(slices=(slice(1, 8), slice(2, 9), slice(3, 10)))
        changed = np.zeros(shape, dtype=bool)
        changed[candidate.slices] = source.patch_mask
        for blend in (0, 1, 3):
            with self.subTest(blend_border=blend):
                out_image, out_label = image.copy(), label.copy()
                occupied = label == original["TUMOR_LABEL"]
                scale, shift = paste_source(out_image, out_label, occupied, source, candidate,
                    tumor_label=original["TUMOR_LABEL"], rng=np.random.default_rng(730),
                    intensity_scale_range=original["INTENSITY_SCALE_RANGE"],
                    intensity_shift_range=original["INTENSITY_SHIFT_RANGE"], blend_border=blend)
                alpha = helpers["feather_alpha"](source.patch_mask, blend)
                if blend == 3:
                    self.assertTrue(np.any((alpha > 0) & (alpha < 1)))
                namespace = {"np": np, "rng": np.random.default_rng(730),
                    "INTENSITY_SCALE_RANGE": original["INTENSITY_SCALE_RANGE"],
                    "INTENSITY_SHIFT_RANGE": original["INTENSITY_SHIFT_RANGE"],
                    "TUMOR_LABEL": original["TUMOR_LABEL"], "tumor_ct_patch": source.patch_image,
                    "tumor_mask_patch": source.patch_mask, "alpha_patch": alpha,
                    "out_img": image.copy(), "out_lab": label.copy(),
                    "occupied_mask": label == original["TUMOR_LABEL"],
                    "z1": 1, "z2": 8, "y1": 2, "y2": 9, "x1": 3, "x2": 10}
                exec(kernel, namespace)
                self.assertEqual((scale, shift), (namespace["scale"], namespace["shift"]))
                np.testing.assert_array_equal(out_image, namespace["out_img"])
                np.testing.assert_array_equal(out_label, namespace["out_lab"])
                np.testing.assert_array_equal(occupied, namespace["occupied_mask"])
                np.testing.assert_array_equal(out_image[~changed], image[~changed])
                np.testing.assert_array_equal(out_label[~changed], label[~changed])
        np.testing.assert_array_equal(image, image_before)
        np.testing.assert_array_equal(label, label_before)

    def test_original_uint8_inversion_is_not_boolean_distance_semantics(self):
        path, functions, original, _ = self.reference()
        assignments = [node for node in ast.walk(functions["augment_case"]) if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "dist_from_occupied"]
        self.assertEqual(len(assignments), 2)
        self.assertEqual(ast.dump(assignments[0].value), ast.dump(assignments[1].value))
        expression = compile(ast.Expression(body=assignments[0].value), str(path), "eval")
        occupied = np.zeros((35, 11, 11), dtype=np.uint8)
        occupied[17, 5, 5] = 1
        np.testing.assert_array_equal(np.unique(~occupied), np.array([254, 255], dtype=np.uint8))
        legacy = eval(expression, {"ndi": ndi, "occupied_mask": occupied})
        corrected = eval(expression, {"ndi": ndi, "occupied_mask": occupied.astype(bool)})
        probe = (17, 5, 8)
        self.assertGreater(legacy[17, 5, 5], 0)
        self.assertEqual(corrected[17, 5, 5], 0)
        self.assertGreaterEqual(legacy[probe], original["MIN_CENTER_SEPARATION_VOX"])
        self.assertEqual(corrected[probe], 3)
        self.assertLess(corrected[probe], original["MIN_CENTER_SEPARATION_VOX"])
        case, source = self.fixture()
        candidates, physical_distance = self.pool(case, source)
        self.assertEqual(physical_distance[17, 5, 5], 0)
        self.assertNotIn(probe, {candidate.center for candidate in candidates})

    def test_native_voxel_reference_predicate_and_physical_gnn_features(self):
        case, source = self.fixture()
        self.assertEqual(source.patch_mask.shape, (5, 5, 5))
        candidates, physical_distance = self.pool(case, source)
        occupied = case.label == 2
        forbidden = ndi.binary_dilation(occupied, structure=ndi.generate_binary_structure(3, 1), iterations=2)
        voxel_distance = ndi.distance_transform_edt(~occupied)
        expected = set()
        # Original mask/coverage/distance predicates; exhaustive extension also
        # visits the last fully in-bounds patch (legacy random proposals do not).
        for center in np.ndindex(case.shape):
            if any(v < 2 or v + 3 > n for v, n in zip(center, case.shape)):
                continue
            slc = tuple(slice(v - 2, v + 3) for v in center)
            if (case.label[center] == 1 and voxel_distance[center] >= 12 and
                    not np.any(forbidden[slc] & source.patch_mask) and
                    np.mean((case.label[slc] == 1)[source.patch_mask]) >= 0.85):
                expected.add(center)
        self.assertTrue(expected)
        self.assertEqual({candidate.center for candidate in candidates}, expected)
        for candidate in candidates:
            self.assertAlmostEqual(candidate.occupied_distance_mm, physical_distance[candidate.center])
        physical_candidates, _ = self.pool(case, source, min_center_separation_vox=0.0,
                                           min_center_separation_mm=12.0)
        self.assertGreater(len(physical_candidates), len(candidates))

    def test_random_and_exhaustive_share_voxel_predicate(self):
        case, source = self.fixture()
        candidates, _ = self.pool(case, source, num_candidates=128, max_draws=50000,
                                  force_exhaustive=False)
        distance = ndi.distance_transform_edt(case.label != 2)
        self.assertEqual(len(candidates), 128)
        self.assertTrue(all(distance[c.center] >= 12.0 for c in candidates))

    def test_hard_paste_preserves_every_non_tumor_voxel_and_source(self):
        case, source = self.fixture()
        candidates, _ = self.pool(case, source, num_candidates=1)
        candidate = candidates[0]
        out_image, out_label = case.image.copy(), case.label.copy()
        occupied = case.label == 2
        scale, shift = paste_source(out_image, out_label, occupied, source, candidate,
                                    tumor_label=2, rng=np.random.default_rng(3),
                                    intensity_scale_range=(0.95, 1.05),
                                    intensity_shift_range=(-5.0, 5.0), blend_border=0)
        changed = np.zeros(case.shape, dtype=bool)
        changed[candidate.slices] = source.patch_mask
        np.testing.assert_array_equal(out_image[~changed], case.image[~changed])
        np.testing.assert_array_equal(out_label[~changed], case.label[~changed])
        self.assertTrue(np.all(out_label[changed] == 2))
        np.testing.assert_allclose(out_image[changed], (source.patch_image * scale + shift)[source.patch_mask])
        self.assertEqual(case.label[17, 5, 5], 2)

    def test_conflicting_distance_units_are_rejected(self):
        case, source = self.fixture()
        with self.assertRaisesRegex(ValueError, "one center separation unit"):
            self.pool(case, source, min_center_separation_mm=12.0)
