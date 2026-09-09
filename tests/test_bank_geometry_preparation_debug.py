"""DEBUG-only analytic bank geometry parity; not medical/model validation.

The small synthetic images and graph profile below never alter production
configuration. The legacy oracle retains the pre-refactor materialisation
order and checks every canonical tensor/edge, not just aggregate counts.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi
import torch

from hiercp import local
from hiercp.common import CasePaths, LoadedCase, SourceTumor
from hiercp.curriculum import CandidateSpec
from hiercp.schema import GraphBuildConfig, LOCAL_EDGE_TYPES
from hiercp.spatial import (
    AdaptiveRoiBudgetError,
    CanonicalGraphUnavailable,
    EmptyCanonicalNodeError,
    LEVEL0_GEOMETRY_CONTRACT,
)


def bank_geometry_classifier():
    """Exercise the actual nested bank classifier without loading nnU-Net."""
    path = Path(__file__).resolve().parents[1] / "tools" / "online_cp_benchmark.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "is_unrepresentable_geometry"]
    if len(matches) != 1:
        raise AssertionError("DEBUG test requires the unique actual bank geometry classifier")
    namespace = {"AdaptiveRoiBudgetError": AdaptiveRoiBudgetError,
                 "CanonicalGraphUnavailable": CanonicalGraphUnavailable}
    isolated = ast.Module(body=[matches[0]], type_ignores=[])
    exec(compile(isolated, str(path), "exec"), namespace)
    return namespace["is_unrepresentable_geometry"]


def legacy_local_graph(case, source, spec, *, full_organ_mask, organ_depth,
                       config, rng, ct_clip, prepared_source):
    """Independent copy of the old target sequence, confined to DEBUG tests."""
    local._require_full_graph(config)
    prepared = prepared_source
    footprint, transform = local._transform_footprint(
        prepared.source_footprint, spec, spacing=case.spacing, config=config
    )
    fields = local._patch_fields(
        case, spec.center, footprint, full_organ_mask, organ_depth,
        config=config, erase_target=True, ct_clip=ct_clip,
    )
    coordinates = local.canonical_coordinate_sets(fields, config, case.spacing)
    nodes = local._pack_nodes(
        fields, local.target_node_specifications(coordinates), case, config,
        branch_flag=-1.0,
    )
    edges = local._canonical_edges(
        {**prepared.canonical_nodes, **nodes},
        [kind for kind in LOCAL_EDGE_TYPES if kind not in prepared.canonical_edges],
        config, transform=transform,
    )
    return local.BuiltLocalGraph(
        graph=None,
        source_patch=prepared.source_patch,
        target_patch=fields.model_input.astype(np.float32, copy=False),
        source_local={
            "format": "canonical-full-v22",
            "geometry_contract": LEVEL0_GEOMETRY_CONTRACT,
            "nodes": prepared.canonical_nodes,
            "edges": prepared.canonical_edges,
            "footprint_voxels": int(prepared.source_footprint.sum()),
            "counts": prepared.canonical_counts,
            "edge_counts": {kind: int(value.shape[1])
                            for kind, value in prepared.canonical_edges.items()},
        },
        target_local={
            "format": "canonical-full-v22",
            "geometry_contract": LEVEL0_GEOMETRY_CONTRACT,
            "nodes": nodes,
            "edges": edges,
            "transform": torch.from_numpy(transform.astype(np.float32)),
            "counts": {kind: int(value["x"].shape[0]) for kind, value in nodes.items()},
            "edge_counts": {kind: int(value.shape[1]) for kind, value in edges.items()},
        },
    )


class BankGeometryPreparationDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classify_geometry = staticmethod(bank_geometry_classifier())
        # Analytic DEBUG geometry only; all nodes/edges of this fixture are used.
        cls.config = GraphBuildConfig(
            patch_size=16, context_radius_mm=6.0, context_shells_mm=(2.0, 4.0, 6.0),
            context_inner_radius_mm=1.0, context_outer_radius_mm=6.0,
            boundary_depth_mm=1.0, context_liver_surface_separation_mm=0.5,
            adaptive_roi_margin_mm=8.0, adaptive_roi_max_radius_mm=24.0,
            liver_anchor_search_mm=24.0,
            canonical_surface_spacing_mm=1.5, canonical_interior_spacing_mm=1.5,
            canonical_context_spacing_mm=2.0, canonical_liver_spacing_mm=3.0,
            surface_edge_radius_mm=2.0, interior_edge_radius_mm=2.0,
            context_edge_radius_mm=3.0, interface_edge_radius_mm=3.0,
            cross_edge_radius_mm=3.0, liver_edge_radius_mm=3.0,
            correspondence_radius_mm=3.0,
        )
        cls.config.validate()
        shape = (25, 25, 25)
        grid = np.indices(shape, dtype=np.float32)
        image = (grid[0] * 2.0 - grid[1] + grid[2] * 0.25).astype(np.float32)
        cls.organ = np.zeros(shape, dtype=bool)
        cls.organ[3:22, 3:22, 3:22] = True
        cls.spacing = np.asarray([1.5, 1.0, 0.8], dtype=np.float32)
        cls.depth = ndi.distance_transform_edt(
            cls.organ, sampling=cls.spacing
        ).astype(np.float32)
        label = cls.organ.astype(np.uint8)
        source_slices = (slice(9, 12), slice(11, 14), slice(11, 14))
        label[source_slices] = 2
        cls.case = LoadedCase(
            paths=CasePaths("DEBUG_BANK_GEOMETRY", Path("DEBUG_IMAGE"), Path("DEBUG_LABEL")),
            image=image, label=label, image_affine=np.eye(4), label_affine=np.eye(4),
            image_header=None, label_header=None, spacing=cls.spacing,
            image_source_signature={"debug": True}, label_source_signature={"debug": True},
        )
        cls.source = SourceTumor(
            component_id=1, full_mask=label == 2,
            patch_mask=(label[source_slices] == 2), patch_image=image[source_slices],
            patch_slices=source_slices, anchor_center=(10, 12, 12),
            centroid=(10.0, 12.0, 12.0), voxel_count=27,
        )
        cls.spec = CandidateSpec(
            center=(15, 12, 12), difficulty=1, corruption=0, region_id=0,
            prototype_id=0, liver_coverage=1.0, border_distance_mm=6.0,
            occupied_distance_mm=5.0, context_mean_hu=20.0, context_std_hu=4.0,
        )
        cls.ct_clip = (-200.0, 250.0)
        cls.prepared = local.prepare_local_source(
            cls.case, cls.source, full_organ_mask=cls.organ, organ_depth=cls.depth,
            config=cls.config, rng=np.random.default_rng(17), ct_clip=cls.ct_clip,
        )

    def options(self, **changes):
        options = dict(full_organ_mask=self.organ, organ_depth=self.depth,
                       config=self.config, ct_clip=self.ct_clip,
                       prepared_source=self.prepared)
        options.update(changes)
        return options

    def assert_nested_exact(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, np.ndarray):
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_exact(actual[key], expected[key])
        else:
            self.assertEqual(actual, expected)

    def test_debug_full_canonical_payload_matches_legacy_for_identity_and_transform(self):
        rotation = tuple(np.asarray([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
                         .reshape(-1).tolist())
        specs = [self.spec, replace(self.spec, center=(5, 12, 12)),
                 replace(self.spec, rotation=rotation, scale_xyz=(1.25, 1.0, 0.9))]
        for spec in specs:
            with self.subTest(center=spec.center, rotation=spec.rotation):
                expected = legacy_local_graph(
                    self.case, self.source, spec, rng=np.random.default_rng(31), **self.options()
                )
                actual = local.build_local_graph(
                    self.case, self.source, spec, rng=np.random.default_rng(31), **self.options()
                )
                self.assert_nested_exact(vars(actual), vars(expected))
                self.assertTrue(any(value.numel() for value in actual.target_local["edges"].values()))
                local.validate_local_geometry(self.case, self.source, spec, **self.options())

    def test_debug_validation_skips_feature_packing_and_all_radius_edges(self):
        with patch.object(local, "_pack_nodes", side_effect=AssertionError("disposable node features")), \
             patch.object(local, "_canonical_edges", side_effect=AssertionError("disposable edges")), \
             patch.object(local, "prepare_local_source", side_effect=AssertionError("duplicate source")), \
             patch.object(local, "canonical_coordinate_sets", wraps=local.canonical_coordinate_sets) as coordinates:
            self.assertIsNone(local.validate_local_geometry(
                self.case, self.source, self.spec, **self.options()
            ))
        self.assertEqual(coordinates.call_count, 1)

    def test_debug_geometry_rejections_and_non_geometry_failures_match_legacy(self):
        examples = [
            (self.spec, self.options(full_organ_mask=np.zeros_like(self.organ),
                                     organ_depth=np.zeros_like(self.depth)), CanonicalGraphUnavailable),
            (self.spec, self.options(organ_depth=self.organ.astype(np.float32)), EmptyCanonicalNodeError),
            (self.spec, self.options(config=replace(self.config, adaptive_roi_max_voxels=8)), AdaptiveRoiBudgetError),
            (replace(self.spec, rotation=(0.0,) * 9), self.options(), ValueError),
        ]
        for spec, options, expected_type in examples:
            outcomes = []
            for builder in (legacy_local_graph, local.build_local_graph, local.validate_local_geometry):
                kwargs = dict(options)
                if builder is not local.validate_local_geometry:
                    kwargs["rng"] = np.random.default_rng(31)
                with self.assertRaises(expected_type) as caught:
                    builder(self.case, self.source, spec, **kwargs)
                error = caught.exception
                outcomes.append((type(error), str(error), self.classify_geometry(error)))
            self.assertEqual(outcomes[0], outcomes[1])
            self.assertEqual(outcomes[0], outcomes[2])

    def test_debug_validation_and_canonical_build_preserve_rng_and_inputs(self):
        before = copy.deepcopy(vars(self.prepared))
        image, label = self.case.image.copy(), self.case.label.copy()
        validation_rng, inference_rng = np.random.default_rng(11), np.random.default_rng(999)
        states = (copy.deepcopy(validation_rng.bit_generator.state),
                  copy.deepcopy(inference_rng.bit_generator.state))
        local.validate_local_geometry(self.case, self.source, self.spec, **self.options())
        left = local.build_local_graph(self.case, self.source, self.spec,
                                      rng=validation_rng, **self.options())
        right = local.build_local_graph(self.case, self.source, self.spec,
                                       rng=inference_rng, **self.options())
        self.assert_nested_exact(vars(left), vars(right))
        self.assertEqual(validation_rng.bit_generator.state, states[0])
        self.assertEqual(inference_rng.bit_generator.state, states[1])
        self.assert_nested_exact(vars(self.prepared), before)
        np.testing.assert_array_equal(self.case.image, image)
        np.testing.assert_array_equal(self.case.label, label)

    def test_debug_validator_propagates_unexpected_errors_and_requires_real_preparation(self):
        with patch.object(local, "_patch_fields", side_effect=MemoryError("DEBUG host allocation")), \
             self.assertRaisesRegex(MemoryError, "DEBUG host allocation"):
            local.validate_local_geometry(self.case, self.source, self.spec, **self.options())
        with self.assertRaisesRegex(TypeError, "PreparedLocalSource"):
            local.validate_local_geometry(self.case, self.source, self.spec,
                                          **self.options(prepared_source=None))


if __name__ == "__main__":
    unittest.main()
