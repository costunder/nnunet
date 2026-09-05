"""Independent DEBUG geometry regressions, not medical/model validation.

The small masks and point chains below are analytic fixtures confined to tests.
Production configuration is never changed. Missing numerical/graph runtime
dependencies explicitly skip the affected tests, never replace their behavior.
"""
from __future__ import annotations

import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_PYG = importlib.util.find_spec("torch_geometric") is not None
HAS_GEOMETRY = HAS_NUMPY and HAS_SCIPY
HAS_SAMPLING = HAS_GEOMETRY and HAS_TORCH and HAS_PYG

if HAS_GEOMETRY:
    import numpy as np
    from scipy import ndimage as ndi
    from hiercp import spatial

if HAS_SAMPLING:
    from hiercp import sample as sampling


def debug_config(**overrides):
    """Explicit isolated small-geometry profile; never a production default."""
    values = dict(
        patch_size=12,
        context_radius_mm=4.0,
        context_inner_radius_mm=1.0,
        context_outer_radius_mm=4.0,
        context_liver_surface_separation_mm=0.5,
        boundary_depth_mm=1.0,
        adaptive_roi_margin_mm=6.0,
        adaptive_roi_max_radius_mm=30.0,
        adaptive_roi_max_voxels=1_000_000,
        liver_anchor_search_mm=30.0,
        canonical_surface_spacing_mm=1.0,
        canonical_interior_spacing_mm=1.0,
        canonical_context_spacing_mm=1.0,
        canonical_liver_spacing_mm=1.0,
        canonical_node_limit=100_000,
        sample_context_nodes=1,
        sample_hops=2,
        sample_interface_radius_mm=0.1,
        sample_hop_radius_mm=1.01,
        context_radial_bins=4,
        context_azimuth_bins=8,
        context_elevation_bins=4,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@unittest.skipUnless(HAS_GEOMETRY, "DEBUG geometry requires actual NumPy and SciPy")
class PhysicalFootprintDebugTests(unittest.TestCase):
    def setUp(self):
        self.config = debug_config()
        self.quarter_turn = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )

    def test_identity_preserves_full_even_mask_and_does_not_alias_input(self):
        mask = np.zeros((6, 8, 4), dtype=bool)
        mask[0:5, 1:7, 0:3] = True
        before = mask.copy()
        result = spatial.transform_footprint_physical(
            mask, np.eye(3), (2.0, 1.0, 0.7), self.config
        )
        np.testing.assert_array_equal(result, before)
        self.assertFalse(np.shares_memory(result, mask))
        np.testing.assert_array_equal(mask, before)

    def test_even_shape_rotation_uses_integer_paste_anchor(self):
        mask = np.zeros((6, 8, 4), dtype=bool)
        mask[3, 4, 2] = True
        result = spatial.transform_footprint_physical(
            mask, self.quarter_turn, (1.0, 1.0, 1.0), self.config
        )
        coordinates = np.argwhere(result) - np.asarray(result.shape) // 2
        np.testing.assert_array_equal(coordinates, [[0, 0, 0]])

    def test_anisotropic_rotation_places_foreground_at_physical_coordinate(self):
        mask = np.zeros((7, 7, 7), dtype=bool)
        mask[4, 3, 3] = True  # +2 mm along first axis, not +1 mm.
        result = spatial.transform_footprint_physical(
            mask, self.quarter_turn, (2.0, 1.0, 1.0), self.config
        )
        anchor = np.asarray(result.shape) // 2
        self.assertTrue(result[tuple(anchor + np.asarray([0, 0, -2]))])
        self.assertTrue(np.all(np.asarray(result.shape) % 2 == 1))

    def test_rotated_long_tumor_is_not_cropped_to_source_box(self):
        mask = np.zeros((6, 6, 16), dtype=bool)
        mask[2:4, 2:4, 1:15] = True
        result = spatial.transform_footprint_physical(
            mask, self.quarter_turn, (1.0, 1.0, 1.0), self.config
        )
        self.assertEqual(int(result.sum()), int(mask.sum()))
        self.assertGreater(result.shape[0], mask.shape[0])
        original_relative = np.argwhere(mask) - np.asarray(mask.shape) // 2
        expected = {tuple(row) for row in (original_relative @ self.quarter_turn.T).astype(int)}
        actual = {tuple(row) for row in np.argwhere(result) - np.asarray(result.shape) // 2}
        self.assertEqual(actual, expected)

    def test_expansion_has_foreground_outside_original_box(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        result = spatial.transform_footprint_physical(
            mask, np.diag([2.0, 2.0, 2.0]), (1.0, 1.0, 1.0), self.config
        )
        coordinates = np.argwhere(result) - np.asarray(result.shape) // 2
        self.assertGreater(int(np.max(np.abs(coordinates))), 2)
        self.assertGreater(int(result.sum()), int(mask.size))

    def test_empty_rasterization_fails_instead_of_returning_identity(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[0, 0, 0] = True
        with self.assertRaisesRegex(spatial.CanonicalGraphUnavailable, "empty footprint"):
            spatial.transform_footprint_physical(
                mask, np.diag([0.001, 0.001, 0.001]), (1.0, 1.0, 1.0), self.config
            )

    def test_bad_transform_and_spacing_fail_explicitly(self):
        mask = np.ones((3, 3, 3), dtype=bool)
        for matrix in (np.zeros((3, 3)), np.full((3, 3), np.nan), np.ones((2, 2))):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                spatial.transform_footprint_physical(mask, matrix, (1, 1, 1), self.config)
        for spacing in ((0, 1, 1), (-1, 1, 1), (np.inf, 1, 1), (1, 1)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                spatial.transform_footprint_physical(mask, np.eye(3), spacing, self.config)

    def test_transform_budget_is_checked_before_dense_resampling(self):
        mask = np.ones((5, 5, 5), dtype=bool)
        with patch.object(spatial.ndi, "affine_transform", side_effect=AssertionError("allocated")):
            with self.assertRaises(spatial.AdaptiveRoiBudgetError):
                spatial.transform_footprint_physical(
                    mask, np.diag([10.0, 10.0, 10.0]), (1, 1, 1),
                    debug_config(adaptive_roi_max_voxels=1000),
                )


@unittest.skipUnless(HAS_GEOMETRY, "DEBUG liver geometry requires actual NumPy and SciPy")
class ActualLiverSurfaceDebugTests(unittest.TestCase):
    def setUp(self):
        self.organ = np.zeros((31, 31, 31), dtype=bool)
        self.organ[1:-1, 1:-1, 1:-1] = True
        self.depth = ndi.distance_transform_edt(self.organ).astype(np.float32)
        self.footprint = np.ones((3, 3, 3), dtype=bool)
        self.center = (15, 15, 15)

    def _expand(self, config):
        return spatial._expanded_surface_shape(
            center=self.center, footprint=self.footprint,
            full_organ=self.organ, organ_depth=self.depth,
            spacing=(1, 1, 1), config=config, native_shape=(7, 7, 7),
        )

    def test_expansion_keeps_context_and_includes_actual_band(self):
        organ_before, depth_before = self.organ.copy(), self.depth.copy()
        shape = self._expand(debug_config())
        self.assertEqual(shape, (29, 7, 7))
        organ = spatial.extract_centered(self.organ, self.center, shape, pad_value=False)
        depth = spatial.extract_centered(self.depth, self.center, shape, pad_value=0.0)
        footprint = spatial.center_crop_or_pad(self.footprint, shape)
        self.assertEqual(int(footprint.sum()), int(self.footprint.sum()))
        self.assertTrue(np.any(organ & ~footprint & (depth > 0) & (depth <= 1)))
        np.testing.assert_array_equal(self.organ, organ_before)
        np.testing.assert_array_equal(self.depth, depth_before)

    def test_surface_outside_radius_fails_without_internal_replacement(self):
        with self.assertRaisesRegex(spatial.CanonicalGraphUnavailable, "No actual liver-surface"):
            self._expand(debug_config(adaptive_roi_max_radius_mm=5, liver_anchor_search_mm=5))

    def test_required_surface_expansion_over_budget_fails(self):
        with self.assertRaisesRegex(spatial.AdaptiveRoiBudgetError, "real liver-surface"):
            self._expand(debug_config(adaptive_roi_max_voxels=1000))

    def _interior_only_fields(self):
        footprint = np.zeros((17, 17, 17), dtype=bool)
        footprint[7:10, 7:10, 7:10] = True
        return SimpleNamespace(
            footprint=footprint,
            organ=np.ones_like(footprint),
            outside_tumor_mm=ndi.distance_transform_edt(~footprint),
            liver_depth_mm=np.full(footprint.shape, 10.0),
        )

    def test_canonical_builder_rejects_interior_only_liver_surface(self):
        with self.assertRaises(spatial.EmptyCanonicalNodeError) as raised:
            spatial.canonical_coordinate_sets(self._interior_only_fields(), debug_config(), (1, 1, 1))
        self.assertEqual(raised.exception.node_type, "liver_surface")

    def test_validator_rejects_forged_internal_liver_anchor(self):
        fields = self._interior_only_fields()
        coordinates = {
            "surface": np.asarray([[7, 8, 8]]),
            "interior": np.asarray([[8, 8, 8]]),
            "context": np.asarray([[5, 8, 8]]),
            "liver_surface": np.asarray([[0, 0, 0]]),
        }
        with self.assertRaisesRegex(ValueError, "actual liver-surface band"):
            spatial.validate_canonical_coordinates(fields, coordinates, debug_config())

    def test_full_patch_payload_preserves_mask_and_has_actual_surface(self):
        payload = spatial.build_patch_payload(
            image=np.zeros(self.organ.shape, dtype=np.float32),
            center=self.center, footprint=self.footprint, full_organ=self.organ,
            organ_depth=self.depth, spacing=(1, 1, 1), config=debug_config(),
            erase_target=False, ct_clip=(-200, 250),
        )
        self.assertEqual(int(payload["footprint"].sum()), int(self.footprint.sum()))
        self.assertEqual(payload["model_input"].shape, (5, 12, 12, 12))
        nodes = spatial.canonical_coordinate_sets(SimpleNamespace(**payload), debug_config(), (1, 1, 1))
        depth = payload["liver_depth_mm"][tuple(nodes["liver_surface"].T)]
        self.assertTrue(np.all((depth > 0) & (depth <= 1)))


@unittest.skipUnless(HAS_SAMPLING, "DEBUG sampling requires actual NumPy/SciPy/PyTorch/PyG")
class ContextClosureDebugTests(unittest.TestCase):
    def _chain(self, count=8):
        positions = np.zeros((count, 3), dtype=np.float32)
        positions[:, 0] = np.arange(count)
        return {"pos_mm": positions, "x": np.zeros((count, 16), dtype=np.float32)}

    def test_each_configured_hop_adds_the_next_chain_node(self):
        node = self._chain()
        for hops in (0, 1, 2, 3):
            with self.subTest(hops=hops):
                selected = sampling._select_context(
                    node, np.asarray([[0, 0, 0]], dtype=np.float32),
                    debug_config(sample_hops=hops), np.random.default_rng(7),
                )
                np.testing.assert_array_equal(selected, np.arange(hops + 1))

    def test_required_interface_and_closure_can_exceed_seed_budget(self):
        node = self._chain()
        selected = sampling._select_context(
            node, np.asarray([[0, 0, 0]], dtype=np.float32),
            debug_config(sample_interface_radius_mm=3.1, sample_hops=2),
            np.random.default_rng(7),
        )
        np.testing.assert_array_equal(selected, np.arange(6))

    def test_balancing_does_not_drop_required_nodes(self):
        node = self._chain()
        selected = sampling._balanced_indices(
            node["pos_mm"], node["x"], 2, np.random.default_rng(7),
            radial_bins=4, azimuth_bins=8, elevation_bins=4,
            required=np.asarray([0, 1, 2, 3, 4]),
        )
        np.testing.assert_array_equal(selected, np.arange(5))

    def test_induced_edges_keep_all_retained_endpoints(self):
        edge = np.asarray([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
        selected = np.arange(4)
        actual = sampling._induced_edge_index(
            edge, selected, selected, source_count=6, destination_count=6
        )
        np.testing.assert_array_equal(actual, [[0, 1, 2], [1, 2, 3]])

    def test_old_canonical_contract_is_rejected_before_graph_materialization(self):
        old = {"format": "canonical-full-v22", "nodes": {}}
        with self.assertRaisesRegex(ValueError, "incompatible geometry contract"):
            sampling.build_local_view(old, old, debug_config(), seed=7)


if __name__ == "__main__":
    unittest.main()
