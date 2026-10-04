"""CPU/static half-A UNIT checks; synthetic fixtures, no CT quality claim.

No CUDA context or CNN forward is used here. The separate real-data DEBUG
learning tool verifies the complete preserved-v1 loss/gradient/optimizer path.
"""
from __future__ import annotations

import unittest

import torch

from hiercp_v1x.half_a_model import (
    HalfALocalCNN, MARKER_KEY, PyramidMaps, checkpoint_marker, half_a_identity, install_half_a,
    half_a_spec, model_contract, organ_ct, require_half_a_state,
)


def unused_sampler(*args):
    raise AssertionError("Static UNIT must not execute dense CNN/sampling")


class HalfAStaticUnit(unittest.TestCase):
    def test_install_replaces_l0_preserves_upper_objects_values_and_rng(self):
        # CPU constructor/interface UNIT only. Archive/source hashes and actual
        # archived CT execution are verified by the separate worker activation.
        from hiercp.model import HierarchicalPyGPlacementModel
        native = HierarchicalPyGPlacementModel(hidden_dim=128, heads=4,
                     local_layers=3, patient_layers=2, prototype_layers=2, dense_batch_size=4)
        native.architecture_version += "|bounded_scope_" + "a" * 64
        old_parameters = {id(value) for value in native.local_encoder.parameters()}
        upper_objects = {name: getattr(native, name) for name in (
            "patient_encoder", "prototype_encoder", "patient_readout", "population_readout", "score_head")}
        upper_state = {name: value.clone() for name, value in native.state_dict().items()
                       if not name.startswith("local_encoder.")}
        rng = torch.get_rng_state().clone()
        self.assertIs(install_half_a(native), native)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, original in upper_objects.items():
            self.assertIs(getattr(native, name), original)
        for name, original in upper_state.items():
            self.assertTrue(torch.equal(native.state_dict()[name], original), name)
        self.assertFalse(old_parameters & {id(value) for value in native.parameters()})
        self.assertIsInstance(native.local_encoder, HalfALocalCNN)
        self.assertIn(MARKER_KEY, native.state_dict())
        require_half_a_state(native.state_dict())
        # Idempotence consumes no RNG and cannot reinstall/change fresh weights.
        local = native.local_encoder
        self.assertIs(install_half_a(native).local_encoder, local)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_exact_width_depth_and_no_old_graph_parameters(self):
        with torch.device("meta"):
            local = HalfALocalCNN(sampler=unused_sampler, dropout=.1)
        self.assertEqual([len(level) for level in local.cnn.levels], [2, 3, 3])
        self.assertEqual([level[-1].conv.out_channels for level in local.cnn.levels], [12, 24, 32])
        self.assertEqual(local.project[0].weight.shape, (128, 68))
        self.assertEqual(local.fuse[0].weight.shape, (256, 512))
        self.assertEqual(local.fuse[-1].weight.shape, (128, 256))
        self.assertEqual(local.dense_batch_size, 4)
        self.assertEqual({name.split(".")[0] for name, _ in local.named_parameters()},
                         {"cnn", "project", "fuse"})

    def test_contract_identifies_dense_roi_bridge_and_native_difference(self):
        contract = model_contract()
        self.assertFalse(contract["native_spacing_v22_equivalence"])
        self.assertEqual(contract["training_candidate_count"], 8)
        self.assertEqual(contract["candidate_pool_size"], 128)
        self.assertEqual(contract["upper_heads"], 4)
        self.assertEqual((contract["patient_layers"], contract["prototype_layers"]), (2, 2))
        self.assertEqual(half_a_identity(), half_a_identity())

    def test_marker_rejects_missing_wrong_or_mutated_checkpoint(self):
        marker = checkpoint_marker()
        require_half_a_state({"_half_a_contract_digest": marker})
        require_half_a_state({"prefix._half_a_contract_digest": marker}, prefix="prefix.")
        for state in ({}, {"_half_a_contract_digest": marker.float()},
                      {"_half_a_contract_digest": marker.roll(1)}):
            with self.assertRaises(RuntimeError):
                require_half_a_state(state)

    def test_local_extra_state_rejects_other_bridge(self):
        with torch.device("meta"):
            local = HalfALocalCNN(sampler=unused_sampler, dropout=.1)
        local.set_extra_state(local.get_extra_state())
        altered = local.get_extra_state()
        altered["native_spacing_v22_equivalence"] = True
        with self.assertRaises(RuntimeError):
            local.set_extra_state(altered)

    def test_ct_conversion_preserves_real_masks_and_masks_external_ct(self):
        # Synthetic UNIT uses the full preserved 48^3 payload shape. It does
        # not stand in for the actual source/target CT validation tool.
        patch = torch.zeros(2, 5, 48, 48, 48)
        patch[:, 0] = 17  # Arbitrary outside-organ CT cannot leak into CNN.
        patch[:, 2, 3:40, 4:41, 5:42] = 1
        patch[:, 0, 3:40, 4:41, 5:42] = -1
        patch[:, 0, 7:10, 8:11, 9:12] = 1
        patch[:, 1, 7:10, 8:11, 9:12] = 1
        image, organ, footprint = organ_ct(patch)
        self.assertTrue(torch.equal(organ, patch[:, 2:3].bool()))
        self.assertTrue(torch.equal(footprint, patch[:, 1:2].bool()))
        self.assertTrue(torch.equal(image[~organ], torch.zeros_like(image[~organ])))
        self.assertTrue(torch.equal(image[footprint], torch.ones_like(image[footprint])))
        patch[:, 1] = 0
        with self.assertRaisesRegex(ValueError, "footprint vanished"):
            organ_ct(patch)

    def test_dense_shape_and_chunk_controls_are_not_silently_changed(self):
        with self.assertRaises(ValueError):
            organ_ct(torch.zeros(1, 5, 24, 24, 24))
        with self.assertRaises(ValueError):
            HalfALocalCNN(sampler=unused_sampler, dropout=.1, dense_batch_size=1)

    def test_source_selection_shares_pyramid_storage_and_retains_duplicates(self):
        features = torch.arange(2 * 12 * 2**3, dtype=torch.float32).reshape(2, 12, 2, 2, 2)
        mask = torch.ones(2, 1, 2, 2, 2, dtype=torch.bool)
        means = features.flatten(2).mean(-1)
        maps = PyramidMaps(((features, mask),), mask, mask, means, means,
                           torch.arange(2, dtype=torch.long))
        selected = maps.index_select(0, torch.tensor([1, 0, 1]))
        self.assertEqual(selected.shape[0], 3)
        self.assertIs(selected.scales[0][0], features)
        self.assertTrue(torch.equal(selected.owners, torch.tensor([1, 0, 1])))
        self.assertTrue(torch.equal(selected.organ_mean[selected.owners], means[[1, 0, 1]]))

    def test_only_optional_real_surface_absence_has_an_empty_set_descriptor(self):
        values = torch.empty(0, 68)
        weights = torch.empty(0, 1)
        owners = torch.empty(0, dtype=torch.long)
        result = HalfALocalCNN._role_mean(values, weights, owners, 3,
                                          role="source_liver_surface", optional=True)
        self.assertEqual(result.shape, (3, 68))
        self.assertTrue(torch.equal(result, torch.zeros(3, 68)))
        with self.assertRaisesRegex(ValueError, "No dense organ/footprint support"):
            HalfALocalCNN._role_mean(values, weights, owners, 3, role="tumor_surface")
        with self.assertRaisesRegex(ValueError, "Present optional"):
            HalfALocalCNN._role_mean(torch.ones(1, 68), torch.zeros(1, 1),
                torch.tensor([0]), 3, role="source_liver_surface", optional=True)

    def test_optional_sampling_selects_present_maps_only_and_restores_order(self):
        calls = []
        def sampler(values, grid, owners):
            calls.append((values[:, 0, 0, 0, 0].clone(), owners.clone()))
            return values[:, :, 0, 0, 0][owners]
        local = HalfALocalCNN(sampler=sampler, dropout=.1)
        features = torch.stack((torch.full((12, 2, 2, 2), 3.), torch.full((12, 2, 2, 2), 9.)))
        mask = torch.ones(2, 1, 2, 2, 2, dtype=torch.bool)
        means = torch.zeros(2, 68)
        maps = PyramidMaps(((features, mask),), mask, mask, means, means,
                           torch.tensor([1, 0, 1]))
        # Only logical graph 0 owns an actual surface; source ROI 0 is unused.
        value = local._sample(maps, features, torch.zeros(2, 3), torch.tensor([0, 0]), optional=True)
        self.assertTrue(torch.equal(value, torch.full((2, 12), 9.)))
        self.assertTrue(torch.equal(calls[-1][0], torch.tensor([9.])))
        self.assertTrue(torch.equal(calls[-1][1], torch.tensor([0, 0])))
        value = local._sample(maps, features, torch.zeros(3, 3), torch.tensor([0, 1, 2]), optional=True)
        self.assertTrue(torch.equal(value[:, 0], torch.tensor([9., 3., 9.])))
        empty = local._sample(maps, features, torch.empty(0, 3),
                              torch.empty(0, dtype=torch.long), optional=True)
        self.assertEqual(empty.shape, (0, 12))

    def test_forward_admission_rejects_native_mixed_and_other_margin_graphs(self):
        local = HalfALocalCNN(sampler=unused_sampler, dropout=.1, scope_contract="a" * 64)
        valid = {"v1x_bounded_scope_contract": ["a" * 64, "a" * 64],
                 "v1x_bounded_scope_margin_mm": torch.tensor([10., 10.])}
        local._require_scope(valid, 2)
        for payload in ({}, {**valid, "v1x_bounded_scope_contract": ["a" * 64, "b" * 64]},
                        {**valid, "v1x_bounded_scope_margin_mm": torch.tensor([10., 20.])}):
            with self.assertRaises(ValueError):
                local._require_scope(payload, 2)

    def test_role_and_shell_means_follow_distinct_support_without_dead_tokens(self):
        values = torch.tensor([[2., 4.], [10., 12.], [50., 70.], [6., 8.]])
        weights = torch.tensor([[1.], [0.], [1.], [1.]])
        owner = torch.tensor([0, 0, 1, 1])
        means = HalfALocalCNN._role_mean(values, weights, owner, 2, role="source_context")
        self.assertTrue(torch.equal(means, torch.tensor([[2., 4.], [28., 39.]])))
        shells = HalfALocalCNN._role_mean(values[[0, 2, 3]], weights[[0, 2, 3]],
                    owner[[0, 2, 3]], 2, role="source_context", shell=torch.tensor([0, 1, 2]))
        self.assertTrue(torch.equal(shells[0, 0], values[0]))
        self.assertTrue(torch.equal(shells[1, 1], values[2]))
        self.assertTrue(torch.equal(shells[0, 1], torch.zeros(2)))
        with self.assertRaisesRegex(ValueError, "no dense organ support"):
            HalfALocalCNN._role_mean(values, weights, owner, 2,
                                   role="source_context", shell=torch.tensor([0, 1, 1, 2]))


if __name__ == "__main__":
    unittest.main()
