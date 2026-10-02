"""Synthetic CUDA UNIT geometry for organ-supported path relay sampling.

These are small, explicit geometric contract fixtures, not actual CT, learned
CP accuracy, full training, or an altered final sampling quota. The numerical
sampler always runs on CUDA; host lists below are assertion/audit data only.
"""
import copy
import math
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from l0_local_cnn.model import LocalBatch, LocalCNN, MODE
from l0_sparse_feature.coverage import all_pool_distances
from l0_sparse_feature.model import _select_context
from l0_sparse_feature.relational import (
    V1RelationalSparseProfile, _build_pair_edges, _dense_pair_adjacency,
)
from l0_sparse_feature.relayed import RelayedV1SparseL0, build_relay_pair_edges
from l0_sparse_feature.relay_sampling import retain_path_relays


def unit_scene(coordinates, seed_coordinates, *, spacing=(1., 1., 1.),
               anchor=(0., 0., 0.), padding=0):
    """Pack explicitly named real UNIT voxels plus ineligible NaN padding."""
    xyz = torch.tensor(coordinates, dtype=torch.float32, device='cuda')
    steps = torch.tensor(spacing, dtype=torch.float32, device='cuda')
    center = torch.tensor(anchor, dtype=torch.float32, device='cuda')
    distance = ((xyz - center) * steps).norm(dim=-1)
    roles = torch.where(distance <= 5., 1, torch.where(distance <= 10., 2, 3))
    lookup = {tuple(point): index for index, point in enumerate(coordinates)}
    seed_ids = torch.tensor([lookup[tuple(point)] for point in seed_coordinates],
        device='cuda', dtype=torch.long)
    eligible = torch.ones(len(xyz), device='cuda', dtype=torch.bool)
    if padding:
        xyz = torch.cat((xyz, xyz.new_full((padding, 3), float('nan'))))
        eligible = torch.cat((eligible, torch.zeros(padding, device='cuda', dtype=torch.bool)))
        roles = torch.cat((roles, torch.zeros(padding, device='cuda', dtype=torch.long)))
        distance = torch.cat((distance, distance.new_full((padding,), float('nan'))))
    return dict(pool_xyz_native=xyz[None], eligible=eligible[None],
        band=roles[None], distance_mm=distance[None], spacing_mm=steps[None],
        seed_index=seed_ids[None], seed_mask=torch.ones_like(seed_ids[None], dtype=torch.bool))


def selected_pool_ids(result, scene=0):
    return result['selection_index'][scene][result['selection_mask'][scene]]


def packed_pair_geometry(inputs, sampling):
    """Add abstract query slots and convert donor-first scenes to joint pairs."""
    selection = sampling['selection_index']
    xyz = inputs['pool_xyz_native'].gather(1, selection[..., None].expand(-1, -1, 3))
    physical = xyz * inputs['spacing_mm'][:, None]
    physical = torch.cat((physical.new_zeros((len(xyz), 1, 3)), physical), 1)
    mask = torch.cat((torch.ones((len(xyz), 1), device='cuda', dtype=torch.bool),
        sampling['selection_mask']), 1)
    pairs = len(xyz) // 2
    return (physical.reshape(2, pairs, physical.shape[1], 3).transpose(0, 1),
            mask.reshape(2, pairs, mask.shape[1]).transpose(0, 1))


def anisotropic_pair_fixture():
    points = [(x, y, z) for x in range(4) for y in range(2) for z in range(4)]
    seeds = [(x, 0, z) for x in range(4) for z in range(4)]
    scene = unit_scene(points, seeds, spacing=(1., 1., 5.))
    inputs = {key: value.repeat((4,) + (1,) * (value.ndim - 1))
        for key, value in scene.items()}
    sampling = retain_path_relays(**inputs)
    pair_xyz, pair_mask = packed_pair_geometry(inputs, sampling)
    return inputs, sampling, pair_xyz, pair_mask


def unit_model_fixture():
    config = dict(architecture=MODE, channels=[12, 24, 32],
        convolutions=[2, 3, 3], hidden_dim=128, margin_mm=10.,
        input='native_spacing_organ_only', readout='organ_masked_mean_each_scale',
        fusion='donor_target_difference_product', initialization='fresh_seed42',
        learning_policy='same_donor_live_v1')
    torch.manual_seed(5342)
    reference = LocalCNN(config).cuda().eval()
    images = torch.randn(3, 1, 21, 21, 21, device='cuda')
    organ = torch.ones_like(images, dtype=torch.bool)
    organ[:, :, :2] = False
    organ[:, :, 20:] = False
    organ[2, :, :, 19:] = False
    origins = [[17, 24, 9], [80, 2, 53], [11, 43, 102]]
    spacing = [[1., 1., 1.], [1., 1.5, 2.], [2., 1., 1.]]
    audit = [dict(case=f'UNIT relay crop {i}', origin=origins[i],
        shape=[21, 19 if i == 2 else 21, 21], spacing=spacing[i]) for i in range(3)]
    batch = LocalBatch(images, organ, torch.zeros(2, device='cuda', dtype=torch.long),
        torch.tensor([1, 2], device='cuda'), torch.arange(2, device='cuda'), audit).validate()
    kwargs = dict(donor_centers_native=torch.tensor(origins[0], device='cuda').float()[None].repeat(2, 1) + 10,
        recipient_centers_native=torch.tensor(origins[1:], device='cuda').float() + 10)
    return reference, V1RelationalSparseProfile(16, 3., 5., 10.), batch, kwargs


def rotated_role_coverage_fixture():
    """Known orthogonal UNIT features; each scene puts roles in other slots."""
    feature_pair = F.pad(torch.eye(2, device='cuda'), (0, 66))
    feature = feature_pair.repeat(3, 1)
    radii = torch.tensor([2., 4., 6., 9., 12., 16.], device='cuda')
    xyz = torch.stack((radii, torch.zeros_like(radii), torch.zeros_like(radii)), -1)
    # Query, selected context, and one explicit NaN padding slot per scene.
    selected = torch.tensor([[0, 2, 4], [3, 5, 1]], device='cuda')
    node_feature = torch.cat((feature.new_ones((2, 1, 68)), feature[selected],
        feature.new_full((2, 1, 68), float('nan'))), 1)
    node_xyz = torch.cat((xyz.new_zeros((2, 1, 3)), xyz[selected],
        xyz.new_full((2, 1, 3), float('inf'))), 1)
    return dict(coverage_pool=dict(features=feature[None].repeat(2, 1, 1),
        relative_mm=xyz[None].repeat(2, 1, 1),
        eligible=torch.ones((2, 6), device='cuda', dtype=torch.bool),
        band=torch.tensor([[1, 1, 2, 2, 3, 3]], device='cuda').repeat(2, 1)),
        features=node_feature, relative_xyz_mm=node_xyz,
        node_mask=torch.tensor([[True, True, True, True, False]], device='cuda').repeat(2, 1),
        role=torch.tensor([[0, 1, 2, 3, 1], [0, 2, 3, 1, 1]], device='cuda'),
        statistics=dict(band_feature_variance=torch.ones((2, 3), device='cuda'),
            band_spatial_variance_mm2=torch.tensor([[2., 4.5, 8.]], device='cuda').repeat(2, 1)))


class CUDARelaySampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('Relay numerical UNIT tests require CUDA; '
                'no CPU fallback or all-skip PASS')
        cls.flags = (torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    @classmethod
    def tearDownClass(cls):
        a, b, c, d, e = cls.flags
        torch.backends.cuda.matmul.allow_tf32 = a
        torch.backends.cudnn.allow_tf32 = b
        torch.backends.cudnn.benchmark = c
        torch.backends.cudnn.deterministic = d
        torch.use_deterministic_algorithms(e)

    def assert_real_paths(self, result, inputs):
        """Independently walk fine parents and audit every compressed edge."""
        selected = result['selection_index'].detach().cpu().tolist()
        parents = result['fine_parent'].detach().cpu().tolist()
        points = inputs['pool_xyz_native'].detach().cpu().tolist()
        spacing = inputs['spacing_mm'].detach().cpu().tolist()
        eligible = inputs['eligible'].detach().cpu().tolist()
        lengths = []
        steps = []

        def ancestry(scene, first, last):
            path = [first]
            for _ in range(len(parents[scene])):
                previous = parents[scene][path[-1]]
                if previous < 0:
                    return None
                self.assertNotIn(previous, path, 'Fine parent forest has a cycle')
                path.append(previous)
                if previous == last:
                    return path
            self.fail('Fine parent chain exceeds the finite UNIT pool')

        for scene, target_slot, source_slot in result['mandatory_edges'].detach().cpu().T.tolist():
            self.assertTrue(result['selection_mask'][scene, target_slot])
            self.assertTrue(result['selection_mask'][scene, source_slot])
            source, target = selected[scene][source_slot], selected[scene][target_slot]
            self.assertNotEqual(source, target)
            path = ancestry(scene, source, target)
            if path is None:
                path = ancestry(scene, target, source)
            self.assertIsNotNone(path, 'Compressed edge is not a real fine-forest path')
            length = 0.
            for first, last in zip(path, path[1:]):
                self.assertTrue(eligible[scene][first])
                self.assertTrue(eligible[scene][last])
                delta = [abs(a - b) for a, b in zip(points[scene][first], points[scene][last])]
                self.assertEqual(sum(delta), 1., 'A fine edge skipped a native voxel')
                self.assertEqual(sum(value != 0 for value in delta), 1,
                    'A fine edge used a diagonal neighbor')
                step_mm = math.sqrt(sum((value * scale) ** 2
                    for value, scale in zip(delta, spacing[scene])))
                self.assertLessEqual(step_mm, 6.)
                length += step_mm
            self.assertLessEqual(length, 6.00001, 'Compression exceeded cumulative 6 mm path length')
            lengths.append(length)
            steps.append(len(path) - 1)
        expected_length = torch.tensor(lengths, device='cuda', dtype=torch.float32)
        expected_steps = torch.tensor(steps, device='cuda', dtype=torch.long)
        self.assertTrue(torch.allclose(result['edge_path_length_mm'], expected_length,
            atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(result['edge_path_steps'], expected_steps))
        # Mandatory edges are reciprocal real paths, never fake joins.
        edge_set = {tuple(edge) for edge in result['mandatory_edges'].detach().cpu().T.tolist()}
        for scene, target, source in edge_set:
            self.assertIn((scene, source, target), edge_set)

    def assert_query_roots_real(self, result, inputs):
        scene, slot = result['query_roots']
        self.assertTrue(result['selection_mask'][scene, slot].all())
        pool = result['selection_index'][scene, slot]
        self.assertTrue(inputs['eligible'][scene, pool].all())
        self.assertTrue((inputs['distance_mm'][scene, pool] <= 3.).all())
        self.assertTrue(result['fine_reachable'][scene, pool].all())
        self.assertTrue((result['fine_parent'][scene, pool] == -1).all())
        self.assertTrue((result['fine_depth'][scene, pool] == 0).all())

    def test_all_original_seeds_and_necessary_relays_survive_without_final_cap(self):
        points = [(x, 0, 0) for x in range(25)]
        inputs = unit_scene(points, [(0, 0, 0), (24, 0, 0)])
        result = retain_path_relays(**inputs)
        selected = selected_pool_ids(result)
        self.assertTrue(torch.isin(inputs['seed_index'][0], selected).all())
        self.assertGreater(len(selected), inputs['seed_mask'][0].sum().item())
        self.assertEqual(len(selected.unique()), len(selected))
        self.assertTrue(inputs['eligible'][0, selected].all())
        original = result['selection_index'][0][result['seed_mask'][0] & result['selection_mask'][0]]
        self.assertTrue(torch.equal(original.sort().values, inputs['seed_index'][0].sort().values))
        self.assertTrue(result['fine_reachable'].all())
        self.assertTrue(result['diagnostics']['sampling_connected_per_scene'].all())
        self.assertGreater(result['mandatory_edges'].shape[1], 0)
        self.assert_real_paths(result, inputs)
        self.assert_query_roots_real(result, inputs)

    def test_anisotropic_z5_mandatory_step_survives_nearest_three_exclusion(self):
        points = [(x, y, z) for x in range(4) for y in range(2) for z in range(4)]
        # Three closer in-plane seed sources exist at each z level. A plain
        # nearest-three graph excludes vertical 5 mm edges between levels.
        seeds = [(x, 0, z) for x in range(4) for z in range(4)]
        inputs = unit_scene(points, seeds, spacing=(1., 1., 5.))
        result = retain_path_relays(**inputs)
        self.assertTrue(result['diagnostics']['sampling_connected_per_scene'].all())
        selected = selected_pool_ids(result)
        self.assertTrue(torch.isin(inputs['seed_index'][0], selected).all())
        self.assert_real_paths(result, inputs)
        self.assert_query_roots_real(result, inputs)
        xyz = inputs['pool_xyz_native'][0, selected]
        physical = xyz * inputs['spacing_mm'][0]
        distance = torch.cdist(physical, physical, compute_mode='donot_use_mm_for_euclid_dist')
        legal = (distance <= 6.) & ~torch.eye(len(selected), device='cuda', dtype=torch.bool)
        ids = distance.masked_fill(~legal, float('inf')).argsort(dim=-1, stable=True)[:, :3]
        nearest_three = torch.zeros_like(legal).scatter(-1, ids, True) & legal
        scene, target, source = result['mandatory_edges']
        self.assertTrue((scene == 0).all())
        edge_pool_target = result['selection_index'][scene, target]
        edge_pool_source = result['selection_index'][scene, source]
        delta = inputs['pool_xyz_native'][scene, edge_pool_target] - inputs['pool_xyz_native'][scene, edge_pool_source]
        vertical = (delta[:, :2] == 0).all(-1) & (delta[:, 2].abs() == 1)
        self.assertTrue(vertical.any(), 'No mandatory real 5 mm z step was retained')
        self.assertTrue((result['edge_path_length_mm'][vertical] == 5.).all())
        self.assertTrue((vertical & ~nearest_three[target, source]).any(),
            'Fixture must expose a real mandatory edge dropped by nearest-three')

    def test_concave_organ_hole_cannot_be_replaced_by_straight_line_shortcut(self):
        points = ([(0, y, 0) for y in range(7)] +
                  [(x, 6, 0) for x in range(1, 7)] +
                  [(6, y, 0) for y in range(5, -1, -1)])
        inputs = unit_scene(points, [(0, 0, 0), (6, 0, 0)])
        result = retain_path_relays(**inputs)
        self.assertTrue(result['fine_reachable'].all())
        self.assertGreater(len(selected_pool_ids(result)), 2)
        self.assert_real_paths(result, inputs)
        self.assert_query_roots_real(result, inputs)
        # The far seed is 6 mm from the anchor but lies 15 native organ steps
        # from the closest query-sphere root. It must use the U-shaped path.
        far_pool = inputs['seed_index'][0, 1]
        self.assertGreaterEqual(result['fine_depth'][0, far_pool].item(), 15)
        roots_pool = result['selection_index'][result['query_roots'][0], result['query_roots'][1]]
        self.assertFalse((roots_pool == far_pool).any())
        pair_edges = result['mandatory_edges']
        source_pool = result['selection_index'][pair_edges[0], pair_edges[2]]
        target_pool = result['selection_index'][pair_edges[0], pair_edges[1]]
        anchor_pool = inputs['seed_index'][0, 0]
        shortcut = ((source_pool == anchor_pool) & (target_pool == far_pool)) | (
            (target_pool == anchor_pool) & (source_pool == far_pool))
        self.assertFalse(shortcut.any())

    def test_truly_unreachable_component_keeps_seed_and_reports_incomplete(self):
        points = [(x, 0, 0) for x in range(13)] + [(x, 0, 0) for x in range(100, 113)]
        inputs = unit_scene(points, [(0, 0, 0), (12, 0, 0), (112, 0, 0)])
        result = retain_path_relays(**inputs)
        selected = selected_pool_ids(result)
        self.assertTrue(torch.isin(inputs['seed_index'][0], selected).all())
        self.assertTrue(result['fine_reachable'][0, :13].all())
        self.assertFalse(result['fine_reachable'][0, 13:].any())
        self.assertTrue((result['fine_depth'][0, 13:] == -1).all())
        self.assertFalse(result['diagnostics']['sampling_connected_per_scene'].any())
        unreachable = result['diagnostics']['unreachable_seed_indices']
        self.assertEqual(unreachable.shape, (2, 1))
        self.assertEqual(unreachable[0].tolist(), [0])
        self.assertEqual(unreachable[1].tolist(), [25])
        scene, target, source = result['mandatory_edges']
        self.assertTrue((result['selection_index'][scene, target] < 13).all())
        self.assertTrue((result['selection_index'][scene, source] < 13).all())
        self.assert_real_paths(result, inputs)
        self.assert_query_roots_real(result, inputs)

    def test_native_axis_step_greater_than_radius_is_unreachable_not_fabricated(self):
        inputs = unit_scene([(0, 0, z) for z in range(3)],
            [(0, 0, 0), (0, 0, 2)], spacing=(1., 1., 7.))
        result = retain_path_relays(**inputs)
        self.assertTrue(torch.isin(inputs['seed_index'][0], selected_pool_ids(result)).all())
        self.assertEqual(result['fine_reachable'].tolist(), [[True, False, False]])
        self.assertEqual(result['mandatory_edges'].shape[1], 0)
        self.assertFalse(result['diagnostics']['sampling_connected_per_scene'].any())
        self.assert_query_roots_real(result, inputs)

    def test_relay_roles_follow_actual_radial_native_positions(self):
        anchor = (17., 24., 9.)
        points = [(17 + x, 24, 9) for x in range(25)]
        inputs = unit_scene(points, [points[1], points[8], points[24]],
            spacing=(1., 2., 5.), anchor=anchor)
        result = retain_path_relays(**inputs)
        selected = selected_pool_ids(result)
        native = inputs['pool_xyz_native'][0, selected]
        center = native.new_tensor(anchor)
        radius = ((native - center) * inputs['spacing_mm'][0]).norm(dim=-1)
        expected = torch.where(radius <= 5., 1, torch.where(radius <= 10., 2, 3))
        roles = result['roles'][0][result['selection_mask'][0]]
        self.assertTrue(torch.equal(roles, expected))
        self.assertTrue(torch.equal(roles, inputs['band'][0, selected]))
        is_relay = ~result['seed_mask'][0][result['selection_mask'][0]]
        self.assertTrue(is_relay.any())
        self.assertTrue(((roles == 2) & is_relay).any())
        self.assertTrue(((roles == 3) & is_relay).any())
        self.assert_real_paths(result, inputs)

    def test_nan_padding_and_false_seed_slots_are_ignored_without_scene_leakage(self):
        first = unit_scene([(x, 0, 0) for x in range(25)],
            [(0, 0, 0), (24, 0, 0)])
        second = unit_scene([(x, 0, 0) for x in range(100, 113)],
            [(100, 0, 0), (112, 0, 0)], anchor=(100., 0., 0.), padding=12)
        inputs = {key: torch.cat((first[key], second[key]), 0) for key in first}
        inputs['seed_index'] = torch.cat((inputs['seed_index'],
            torch.full((2, 1), -999, device='cuda', dtype=torch.long)), 1)
        inputs['seed_mask'] = torch.cat((inputs['seed_mask'],
            torch.zeros((2, 1), device='cuda', dtype=torch.bool)), 1)
        result = retain_path_relays(**inputs)
        first_alone = retain_path_relays(**first)
        second_alone = retain_path_relays(**second)
        self.assertTrue(torch.equal(selected_pool_ids(result, 0), selected_pool_ids(first_alone)))
        self.assertTrue(torch.equal(selected_pool_ids(result, 1), selected_pool_ids(second_alone)))
        self.assertTrue(torch.equal(result['fine_parent'][0], first_alone['fine_parent'][0]))
        self.assertTrue(torch.equal(result['fine_parent'][1], second_alone['fine_parent'][0]))
        self.assertFalse(result['fine_reachable'][1, 13:].any())
        self.assertTrue((selected_pool_ids(result, 1) < 13).all())
        self.assert_real_paths(result, inputs)
        self.assert_query_roots_real(result, inputs)
        changed_padding = {key: value.clone() for key, value in inputs.items()}
        changed_padding['pool_xyz_native'][1, 13:] = float('inf')
        changed_padding['distance_mm'][1, 13:] = -float('inf')
        changed_padding['band'][1, 13:] = -99
        identical = retain_path_relays(**changed_padding)
        for key in ('selection_index', 'selection_mask', 'roles', 'seed_mask',
                    'mandatory_edges', 'edge_path_length_mm', 'edge_path_steps',
                    'query_roots', 'fine_parent', 'fine_reachable', 'fine_depth'):
            self.assertTrue(torch.equal(result[key], identical[key]), key)
        # Cutting only scene1's native path cannot alter scene0's graph or
        # parent forest, despite shared padded buffers and batched BFS waves.
        cut_second = {key: value.clone() for key, value in inputs.items()}
        cut_second['eligible'][1, 5:8] = False
        cut_result = retain_path_relays(**cut_second)
        self.assertTrue(torch.equal(selected_pool_ids(result, 0), selected_pool_ids(cut_result, 0)))
        self.assertTrue(torch.equal(result['fine_parent'][0], cut_result['fine_parent'][0]))
        self.assertTrue(torch.equal(result['fine_reachable'][0], cut_result['fine_reachable'][0]))
        before_edges = result['mandatory_edges'][:, result['mandatory_edges'][0] == 0]
        after_edges = cut_result['mandatory_edges'][:, cut_result['mandatory_edges'][0] == 0]
        self.assertTrue(torch.equal(before_edges, after_edges))
        self.assertTrue(cut_result['diagnostics']['sampling_connected_per_scene'][0])
        self.assertFalse(cut_result['diagnostics']['sampling_connected_per_scene'][1])
        self.assertTrue(torch.isin(inputs['seed_index'][1, :2], selected_pool_ids(cut_result, 1)).all())

    def test_invalid_native_coordinates_support_seeds_and_query_coverage_fail(self):
        original = unit_scene([(x, 0, 0) for x in range(13)],
            [(0, 0, 0), (12, 0, 0)], padding=2)
        cases = []

        def altered(name, key, index, value):
            inputs = {field: tensor.clone() for field, tensor in original.items()}
            inputs[key][index] = value
            cases.append((name, inputs))

        altered('nonfinite eligible coordinate', 'pool_xyz_native', (0, 1, 0), float('nan'))
        altered('fractional native coordinate', 'pool_xyz_native', (0, 1, 0), 1.5)
        altered('negative native coordinate', 'pool_xyz_native', (0, 1, 0), -1.)
        altered('duplicate eligible native voxel', 'pool_xyz_native', (0, 1),
            torch.tensor([0., 0., 0.], device='cuda'))
        altered('out of range seed', 'seed_index', (0, 1), 15)
        altered('negative seed', 'seed_index', (0, 1), -1)
        altered('unsupported original seed', 'seed_index', (0, 1), 13)
        altered('duplicate original seed', 'seed_index', (0, 1), 0)
        altered('no active original seed', 'seed_mask', (0, slice(None)), False)
        altered('zero native spacing', 'spacing_mm', (0, 2), 0.)
        altered('nonfinite spacing', 'spacing_mm', (0, 2), float('nan'))
        altered('nonfinite supported distance', 'distance_mm', (0, 1), float('nan'))
        altered('negative supported distance', 'distance_mm', (0, 1), -1.)
        altered('undefined supported shell', 'band', (0, 1), 0)
        altered('original anchor sphere has no support', 'distance_mm', (0, slice(0, 13)), 100.)
        for name, inputs in cases:
            with self.subTest(contract=name), self.assertRaises(ValueError):
                retain_path_relays(**inputs)

    def test_integrated_mandatory_z_edges_survive_in_each_context_relation(self):
        inputs, sampling, pair_xyz, pair_mask = anisotropic_pair_fixture()
        profile = V1RelationalSparseProfile(16, 3., 5., 10.)
        base = _build_pair_edges(pair_xyz, pair_mask, profile)
        result = build_relay_pair_edges(pair_xyz, pair_mask, profile, sampling)
        ordinary = _dense_pair_adjacency(base)
        actual = _dense_pair_adjacency(result)
        scenes, target_slot, source_slot = sampling['mandatory_edges']
        pairs, slots = pair_mask.shape[0], pair_mask.shape[-1]
        side, pair = scenes // pairs, scenes % pairs
        target, source = side * slots + target_slot + 1, side * slots + source_slot + 1
        expected = ordinary.clone()
        expected[pair, side, target, source] = True
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(actual[pair, side, target, source].all())
        delta = pair_xyz.reshape(pairs, 2 * slots, 3)[pair, target] - pair_xyz.reshape(pairs, 2 * slots, 3)[pair, source]
        vertical = (delta[:, :2] == 0).all(-1) & (delta[:, 2].abs() == 5.)
        for relation in (0, 1):
            rescued = vertical & (side == relation) & ~ordinary[pair, side, target, source]
            self.assertTrue(rescued.any(), f'No real Z5 edge rescued in context relation {relation}')
            self.assertTrue(actual[pair[rescued], side[rescued], target[rescued], source[rescued]].all())
            self.assertGreater(result['in_degree'][:, relation].max().item(), 3)
        self.assertTrue(torch.equal(result['in_degree'], actual.sum(-1)))
        self.assertTrue(torch.equal(result['out_degree'], actual.sum(-2)))
        # Query/cross relations remain exactly their radius-gated baseline.
        self.assertTrue(torch.equal(actual[:, 2:], ordinary[:, 2:]))
        edge_pair, relation, edge_target, edge_source = result['edge_index']
        self.assertTrue(result['relation_target_mask'][edge_pair, relation, edge_target].all())
        self.assertTrue(result['relation_source_mask'][edge_pair, relation, edge_source].all())
        self.assertTrue((result['edge_distance_mm'] <=
            pair_xyz.new_tensor(profile.relation_radii_mm)[relation]).all())
        rows, columns = result['mean_adjacency'].coalesce().indices()
        self.assertTrue(torch.equal(rows // (8 * 2 * slots), columns // (2 * slots)))
        self.assertTrue(result['mandatory_edge_audit']['native_parent_path_recomputed_on_GPU'])
        self.assertFalse(result['mandatory_edge_audit']['mst'])
        self.assertFalse(result['mandatory_edge_audit']['out_of_radius_fallback'])

    def test_duplicate_union_uses_unique_degree_and_separate_relation_means(self):
        _, sampling, pair_xyz, pair_mask = anisotropic_pair_fixture()
        profile = V1RelationalSparseProfile(16, 3., 5., 10.)
        ordinary = build_relay_pair_edges(pair_xyz, pair_mask, profile, sampling)
        repeated = copy.deepcopy(sampling)
        repeated['mandatory_edges'] = repeated['mandatory_edges'].repeat(1, 2)
        repeated['edge_path_length_mm'] = repeated['edge_path_length_mm'].repeat(2)
        repeated['edge_path_steps'] = repeated['edge_path_steps'].repeat(2)
        result = build_relay_pair_edges(pair_xyz, pair_mask, profile, repeated)
        for key in ('edge_index', 'in_degree', 'out_degree'):
            self.assertTrue(torch.equal(ordinary[key], result[key]), key)
        self.assertTrue(torch.equal(ordinary['mean_adjacency'].to_dense(),
            result['mean_adjacency'].to_dense()))
        adjacency = _dense_pair_adjacency(result).float()
        batch, relations, nodes, _ = adjacency.shape
        hidden = torch.arange(batch * nodes * 4, device='cuda').float().reshape(batch, nodes, 4)
        actual = torch.sparse.mm(result['mean_adjacency'], hidden.reshape(batch * nodes, 4))
        actual = actual.reshape(batch, relations, nodes, 4)
        expected = torch.matmul(adjacency, hidden[:, None]) / adjacency.sum(-1, keepdim=True).clamp_min(1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-6))
        rows, _ = result['mean_adjacency'].indices()
        mass = torch.zeros(batch * relations * nodes, device='cuda')
        mass.index_add_(0, rows, result['mean_adjacency'].values())
        self.assertTrue(torch.allclose(mass.reshape(batch, relations, nodes),
            (result['in_degree'] > 0).float(), atol=1e-6, rtol=0))

    def test_forged_native_witness_role_pair_padding_and_radius_are_rejected(self):
        _, sampling, pair_xyz, pair_mask = anisotropic_pair_fixture()
        profile = V1RelationalSparseProfile(16, 3., 5., 10.)
        invalid = []
        short = copy.deepcopy(sampling)
        short['edge_path_length_mm'][0] *= .5
        invalid.append(('falsely shortened real path', pair_xyz, pair_mask, short))
        steps = copy.deepcopy(sampling)
        steps['edge_path_steps'][0] += 1
        invalid.append(('false native step count', pair_xyz, pair_mask, steps))
        for label, edge_row, value in (('foreign physical pair', 0, 4),
                                       ('abstract query used as context', 1, -1)):
            indices = copy.deepcopy(sampling)
            indices['mandatory_edges'][edge_row, 0] = value
            invalid.append((label, pair_xyz, pair_mask, indices))
        scene, target_slot, source_slot = sampling['mandatory_edges'][:, 0]
        pairs, slots = pair_mask.shape[0], pair_mask.shape[-1]
        side, pair = scene // pairs, scene % pairs
        padded = pair_mask.clone()
        padded[pair, side, target_slot + 1] = False
        invalid.append(('padding endpoint', pair_xyz, padded, sampling))
        mismatch = pair_xyz.clone()
        mismatch[pair, side, target_slot + 1, 0] += .125
        invalid.append(('foreign scene geometry', mismatch, pair_mask, sampling))
        out_radius = pair_xyz.clone()
        out_radius[pair, side, target_slot + 1, 2] += 20.
        invalid.append(('out of physical radius', out_radius, pair_mask, sampling))
        child_pool = sampling['selection_index'][scene, target_slot]
        ancestor_pool = sampling['selection_index'][scene, source_slot]
        if sampling['fine_depth'][scene, child_pool] < sampling['fine_depth'][scene, ancestor_pool]:
            child_pool, ancestor_pool = ancestor_pool, child_pool
        unsupported = copy.deepcopy(sampling)
        unsupported['fine_eligible'][scene, child_pool] = False
        invalid.append(('background witness', pair_xyz, pair_mask, unsupported))
        cycle = copy.deepcopy(sampling)
        cycle['fine_parent'][scene, child_pool] = child_pool
        invalid.append(('cyclic witness', pair_xyz, pair_mask, cycle))
        nonnative = copy.deepcopy(sampling)
        nonnative['fine_xyz_local'][scene, child_pool, 0] += 1.
        invalid.append(('diagonal native witness', pair_xyz, pair_mask, nonnative))
        for label, xyz, mask, witness in invalid:
            with self.subTest(contract=label), self.assertRaises((ValueError, IndexError)):
                build_relay_pair_edges(xyz, mask, profile, witness)

    def test_forged_concave_hole_shortcut_cannot_enter_integrated_edges(self):
        points = ([(0, y, 0) for y in range(7)] +
                  [(x, 6, 0) for x in range(1, 7)] +
                  [(6, y, 0) for y in range(5, -1, -1)])
        one = unit_scene(points, [(0, 0, 0), (6, 0, 0)])
        inputs = {key: value.repeat((2,) + (1,) * (value.ndim - 1)) for key, value in one.items()}
        sampling = retain_path_relays(**inputs)
        pair_xyz, pair_mask = packed_pair_geometry(inputs, sampling)
        anchor_slot = (sampling['selection_index'][0] == 0).nonzero()[0, 0]
        far_slot = (sampling['selection_index'][0] == 18).nonzero()[0, 0]
        fake = torch.stack((torch.zeros(2, device='cuda', dtype=torch.long),
            torch.stack((far_slot, anchor_slot)), torch.stack((anchor_slot, far_slot))))
        sampling['mandatory_edges'] = torch.cat((sampling['mandatory_edges'], fake), -1)
        sampling['edge_path_length_mm'] = torch.cat((sampling['edge_path_length_mm'],
            torch.ones(2, device='cuda')))
        sampling['edge_path_steps'] = torch.cat((sampling['edge_path_steps'],
            torch.ones(2, device='cuda', dtype=torch.long)))
        with self.assertRaises(ValueError):
            build_relay_pair_edges(pair_xyz, pair_mask,
                V1RelationalSparseProfile(16, 3., 5., 10.), sampling)

    def test_dynamic_role_coverage_uses_each_scene_slots_and_known_distances(self):
        graph = rotated_role_coverage_fixture()
        result = all_pool_distances(graph, workspace_bytes=1024 * 1024)
        streamed = all_pool_distances(graph, workspace_bytes=2 * 3 * 4 * 12)
        for band, small in zip(result, streamed):
            role = band['role']
            point = (role - 1) * 2
            difference = (2., 3., 4.)[role - 1]
            feature = torch.tensor([[0., 1.], [1., 0.]], device='cuda')
            spatial = feature * difference
            joint = feature * 4.
            self.assertTrue(torch.equal(band['full_counts'], torch.tensor([2, 2], device='cuda')))
            self.assertTrue(torch.equal(band['selected_counts'], torch.tensor([1, 1], device='cuda')))
            for key, expected in (('feature_deficit', feature), ('spatial_mm', spatial), ('joint', joint)):
                self.assertTrue(torch.allclose(band['values'][key][:, point:point + 2],
                    expected, atol=1e-6, rtol=1e-6), (role, key))
                self.assertTrue(torch.allclose(band['values'][key][band['mask']],
                    small['values'][key][small['mask']], atol=1e-6, rtol=1e-6))
                self.assertTrue(torch.allclose(band['summaries'][key], small['summaries'][key],
                    atol=1e-6, rtol=1e-6))
            self.assertTrue(torch.allclose(band['summaries']['feature_deficit'],
                feature.new_tensor([.5, .95, 1.]).expand(2, -1), atol=1e-6, rtol=1e-6))
            self.assertTrue(torch.allclose(band['summaries']['spatial_mm'],
                feature.new_tensor([difference / 2, difference * .95, difference]).expand(2, -1),
                atol=1e-6, rtol=1e-6))
            self.assertTrue(torch.allclose(band['summaries']['joint'],
                feature.new_tensor([2., 3.8, 4.]).expand(2, -1), atol=1e-6, rtol=1e-6))
        # A role present in scene0 cannot stand in for its absence in scene1.
        missing = copy.deepcopy(graph)
        missing['role'][1, 3] = 2
        with self.assertRaisesRegex(ValueError, 'Empty selected context band'):
            all_pool_distances(missing, workspace_bytes=1024 * 1024)

    def test_model_deduplicates_exact_scene_work_and_matches_expanded_oracle(self):
        reference, profile, batch, kwargs = unit_model_fixture()
        model = RelayedV1SparseL0(reference, profile).eval()
        with torch.no_grad(), patch.object(model.cnn, 'forward', wraps=model.cnn.forward) as cnn:
            output, graph = model(batch, return_graph=True, return_pool=True, **kwargs)
        self.assertEqual(cnn.call_count, 1)
        self.assertEqual(output.shape, (2, 128))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(graph['scene_pair'].tolist(), [0, 1, 0, 1])
        self.assertEqual(graph['scene_side'].tolist(), [0, 0, 1, 1])
        self.assertTrue(torch.equal(graph['anchors_native'], torch.cat((
            kwargs['donor_centers_native'], kwargs['recipient_centers_native']))))
        sampling = graph['sampling']
        self.assertEqual(sampling['diagnostics']['selection_unique_scenes'], 3)
        self.assertEqual(sampling['diagnostics']['selection_physical_scene_occurrences'], 4)
        self.assertEqual(sampling['diagnostics']['duplicate_scene_reuse'], 1)
        self.assertTrue(sampling['diagnostics']['exact_scene_geometry_support_verified'])
        for key, value in reference.cnn.state_dict().items():
            self.assertTrue(torch.equal(value, model.cnn.state_dict()[key]), key)
        coverage = graph['coverage_pool']
        xyz = sampling['fine_xyz_local']
        relative = coverage['relative_mm']
        distance = relative.norm(dim=-1)
        spacing = sampling['fine_spacing_mm']
        # Independent, fully expanded selection/traversal has no scene reuse.
        seeds, seed_mask, _ = _select_context(relative, coverage['features'],
            coverage['eligible'], coverage['band'], distance, profile)
        oracle = retain_path_relays(xyz, coverage['eligible'], coverage['band'],
            distance, spacing, seeds, seed_mask)
        for key in ('selection_index', 'selection_mask', 'roles', 'seed_mask',
                    'mandatory_edges', 'edge_path_length_mm', 'edge_path_steps',
                    'query_roots', 'fine_parent', 'fine_reachable', 'fine_depth'):
            self.assertTrue(torch.equal(sampling[key], oracle[key]), key)
        origin = torch.tensor([a['origin'] for a in batch.audit], device='cuda')[graph['crop_index']]
        anchor = graph['anchors_native'] - origin
        scene_keys = torch.cat((graph['crop_index'][:, None], anchor.long()), -1)
        args = (xyz, relative, coverage['features'], coverage['eligible'],
            coverage['band'], distance, spacing)
        invalid = xyz.clone()
        invalid[1, 0, 0] += 1.
        with self.assertRaisesRegex(ValueError, 'identical native scene geometry/support'):
            model._select_nodes(invalid, *args[1:], scene_keys=scene_keys)
        # The same three unique scenes must still expand to all 64 scene
        # occurrences of a physical32 batch, with exact donor/recipient mapping.
        chosen = torch.cat((torch.zeros(32, device='cuda', dtype=torch.long),
            torch.tensor([2, 3] * 16, device='cuda')))
        expanded_args = tuple(value[chosen] for value in args)
        selected, mask, roles, _, physical32 = model._select_nodes(
            *expanded_args, scene_keys=scene_keys[chosen])
        self.assertEqual(selected.shape[0], 64)
        self.assertEqual(physical32['diagnostics']['selection_unique_scenes'], 3)
        self.assertEqual(physical32['diagnostics']['selection_physical_scene_occurrences'], 64)
        self.assertEqual(physical32['diagnostics']['duplicate_scene_reuse'], 61)
        self.assertTrue(torch.equal(selected, sampling['selection_index'][chosen]))
        self.assertTrue(torch.equal(mask, sampling['selection_mask'][chosen]))
        self.assertTrue(torch.equal(roles, sampling['roles'][chosen]))
        self.assertTrue(torch.equal(physical32['fine_parent'], sampling['fine_parent'][chosen]))
        slots = graph['node_mask'].shape[1]
        pair_xyz = graph['relative_xyz_mm'][chosen].reshape(2, 32, slots, 3).transpose(0, 1)
        pair_mask = graph['node_mask'][chosen].reshape(2, 32, slots).transpose(0, 1)
        edges = build_relay_pair_edges(pair_xyz, pair_mask, profile, physical32, diagnostics=False)
        self.assertEqual(edges['in_degree'].shape, (32, 8, 2 * slots))
        self.assertEqual(edges['mean_adjacency'].shape,
            (32 * 8 * 2 * slots, 32 * 2 * slots))
        self.assertTrue((torch.bincount(edges['edge_index'][0], minlength=32) > 0).all())
        row, column = edges['mean_adjacency'].indices()
        self.assertTrue(torch.equal(row // (8 * 2 * slots), column // (2 * slots)))


if __name__ == '__main__':
    unittest.main()
