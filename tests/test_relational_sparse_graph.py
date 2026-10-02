"""Explicit synthetic CUDA UNIT tensors, not CT or learned CP performance.

The CPU class checks profile metadata only. Every numerical CNN, graph,
message-passing, pooling and backward check requires real CUDA. These tests do
not change the final model/data/training configuration or run full training.
"""
import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from l0_local_cnn.model import LocalBatch, LocalCNN, MODE
from l0_sparse_feature.model import SparseFeatureCoverageError
from l0_sparse_feature.relational import (
    RELATION_NAMES, V1RelationalSparseL0, V1RelationalSparseProfile,
    _RelationalMeanSAGE, _build_pair_edges, _dense_pair_adjacency,
)


def unit_fixture(*, quota=16):
    """Full-width DEBUG CNN with two physical UNIT pairs and three crops."""
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
    spacings = [[1., 1., 1.], [1., 1.5, 2.], [2., 1., 1.]]
    audit = [dict(case=f'UNIT crop {i}', origin=origins[i],
        shape=[21, 19 if i == 2 else 21, 21], spacing=spacings[i],
        anchor_in_organ=True) for i in range(3)]
    donor = torch.zeros(2, device='cuda', dtype=torch.long)
    recipient = torch.tensor([1, 2], device='cuda')
    batch = LocalBatch(images, organ, donor, recipient,
        torch.arange(2, device='cuda'), audit).validate()
    donor_native = torch.tensor(origins[0], device='cuda').float()[None] + 10
    donor_native = donor_native.expand(2, -1).clone()
    recipient_native = torch.tensor(origins[1:], device='cuda').float() + 10
    profile = V1RelationalSparseProfile(context_nodes_per_band=quota,
        query_radius_mm=3., near_radius_mm=5., mid_radius_mm=10.)
    kwargs = dict(recipient_centers_native=recipient_native,
        donor_centers_native=donor_native)
    return reference, profile, batch, kwargs


def clone_unit_batch(batch):
    return LocalBatch(batch.images.clone(), batch.organ.clone(),
        batch.donor.clone(), batch.recipient.clone(), batch.indices.clone(),
        copy.deepcopy(batch.audit))


def unit_edge_fixture():
    """Two tiny UNIT graphs, including actual isolated and padded nodes."""
    donor = [0., 1., 2., 3., 4., 6., 30., float('nan')]
    recipient = [0., .25, 1.25, 2.25, 4.25, 8.25, 40., float('nan')]
    axis = torch.tensor([donor, recipient], device='cuda')
    xyz = torch.stack((axis, torch.zeros_like(axis), torch.zeros_like(axis)), -1)
    xyz = xyz[None].repeat(2, 1, 1, 1)
    mask = torch.ones((2, 2, 8), device='cuda', dtype=torch.bool)
    mask[:, :, -1] = False
    return xyz, mask


def reference_relations(xyz, mask):
    """Independent receiver-row/source-column radius-then-nearest oracle."""
    pairs, _, slots = mask.shape
    nodes = slots * 2
    flat_mask = mask.reshape(pairs, nodes)
    index = torch.arange(nodes, device=xyz.device)
    source_query = (index == 0)[None] & flat_mask
    target_query = (index == slots)[None] & flat_mask
    source_context = ((index > 0) & (index < slots))[None] & flat_mask
    target_context = (index > slots)[None] & flat_mask
    source_masks = (source_context, target_context, source_query,
        source_context, target_query, target_context,
        source_context, source_query)
    target_masks = (source_context, target_context, source_context,
        source_query, target_context, target_query,
        target_context, target_context)
    distance = torch.cdist(xyz.reshape(pairs, nodes, 3),
        xyz.reshape(pairs, nodes, 3))
    self_edge = torch.eye(nodes, device=xyz.device, dtype=torch.bool)[None]
    adjacency, candidates = [], []
    for relation, (radius, k) in enumerate(zip((6., 6., 8., 8., 8., 8., 5., 8.),
                                               (3, 3, 3, 3, 3, 3, 1, 3))):
        legal = target_masks[relation][:, :, None] & source_masks[relation][:, None, :]
        legal = legal & ~self_edge & (distance <= radius)
        nearest = distance.masked_fill(~legal, float('inf')).argsort(dim=-1, stable=True)[..., :k]
        chosen = torch.zeros_like(legal).scatter(-1, nearest, True) & legal
        adjacency.append(chosen)
        candidates.append(legal.sum(-1))
    return (torch.stack(adjacency, 1), torch.stack(candidates, 1),
        torch.stack(source_masks, 1), torch.stack(target_masks, 1))


class Budget:
    def __init__(self, fail=None):
        self.calls, self.fail = 0, fail

    def check(self):
        self.calls += 1
        if self.calls == self.fail:
            raise MemoryError('UNIT explicit resource budget rejection')


class CPUProfileMetadata(unittest.TestCase):
    """CPU metadata checks; never evidence of numerical model execution."""

    def test_profile_retains_reviewed_debug_width_depth_and_quota(self):
        profile = V1RelationalSparseProfile(16, 3., 5., 10.)
        profile.validate()
        metadata = asdict(profile)
        self.assertTrue(metadata['debug'])
        self.assertEqual(metadata['hidden_dim'], 128)
        self.assertEqual(metadata['sage_layers'], 3)
        self.assertEqual(len(RELATION_NAMES), 8)
        self.assertEqual(len(set(RELATION_NAMES)), 8)
        for quota in (16, 32, 64):
            replace(profile, context_nodes_per_band=quota).validate()
        for changes in (dict(debug=False), dict(context_nodes_per_band=48),
                        dict(hidden_dim=64), dict(sage_layers=1),
                        dict(query_radius_mm=0.)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(profile, **changes).validate()


class CUDARelationalGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('Numerical UNIT tests require real CUDA; '
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

    def test_radius_nearest_direction_counts_and_isolation_have_no_fallback(self):
        profile = V1RelationalSparseProfile(16, 3., 5., 10.)
        xyz, mask = unit_edge_fixture()
        edges = _build_pair_edges(xyz, mask, profile)
        actual = _dense_pair_adjacency(edges)
        expected, candidates, source_mask, target_mask = reference_relations(xyz, mask)
        self.assertEqual(actual.shape, (2, 8, 16, 16))
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(edges['candidate_counts'], candidates))
        self.assertTrue(torch.equal(edges['relation_source_mask'], source_mask))
        self.assertTrue(torch.equal(edges['relation_target_mask'], target_mask))
        self.assertTrue(torch.equal(edges['in_degree'], expected.sum(-1)))
        self.assertTrue(torch.equal(edges['out_degree'], expected.sum(-2)))
        self.assertTrue(torch.equal(edges['unmatched_target_mask'],
            target_mask & ~expected.any(-1)))
        self.assertTrue(torch.equal(edges['unmatched_source_mask'],
            source_mask & ~expected.any(-2)))
        limits = torch.tensor([3, 3, 3, 3, 3, 3, 1, 3], device='cuda')
        self.assertTrue((edges['in_degree'] <= limits[None, :, None]).all())
        # The strict radius gate leaves genuine nodes at 30/40 mm isolated.
        # No nearest-outside-radius, symmetry closure or MST edge is legal.
        for isolated in (6, 14):
            self.assertFalse(actual[:, :, isolated].any())
            self.assertFalse(actual[:, :, :, isolated].any())
        self.assertEqual(edges['components'].tolist(), [3, 3])
        self.assertFalse(edges['weak_connected'].any())
        self.assertFalse(actual.diagonal(dim1=-2, dim2=-1).any())
        for padding in (7, 15):
            self.assertFalse(actual[:, :, padding].any())
            self.assertFalse(actual[:, :, :, padding].any())
        # The only cross-branch edges point donor -> recipient.
        self.assertFalse(actual[:, :, :8, 8:].any())
        self.assertTrue(actual[:, 6:, 8:, :8].any())
        pair, relation, target, source = edges['edge_index']
        self.assertTrue(torch.equal(actual.nonzero().T, edges['edge_index']))
        self.assertTrue((target[relation >= 6] >= 8).all())
        self.assertTrue((source[relation >= 6] < 8).all())
        distance = torch.linalg.vector_norm(
            xyz.reshape(2, 16, 3)[pair, target] -
            xyz.reshape(2, 16, 3)[pair, source], dim=-1)
        self.assertTrue(torch.allclose(edges['edge_distance_mm'], distance))
        radii = distance.new_tensor([6., 6., 8., 8., 8., 8., 5., 8.])
        self.assertTrue((distance <= radii[relation]).all())
        sparse = edges['mean_adjacency'].coalesce()
        self.assertEqual(sparse.shape, (2 * 8 * 16, 2 * 16))
        row, column = sparse.indices()
        self.assertTrue(torch.equal(row // (8 * 16), column // 16))
        normalized = expected.float() / expected.sum(-1, keepdim=True).clamp_min(1)
        expected_matrix = torch.zeros((256, 32), device='cuda')
        expected_matrix[pair * 128 + relation * 16 + target,
            pair * 16 + source] = normalized[pair, relation, target, source]
        self.assertTrue(torch.equal(sparse.to_dense(), expected_matrix))

    def test_radius_boundary_keeps_only_in_radius_sources(self):
        profile = V1RelationalSparseProfile(16, 3., 5., 10.)
        xyz, mask = unit_edge_fixture()
        # A receiver at precisely 5 mm accepts donor context x=1 as one
        # candidate. Moving it just outside must remove that candidate.
        xyz[:, 1, 1, 0] = 6.
        boundary = _build_pair_edges(xyz, mask, profile)
        xyz[:, 1, 1, 0] = 6.001
        outside = _build_pair_edges(xyz, mask, profile)
        self.assertEqual((boundary['candidate_counts'][:, 6, 9] -
                          outside['candidate_counts'][:, 6, 9]).tolist(), [1, 1])
        invalid = xyz.clone()
        invalid[:, 0, 1, 0] = float('nan')
        with self.assertRaises((FloatingPointError, ValueError)):
            _build_pair_edges(invalid, mask, profile)

    def test_each_relation_mean_is_separate_and_root_residual_occurs_once(self):
        xyz, mask = unit_edge_fixture()
        edges = _build_pair_edges(xyz, mask, V1RelationalSparseProfile(16, 3., 5., 10.))
        adjacency = _dense_pair_adjacency(edges).float()
        # Hand-specified UNIT features provide unequal, identifiable means.
        node = torch.arange(16, device='cuda').float().square()
        channel = torch.linspace(.01, 1., 128, device='cuda')
        x = (node[:, None] * channel[None])[None].repeat(2, 1, 1)
        x[~mask.reshape(2, 16)] = 0
        block = _RelationalMeanSAGE(128, 8).cuda()
        mean = block.relation_means(x, edges['mean_adjacency'])
        expected_mean = torch.matmul(adjacency, x[:, None]) / adjacency.sum(-1, keepdim=True).clamp_min(1)
        self.assertTrue(torch.allclose(mean, expected_mean, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.equal(mean[:, 1, 10], mean[:, 6, 10]))
        with torch.no_grad():
            block.neighbor_weight.zero_()
        actual = block(x, edges, mask.reshape(2, 16))
        expected = torch.where(mask.reshape(2, 16, 1), F.silu(block.norm(x)), 0)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(block.root_weight)
        self.assertFalse(any(name.startswith('root') for name, _ in block.named_parameters()))
        with torch.no_grad():
            block.neighbor_weight[0].copy_(torch.eye(128, device='cuda'))
            block.neighbor_weight[6].copy_(2 * torch.eye(128, device='cuda'))
        actual = block(x, edges, mask.reshape(2, 16))
        expected = torch.where(mask.reshape(2, 16, 1),
            F.silu(block.norm(x + expected_mean[:, 0] + 2 * expected_mean[:, 6])), 0)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=1e-6))

    def test_native_anchors_roles_full_pool_and_single_cnn_forward(self):
        reference, profile, batch, kwargs = unit_fixture()
        model = V1RelationalSparseL0(reference, profile).eval()
        with torch.no_grad(), patch.object(model.cnn, 'forward', wraps=model.cnn.forward) as cnn:
            output, graph = model(batch, return_graph=True, return_pool=True, **kwargs)
        self.assertEqual(cnn.call_count, 1)
        self.assertEqual(output.shape, (2, 128))
        self.assertEqual(graph['features'].shape, (4, 49, 68))
        self.assertEqual(graph['joint_hidden'].shape, (2, 98, 128))
        self.assertEqual(graph['scene_readout'].shape, (4, 128))
        self.assertEqual(graph['pair_adjacency'].shape, (2, 8, 98, 98))
        self.assertEqual(graph['scene_pair'].tolist(), [0, 1, 0, 1])
        self.assertEqual(graph['scene_side'].tolist(), [0, 0, 1, 1])
        self.assertTrue(torch.equal(graph['anchors_native'], torch.cat((
            kwargs['donor_centers_native'], kwargs['recipient_centers_native']))))
        self.assertTrue(torch.equal(graph['xyz_native'][:, 0], graph['anchors_native']))
        self.assertTrue(graph['query_is_abstract'])
        self.assertFalse(graph['query_is_ct_sample'])
        expected_role = torch.cat((torch.zeros(1, device='cuda', dtype=torch.long),
            torch.arange(1, 4, device='cuda').repeat_interleave(16)))
        self.assertTrue(torch.equal(graph['role'], expected_role[None].expand(4, -1)))
        self.assertTrue(torch.equal(graph['global_role'],
            graph['role'] + graph['scene_side'][:, None] * 4))
        self.assertTrue((graph['statistics']['band_selected_counts'] == 16).all())
        crop = graph['crop_index']
        spacing = torch.tensor([a['spacing'] for a in batch.audit], device='cuda')[crop]
        self.assertTrue(torch.equal(graph['xyz_mm'], graph['xyz_native'] * spacing[:, None]))
        origin = torch.tensor([a['origin'] for a in batch.audit], device='cuda')[crop]
        local = (graph['xyz_native'] - origin[:, None]).long()
        scene = crop[:, None].expand_as(graph['node_mask'])
        organ_at_node = batch.organ[scene, 0, local[..., 0], local[..., 1], local[..., 2]]
        self.assertTrue(organ_at_node[:, 1:][graph['node_mask'][:, 1:]].all())
        for index in range(4):
            real = graph['xyz_native'][index, 1:][graph['node_mask'][index, 1:]]
            self.assertEqual(len(real.unique(dim=0)), len(real))
        radii = graph['relative_xyz_mm'].norm(dim=-1)
        self.assertTrue((radii[:, 1:17] <= 5.00001).all())
        self.assertTrue(((radii[:, 17:33] > 5.) & (radii[:, 17:33] <= 10.00001)).all())
        self.assertTrue((radii[:, 33:] > 10.).all())
        organ_counts = batch.organ.flatten(1).sum(1)[crop]
        self.assertTrue(torch.equal(graph['statistics']['organ_fine_pool_counts'], organ_counts))
        self.assertEqual(graph['statistics']['unique_cnn_crops'], 3)
        self.assertEqual(graph['statistics']['physical_pairs'], 2)
        self.assertEqual(graph['statistics']['fine_pool_stride'], 1)
        self.assertFalse(graph['statistics']['hard_selection_differentiable'])
        coverage = graph['coverage_pool']
        self.assertEqual(coverage['features'].shape[-1], 68)
        self.assertEqual(coverage['features'].device.type, 'cuda')
        self.assertFalse(coverage['features'].requires_grad)
        self.assertTrue(torch.equal(coverage['eligible'].sum(-1),
            graph['statistics']['eligible_all_scale_pool_counts']))
        # Each learned role/shell pool has unit mass on its own real nodes.
        pair_roles = torch.cat((graph['global_role'][:2], graph['global_role'][2:]), 1)
        pair_mask = torch.cat((graph['node_mask'][:2], graph['node_mask'][2:]), 1)
        weights = graph['pool_weights']
        self.assertEqual(weights.shape, (2, 8, 98))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones((2, 8), device='cuda')))
        for role in range(8):
            legal = pair_mask & (pair_roles == role)
            self.assertFalse(weights[:, role][~legal].any())
            self.assertTrue((weights[:, role][legal] > 0).all())
        self.assertTrue(torch.equal(weights[:, 0, 0], torch.ones(2, device='cuda')))
        self.assertTrue(torch.equal(weights[:, 4, 49], torch.ones(2, device='cuda')))

    def test_donor_changes_recipient_hidden_before_fusion_and_pairs_do_not_leak(self):
        reference, profile, batch, kwargs = unit_fixture()
        model = V1RelationalSparseL0(reference, profile).eval()
        donor_changed = clone_unit_batch(batch)
        donor_changed.images[0][donor_changed.organ[0]] += .75
        donor_changed.validate()
        recipient_changed = clone_unit_batch(batch)
        recipient_changed.images[2][recipient_changed.organ[2]] -= 1.25
        recipient_changed.validate()
        fusion_inputs = []
        hook = model.fuse.register_forward_pre_hook(
            lambda module, args: fusion_inputs.append(args[0].detach().clone()))
        try:
            with torch.no_grad():
                output, graph = model(batch, return_graph=True, **kwargs)
                donor_output, donor_graph = model(donor_changed, return_graph=True, **kwargs)
                recipient_output, recipient_graph = model(recipient_changed, return_graph=True, **kwargs)
        finally:
            hook.remove()
        self.assertEqual(len(fusion_inputs), 3)
        # Donor alteration cannot alter recipient CNN features or selected
        # recipient positions. It reaches recipient h via typed cross edges.
        self.assertTrue(torch.equal(graph['features'][2:], donor_graph['features'][2:]))
        self.assertTrue(torch.equal(graph['xyz_native'][2:], donor_graph['xyz_native'][2:]))
        self.assertFalse(torch.equal(graph['joint_hidden'][:, 49:],
            donor_graph['joint_hidden'][:, 49:]))
        self.assertFalse(torch.equal(graph['scene_readout'][2:],
            donor_graph['scene_readout'][2:]))
        self.assertFalse(torch.equal(fusion_inputs[0][:, 128:256],
            fusion_inputs[1][:, 128:256]))
        self.assertFalse(torch.equal(output, donor_output))
        # Recipient crop2 belongs only to pair1. No graph, CNN or readout
        # operation may couple it into pair0 or either donor hidden stream.
        self.assertTrue(torch.equal(output[0], recipient_output[0]))
        self.assertTrue(torch.equal(graph['joint_hidden'][0], recipient_graph['joint_hidden'][0]))
        self.assertTrue(torch.equal(graph['joint_hidden'][:, :49],
            recipient_graph['joint_hidden'][:, :49]))
        self.assertFalse(torch.equal(output[1], recipient_output[1]))
        self.assertTrue(torch.equal(graph['scene_readout'][0], recipient_graph['scene_readout'][0]))

    def test_all_trainable_modules_and_role_pool_rows_reach_optimizer(self):
        reference, profile, batch, kwargs = unit_fixture()
        model = V1RelationalSparseL0(reference, profile).train()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.)
        output, graph = model(batch, return_graph=True, **kwargs)
        self.assertTrue(torch.isfinite(output).all())
        direction = torch.linspace(-1., 1., 128, device='cuda')
        loss = (output * direction).mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0., name)
        # Every role's value transform and each context-only attention scorer
        # is connected. There is no unused singleton-query scorer parameter.
        self.assertTrue((model.attention_pool.value_weight.grad.flatten(1).abs().sum(1) > 0).all())
        self.assertTrue((model.attention_pool.score_weight.grad.abs().sum(1) > 0).all())
        for layer in model.blocks:
            self.assertTrue((layer.neighbor_weight.grad.flatten(1).abs().sum(1) > 0).all())
        optimizer.step()
        for name, parameter in model.named_parameters():
            self.assertFalse(torch.equal(before[name], parameter.detach()), name)

    def test_outside_organ_and_native_padding_nan_cannot_change_graph(self):
        reference, profile, batch, kwargs = unit_fixture()
        model = V1RelationalSparseL0(reference, profile).eval()
        altered = clone_unit_batch(batch)
        altered.images[~altered.organ] = float('nan')
        altered.validate()
        with torch.no_grad():
            output, graph = model(batch, return_graph=True, **kwargs)
            other, other_graph = model(altered, return_graph=True, **kwargs)
        self.assertTrue(torch.equal(output, other))
        for key in ('features', 'xyz_native', 'relative_xyz_mm', 'node_mask',
                    'pair_adjacency', 'joint_hidden', 'scene_readout',
                    'global_role', 'pool_weights', 'pooled_roles'):
            self.assertTrue(torch.equal(graph[key], other_graph[key]), key)

    def test_real_anchor_in_background_is_preserved_without_query_fallback(self):
        reference, profile, batch, kwargs = unit_fixture()
        hole = clone_unit_batch(batch)
        hole.organ[1, 0, 10, 10, 10] = False
        hole.audit[1]['anchor_in_organ'] = False
        hole.validate()
        with torch.no_grad():
            _, graph = V1RelationalSparseL0(reference, profile).eval()(hole, return_graph=True, **kwargs)
        self.assertTrue(torch.equal(graph['anchors_native'][2], kwargs['recipient_centers_native'][0]))
        self.assertTrue(torch.equal(graph['xyz_native'][2, 0], kwargs['recipient_centers_native'][0]))
        self.assertGreater(graph['statistics']['query_pool_counts'][2].item(), 0)
        self.assertFalse(hole.organ[1, 0, 10, 10, 10])

    def test_missing_band_query_native_audit_and_budget_fail_explicitly(self):
        reference, profile, batch, kwargs = unit_fixture()
        model = V1RelationalSparseL0(reference, profile).eval()
        axis = torch.arange(21, device='cuda').float() - 10
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1)
        near_only = clone_unit_batch(batch)
        near_only.organ[0, 0] &= grid.norm(dim=-1) <= 5.
        near_only.validate()
        with self.assertRaisesRegex(SparseFeatureCoverageError, 'Empty'):
            model(near_only, **kwargs)
        no_query = clone_unit_batch(batch)
        spacing = grid.new_tensor(batch.audit[1]['spacing'])
        no_query.organ[1, 0] &= (grid * spacing).norm(dim=-1) > 3.
        no_query.validate()
        with self.assertRaisesRegex(SparseFeatureCoverageError, 'anchor sphere'):
            model(no_query, **kwargs)
        bad_audit = clone_unit_batch(batch)
        bad_audit.audit[1]['shape'] = [1, 1, 1]
        bad_audit.validate()
        with self.assertRaisesRegex(ValueError, 'padding'):
            model(bad_audit, **kwargs)
        with self.assertRaisesRegex(MemoryError, 'budget'):
            V1RelationalSparseL0(reference, profile, Budget(fail=1)).eval()(batch, **kwargs)
        with patch.object(model.cnn, 'forward', wraps=model.cnn.forward) as cnn:
            with self.assertRaisesRegex(ValueError, 'return_pool requires return_graph'):
                model(batch, return_pool=True, **kwargs)
            self.assertEqual(cnn.call_count, 0)


if __name__ == '__main__':
    unittest.main()
