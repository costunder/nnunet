"""CUDA synthetic UNIT geometry/features; no actual CT or accuracy evidence."""
import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import unittest

import torch
from torch.nn import functional as F

from l0_sparse_feature.coverage import all_pool_distances


def graph_fixture(selected=(1, 2), padding=False):
    """Two parallel scenes with three distinct points in each role band."""
    feature = F.pad(torch.tensor([[1., 0.], [1., 0.], [0., 1.]], device='cuda'), (0, 66))
    xyz = torch.tensor([[0., 0., 0.], [10., 0., 0.], [0., 1., 0.]], device='cuda')
    pools, positions, eligibility, labels = [], [], [], []
    nodes = [feature.mean(0)[None]]
    node_positions = [xyz.new_zeros((1, 3))]
    node_masks = [torch.ones(1, device='cuda', dtype=torch.bool)]
    roles = [torch.zeros(1, device='cuda', dtype=torch.long)]
    df, ds = [], []
    for role in (1, 2, 3):
        pos = xyz + xyz.new_tensor([0., 0., role * 3.])
        df.append(2 * (feature - feature.mean(0)).square().sum(-1).mean())
        ds.append(2 * (pos - pos.mean(0)).square().sum(-1).mean())
        pools.append(feature)
        positions.append(pos)
        eligibility.append(torch.ones(3, device='cuda', dtype=torch.bool))
        labels.append(torch.full((3,), role, device='cuda', dtype=torch.long))
        nodes.append(feature[list(selected)])
        node_positions.append(pos[list(selected)])
        node_masks.append(torch.ones(len(selected), device='cuda', dtype=torch.bool))
        roles.append(torch.full((len(selected),), role, device='cuda', dtype=torch.long))
        if padding:
            pools.append(feature.new_full((1, 68), 9.))
            positions.append(pos.new_full((1, 3), 1000.))
            eligibility.append(torch.zeros(1, device='cuda', dtype=torch.bool))
            labels.append(torch.full((1,), role, device='cuda', dtype=torch.long))
            nodes.append(feature.new_full((1, 68), 9.))
            node_positions.append(pos.new_full((1, 3), 1000.))
            node_masks.append(torch.zeros(1, device='cuda', dtype=torch.bool))
            roles.append(torch.full((1,), role, device='cuda', dtype=torch.long))
    repeat = lambda x: x[None].expand(2, *x.shape).clone()
    return dict(coverage_pool=dict(features=repeat(torch.cat(pools)),
        relative_mm=repeat(torch.cat(positions)), eligible=repeat(torch.cat(eligibility)),
        band=repeat(torch.cat(labels))), features=repeat(torch.cat(nodes)),
        relative_xyz_mm=repeat(torch.cat(node_positions)), node_mask=repeat(torch.cat(node_masks)),
        role=repeat(torch.cat(roles)), statistics=dict(
            band_feature_variance=repeat(torch.stack(df)),
            band_spatial_variance_mm2=repeat(torch.stack(ds))))


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA numerical UNIT tests require a GPU; no CPU fallback/all-skip PASS')
        cls.flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = cls.flags

    def assert_same(self, a, b):
        for left, right in zip(a, b):
            self.assertTrue(torch.equal(left['mask'], right['mask']))
            for key in left['values']:
                self.assertTrue(torch.allclose(left['values'][key][left['mask']],
                    right['values'][key][right['mask']], atol=1e-6, rtol=1e-6), key)
                self.assertTrue(torch.allclose(left['summaries'][key],
                    right['summaries'][key], atol=1e-6, rtol=1e-6), key)

    def test_joint_uses_same_node_not_independent_minima(self):
        graph = graph_fixture()
        result = all_pool_distances(graph, workspace_bytes=1024 * 1024)
        # Target feature matches the distant A; the nearer B has another feature.
        for band in result:
            target = (band['role'] - 1) * 3
            feature = band['values']['feature_deficit'][:, target]
            space = band['values']['spatial_mm'][:, target]
            joint = band['values']['joint'][:, target]
            ds = graph['statistics']['band_spatial_variance_mm2'][:, band['role']-1]
            independent = space.square() / ds + 2 * feature / graph['statistics']['band_feature_variance'][:, band['role']-1]
            self.assertTrue(torch.equal(feature, torch.zeros_like(feature)))
            self.assertTrue(torch.allclose(space, torch.ones_like(space)))
            # Three full-pool spatial gaps [1, 0, 0] give a linear p95 of .9.
            expected_summary = space.new_tensor([1/3, .9, 1]).expand(2, -1)
            self.assertTrue(torch.allclose(band['summaries']['spatial_mm'],
                expected_summary, atol=1e-6, rtol=1e-6))
            self.assertTrue((joint > independent + 1).all())
            # Explicit CUDA direct-distance oracle keeps both terms at one j.
            slots = (graph['role'][0] == band['role']).nonzero().flatten()
            u = F.normalize(graph['coverage_pool']['features'][:, target], dim=-1)
            v = F.normalize(graph['features'][:, slots], dim=-1)
            x = graph['coverage_pool']['relative_mm'][:, target]
            y = graph['relative_xyz_mm'][:, slots]
            expected = (((x[:, None]-y).square().sum(-1) / ds[:, None])
                + ((u[:, None]-v).square().sum(-1)
                   / graph['statistics']['band_feature_variance'][:, band['role']-1, None])).min(-1).values
            self.assertTrue(torch.allclose(joint, expected, atol=1e-6, rtol=1e-6))

    def test_full_retention_has_zero_coverage_error(self):
        result = all_pool_distances(graph_fixture((0, 1, 2)), workspace_bytes=1024 * 1024)
        for band in result:
            self.assertTrue(torch.equal(band['full_counts'], band['selected_counts']))
            for values in band['summaries'].values():
                self.assertTrue(torch.allclose(values, torch.zeros_like(values), atol=1e-6, rtol=0))

    def test_nested_selected_sets_cannot_increase_any_point_or_summary(self):
        results = [all_pool_distances(graph_fixture(selected), workspace_bytes=1024*1024)
                   for selected in ((1,), (1, 2), (1, 2, 0))]
        for small, large in zip(results, results[1:]):
            for before, after in zip(small, large):
                for key in before['values']:
                    mask = before['mask']
                    self.assertTrue((after['values'][key][mask] <= before['values'][key][mask]+1e-6).all())
                    self.assertTrue((after['summaries'][key] <= before['summaries'][key]+1e-6).all())

    def test_padding_and_ineligible_values_cannot_change_metrics(self):
        graph = graph_fixture(padding=True)
        altered = copy.deepcopy(graph)
        excluded = ~altered['coverage_pool']['eligible']
        altered['coverage_pool']['features'][excluded] = float('nan')
        altered['coverage_pool']['relative_mm'][excluded] = float('inf')
        pad = ~altered['node_mask']
        altered['features'][pad] = float('nan')
        altered['relative_xyz_mm'][pad] = float('inf')
        self.assert_same(all_pool_distances(graph, workspace_bytes=1024*1024),
                         all_pool_distances(altered, workspace_bytes=1024*1024))

    def test_workspace_chunking_preserves_complete_results(self):
        graph = graph_fixture()
        one_row = 2 * 2 * 4 * 12
        streamed = all_pool_distances(graph, workspace_bytes=one_row)
        full = all_pool_distances(graph, workspace_bytes=1024*1024)
        self.assertTrue(all(b['workspace_chunk_points']==1 for b in streamed))
        self.assert_same(streamed, full)
        with self.assertRaisesRegex(MemoryError, 'one complete point'):
            all_pool_distances(graph, workspace_bytes=one_row-1)

    def test_eligible_zero_and_nonfinite_norms_fail_explicitly(self):
        for key in ('pool', 'selected'):
            # Finite FP32 components can still overflow their computed norm.
            for invalid in (0., float('nan'), float('inf'), 1e30):
                with self.subTest(source=key, invalid=invalid):
                    graph = graph_fixture()
                    if key=='pool': graph['coverage_pool']['features'][:, 0] = invalid
                    else: graph['features'][:, 1] = invalid
                    with self.assertRaises(FloatingPointError):
                        all_pool_distances(graph, workspace_bytes=1024*1024)


if __name__ == '__main__':
    unittest.main()
