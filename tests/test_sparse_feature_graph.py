"""CUDA-only synthetic UNIT data; not actual CT or learned CP performance."""
import copy
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from l0_local_cnn.model import LocalBatch, LocalCNN, MODE
from l0_sparse_feature.model import (SparseFeatureL0, SparseFeatureProfile,
    SparseFeatureCoverageError, _MeanSAGE, _build_edges, _connected,
    _masked_corner_features, _select_context)


def fixture(*, narrow=False):
    config = dict(architecture=MODE, channels=[12,24,32], convolutions=[2,3,3],
        hidden_dim=128, margin_mm=10., input='native_spacing_organ_only',
        readout='organ_masked_mean_each_scale', fusion='donor_target_difference_product',
        initialization='fresh_seed42', learning_policy='same_donor_live_v1')
    torch.manual_seed(5342)
    reference = LocalCNN(config).cuda().eval()
    images = torch.randn(3, 1, 21, 21, 21, device='cuda')
    organ = torch.ones_like(images, dtype=torch.bool)
    organ[:, :, :2] = False
    organ[:, :, 20:] = False
    # Native shapes differ: upper padded cells must not be eligible positions.
    organ[2, :, :, 19:] = False
    origins = [[17,24,9], [80,2,53], [11,43,102]]
    spacing = [[1.,1.,1.], [1.,1.5,2.], [2.,1.,1.]]
    audit = [dict(case=f'UNIT crop {i}', origin=origins[i], shape=[21,19 if i==2 else 21,21],
                  spacing=spacing[i], anchor_in_organ=True) for i in range(3)]
    donor = torch.zeros(2, device='cuda', dtype=torch.long)
    recipient = torch.tensor([1,2], device='cuda')
    batch = LocalBatch(images, organ, donor, recipient, torch.arange(2, device='cuda'), audit).validate()
    d = torch.tensor(origins[0], device='cuda').float()[None] + torch.tensor([[10,10,10],[10,10,10]], device='cuda')
    r = torch.tensor(origins[1:], device='cuda').float() + torch.tensor([[10,10,10],[10,10,10]], device='cuda')
    profile = SparseFeatureProfile(context_nodes_per_band=16, query_radius_mm=3.,
        near_radius_mm=.5 if narrow else 5., mid_radius_mm=10.)
    return reference, profile, batch, dict(recipient_centers_native=r, donor_centers_native=d)


class Budget:
    def __init__(self, fail=None): self.calls, self.fail = 0, fail
    def check(self):
        self.calls += 1
        if self.calls == self.fail: raise MemoryError('UNIT explicit budget rejection')


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('These numerical UNIT tests require CUDA; no CPU fallback or all-skip PASS')
        cls.flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    @classmethod
    def tearDownClass(cls):
        a,b,c,d,e = cls.flags
        torch.backends.cuda.matmul.allow_tf32 = a
        torch.backends.cudnn.allow_tf32 = b
        torch.backends.cudnn.benchmark = c
        torch.backends.cudnn.deterministic = d
        torch.use_deterministic_algorithms(e)

    def test_native_geometry_full_pool_roles_unique_crops_and_disjoint_graph(self):
        reference, profile, batch, kwargs = fixture()
        model = SparseFeatureL0(reference, profile).eval()
        with torch.no_grad(), patch.object(model.cnn, 'forward', wraps=model.cnn.forward) as cnn:
            output, graph = model(batch, return_graph=True, **kwargs)
        self.assertEqual(cnn.call_count, 1)
        self.assertEqual(output.shape, (2,128))
        self.assertEqual(graph['features'].shape, (4,49,68))
        self.assertEqual(graph['adjacency'].shape, (4,49,49))
        self.assertTrue(_connected(graph['adjacency'], graph['node_mask']).all())
        self.assertTrue(torch.equal(graph['adjacency'], graph['adjacency'].transpose(1,2)))
        self.assertFalse(graph['adjacency'].diagonal(dim1=1,dim2=2).any())
        self.assertTrue(torch.equal(graph['anchors_native'], torch.cat((kwargs['donor_centers_native'],kwargs['recipient_centers_native']))))
        self.assertTrue(torch.equal(graph['xyz_native'][:,0], graph['anchors_native']))
        self.assertTrue(graph['query_is_abstract'])
        self.assertFalse(graph['query_is_ct_sample'])
        self.assertEqual(graph['scene_pair'].tolist(), [0,1,0,1])
        self.assertEqual(graph['scene_side'].tolist(), [0,0,1,1])
        spacing = torch.tensor([a['spacing'] for a in batch.audit], device='cuda')[graph['crop_index']]
        self.assertTrue(torch.equal(graph['xyz_mm'], graph['xyz_native'] * spacing[:,None]))
        organ_counts = batch.organ.flatten(1).sum(1)[graph['crop_index']]
        self.assertTrue(torch.equal(graph['statistics']['organ_fine_pool_counts'],organ_counts))
        self.assertTrue((graph['statistics']['band_selected_counts']==16).all())
        origins = torch.tensor([a['origin'] for a in batch.audit],device='cuda')[graph['crop_index']]
        local = (graph['xyz_native'] - origins[:,None]).long()
        scene = graph['crop_index'][:,None].expand_as(graph['node_mask'])
        self.assertTrue(batch.organ[scene,0,local[...,0],local[...,1],local[...,2]][:,1:].all())
        for g in range(4):
            valid_xyz=graph['xyz_native'][g,1:][graph['node_mask'][g,1:]]
            self.assertEqual(len(valid_xyz.unique(dim=0)),len(valid_xyz))
        radii = graph['relative_xyz_mm'].norm(dim=-1)
        self.assertTrue((radii[:,1:17] <= 5.00001).all())
        self.assertTrue(((radii[:,17:33]>5)&(radii[:,17:33]<=10.00001)).all())
        self.assertTrue((radii[:,33:] > 10).all())

    def test_gradient_every_trainable_module_and_reference_is_unchanged(self):
        reference, profile, batch, kwargs = fixture()
        saved={k:v.clone() for k,v in reference.state_dict().items() if isinstance(v,torch.Tensor)}
        budget = Budget()
        model=SparseFeatureL0(reference,profile,budget).eval()
        output,graph=model(batch,return_graph=True,**kwargs)
        output.square().mean().backward()
        missing=[n for n,p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all() or not p.grad.abs().sum()>0)]
        self.assertEqual(missing,[])
        self.assertFalse(graph['statistics']['hard_selection_differentiable'])
        self.assertEqual(len(model.blocks),3)
        self.assertEqual(sum(isinstance(m,torch.nn.Conv3d) for m in model.cnn.modules()),8)
        self.assertTrue(all(not block.root_weight for block in model.blocks))
        for k,v in saved.items(): self.assertTrue(torch.equal(v,reference.state_dict()[k]))
        self.assertGreaterEqual(budget.calls,4)

    def test_approved_sizes_preserve_architecture_gradients_and_nested_selection(self):
        reference,profile,batch,kwargs=fixture()
        graphs,states={},{}
        for quota in (16,32,64):
            with self.subTest(context_nodes=quota*3):
                model=SparseFeatureL0(reference,replace(profile,context_nodes_per_band=quota)).eval()
                states[quota]={name:value.detach().clone() for name,value in model.state_dict().items()}
                output,graph=model(batch,return_graph=True,**kwargs)
                n=1+3*quota
                self.assertEqual(output.shape,(2,128))
                self.assertEqual(graph['features'].shape,(4,n,68))
                self.assertEqual(graph['adjacency'].shape,(4,n,n))
                self.assertEqual(graph['node_mask'].shape,(4,n))
                self.assertEqual(len(model.blocks),3)
                self.assertEqual(sum(isinstance(m,torch.nn.Conv3d) for m in model.cnn.modules()),8)
                self.assertTrue((graph['statistics']['band_pool_counts']>64).all())
                self.assertTrue((graph['statistics']['band_selected_counts']==quota).all())
                self.assertTrue(graph['statistics']['band_normalization_required'].all())
                self.assertTrue(graph['node_mask'].all())
                self.assertTrue(_connected(graph['adjacency'],graph['node_mask']).all())
                self.assertFalse(graph['adjacency'].diagonal(dim1=1,dim2=2).any())
                output.square().mean().backward()
                missing=[name for name,p in model.named_parameters() if p.grad is None or
                    not torch.isfinite(p.grad).all() or not p.grad.abs().sum()>0]
                self.assertEqual(missing,[])
                # Explicit UNIT optimizer step, never a training checkpoint.
                optimizer=torch.optim.SGD(model.parameters(),lr=.001)
                before=model.node_project.weight.detach().clone()
                optimizer.step()
                self.assertFalse(torch.equal(before,model.node_project.weight))
                graphs[quota]=graph['xyz_native'].detach().clone()
                del output,graph,model,optimizer
        for quota in (32,64):
            for name in states[16]: self.assertTrue(torch.equal(states[16][name],states[quota][name]),name)
        # Same pool/feature values, metric, seed and nontrivial band: larger
        # FPS quotas retain the smaller quota's ordered selection as a prefix.
        for small,large in ((16,32),(32,64),(16,64)):
            for band in range(3):
                first=graphs[small][:,1+band*small:1+(band+1)*small]
                prefix=graphs[large][:,1+band*large:1+band*large+small]
                self.assertTrue(torch.equal(first,prefix),(small,large,band))
        with self.assertRaisesRegex(ValueError,'16/32/64'):
            SparseFeatureL0(reference,replace(profile,context_nodes_per_band=48))

    def test_optional_coverage_pool_reuses_one_cnn_and_requires_graph(self):
        reference,profile,batch,kwargs=fixture()
        model=SparseFeatureL0(reference,profile).eval()
        with torch.no_grad(),patch.object(model.cnn,'forward',wraps=model.cnn.forward) as cnn:
            with self.assertRaisesRegex(ValueError,'return_pool requires return_graph'):
                model(batch,return_pool=True,**kwargs)
            self.assertEqual(cnn.call_count,0)
            _,graph=model(batch,return_graph=True,return_pool=True,**kwargs)
            self.assertEqual(cnn.call_count,1)
        pool=graph['coverage_pool']
        self.assertEqual(pool['features'].shape[:2],pool['eligible'].shape)
        self.assertEqual(pool['features'].shape[-1],68)
        self.assertEqual(pool['relative_mm'].shape,pool['eligible'].shape+(3,))
        self.assertEqual(pool['band'].shape,pool['eligible'].shape)
        self.assertEqual(pool['features'].device.type,'cuda')
        self.assertFalse(pool['features'].requires_grad)
        self.assertFalse(pool['relative_mm'].requires_grad)
        self.assertTrue(torch.equal(pool['eligible'].sum(-1),
            graph['statistics']['eligible_all_scale_pool_counts']))
        with torch.no_grad(): _,ordinary=model(batch,return_graph=True,**kwargs)
        self.assertNotIn('coverage_pool',ordinary)

    def test_outside_organ_and_padding_perturbation_cannot_change_feature_graph(self):
        reference,profile,batch,kwargs=fixture()
        model=SparseFeatureL0(reference,profile).eval()
        altered=LocalBatch(batch.images.clone(),batch.organ.clone(),batch.donor.clone(),batch.recipient.clone(),
            batch.indices.clone(),copy.deepcopy(batch.audit))
        altered.images[~altered.organ]=float('nan')
        altered.validate()
        with torch.no_grad():
            output,graph=model(batch,return_graph=True,**kwargs)
            other,other_graph=model(altered,return_graph=True,**kwargs)
        self.assertTrue(torch.equal(output,other))
        for key in ('features','xyz_native','adjacency','node_mask','edge_kind'):
            self.assertTrue(torch.equal(graph[key],other_graph[key]),key)

    def test_deterministic_initialization_selection_and_no_gt_input(self):
        reference,profile,batch,kwargs=fixture()
        rng=torch.random.get_rng_state().clone()
        cuda_rng=torch.cuda.get_rng_state().clone()
        model=SparseFeatureL0(reference,profile).eval()
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(cuda_rng,torch.cuda.get_rng_state()))
        second=SparseFeatureL0(reference,profile).eval()
        with torch.no_grad():
            first_out,first_graph=model(batch,return_graph=True,**kwargs)
            next_out,next_graph=second(batch,return_graph=True,**kwargs)
        self.assertTrue(torch.equal(first_out,next_out))
        self.assertTrue(torch.equal(first_graph['xyz_native'],next_graph['xyz_native']))
        # An unrelated GT annotation field is not read by this encoder.
        annotated=LocalBatch(batch.images,batch.organ,batch.donor,batch.recipient,batch.indices,
            [{**a,'UNIT_unused_tumor_ground_truth':i%2} for i,a in enumerate(batch.audit)]).validate()
        with torch.no_grad(): actual=model(annotated,**kwargs)
        self.assertTrue(torch.equal(first_out,actual))

    def test_small_band_padding_does_not_create_nodes_or_edges(self):
        reference,profile,batch,kwargs=fixture(narrow=True)
        model=SparseFeatureL0(reference,profile).eval()
        with torch.no_grad(): _,graph=model(batch,return_graph=True,**kwargs)
        count=graph['statistics']['band_pool_counts'][:,0]
        self.assertTrue((count==1).all())
        self.assertTrue(torch.equal(graph['statistics']['band_selected_counts'][:,0],count))
        expected=torch.arange(16,device='cuda')[None]<count[:,None]
        self.assertTrue(torch.equal(graph['node_mask'][:,1:17],expected))
        invalid=~graph['node_mask']
        self.assertFalse(graph['adjacency'][invalid].any())
        self.assertFalse(graph['adjacency'].transpose(1,2)[invalid].any())
        self.assertFalse(graph['features'][invalid].any())
        self.assertFalse(graph['statistics']['band_normalization_required'][:,0].any())
        self.assertFalse(graph['statistics']['band_feature_variance'][:,0].any())
        self.assertFalse(graph['statistics']['band_spatial_variance_mm2'][:,0].any())

    def test_empty_band_query_coverage_native_audit_and_budget_fail_explicitly(self):
        reference,profile,batch,kwargs=fixture()
        with self.assertRaisesRegex(SparseFeatureCoverageError,'Empty'):
            SparseFeatureL0(reference,replace(profile,mid_radius_mm=100.)).eval()(batch,**kwargs)
        bad_audit=copy.deepcopy(batch.audit)
        bad_audit[1]['shape']=[1,1,1]
        bad=LocalBatch(batch.images,batch.organ,batch.donor,batch.recipient,batch.indices,bad_audit).validate()
        with self.assertRaisesRegex(ValueError,'padding'):
            SparseFeatureL0(reference,profile).eval()(bad,**kwargs)
        with self.assertRaisesRegex(MemoryError,'budget'):
            SparseFeatureL0(reference,profile,Budget(fail=1)).eval()(batch,**kwargs)
        # Same valid crop, original anchor in a background hole: do not move it.
        hole=LocalBatch(batch.images.clone(),batch.organ.clone(),batch.donor,batch.recipient,batch.indices,copy.deepcopy(batch.audit))
        hole.organ[1,:,9:12,9:12,9:12]=False
        hole.validate()
        with self.assertRaisesRegex(SparseFeatureCoverageError,'anchor sphere'):
            SparseFeatureL0(reference,replace(profile,query_radius_mm=.4)).eval()(hole,**kwargs)

    def test_balanced_metric_features_change_selected_nodes_at_fixed_positions(self):
        # Explicit synthetic UNIT vectors: same physical coordinates, no CNN or
        # CT accuracy assertion. Feature pattern changes selection itself.
        profile=SparseFeatureProfile(16,3.,5.,10.)
        axis=torch.linspace(0,1,40,device='cuda')
        xyz=torch.stack((axis,axis.square(),torch.sin(axis*5)),dim=-1)[None].repeat(2,3,1)
        bands=torch.arange(1,4,device='cuda').repeat_interleave(40)[None].expand(2,-1)
        eligible=torch.ones((2,120),device='cuda',dtype=torch.bool)
        base=torch.stack((torch.cos(axis*4),torch.sin(axis*4),axis),-1).repeat(3,1)
        base=torch.nn.functional.pad(base,(0,65))[None].repeat(2,1,1)
        changed=base.clone()
        permutation=torch.cat((torch.arange(0,40,2,device='cuda'),torch.arange(1,40,2,device='cuda')))
        changed[:,40:80]=changed[:,40:80][:,permutation]
        distance=xyz.norm(dim=-1)
        before,mask,stats=_select_context(xyz,base,eligible,bands,distance,profile)
        after,other_mask,other_stats=_select_context(xyz,changed,eligible,bands,distance,profile)
        self.assertTrue(torch.equal(mask,other_mask))
        self.assertFalse(torch.equal(before[:,16:32],after[:,16:32]))
        self.assertTrue(torch.equal(before[:,:16],after[:,:16]))
        self.assertTrue(torch.equal(before[:,32:],after[:,32:]))
        self.assertTrue(torch.allclose(stats['band_feature_variance'],other_stats['band_feature_variance']))
        self.assertTrue(torch.equal(stats['band_spatial_variance_mm2'],other_stats['band_spatial_variance_mm2']))
        # Changing physical units alone cancels through measured dispersion.
        rescaled,_,scaled_stats=_select_context(xyz*100,base,eligible,bands,distance*100,profile)
        self.assertTrue(torch.equal(before,rescaled))
        self.assertTrue(torch.allclose(scaled_stats['band_spatial_variance_mm2'],
            stats['band_spatial_variance_mm2']*10000,rtol=1e-5))
        with self.assertRaisesRegex(FloatingPointError,'Zero or nonfinite'):
            _select_context(xyz,torch.ones_like(base),eligible,bands,distance,profile)
        with self.assertRaisesRegex(FloatingPointError,'Zero or nonfinite'):
            _select_context(torch.zeros_like(xyz),base,eligible,bands,distance,profile)
        invalid=base.clone();invalid[:,0]=float('nan')
        with self.assertRaisesRegex(FloatingPointError,'Zero or nonfinite'):
            _select_context(xyz,invalid,eligible,bands,distance,profile)

    def test_masked_corner_support_has_explicit_unavailable_scale(self):
        reference,_,batch,_=fixture()
        with torch.no_grad():
            maps=reference.cnn(batch.images,batch.organ)
            features,support=_masked_corner_features(maps,torch.tensor([[3.,10.,10.],[4.,10.,10.]],device='cuda'),
                torch.tensor([0,0],device='cuda'))
        self.assertEqual(features.shape,(2,68))
        self.assertTrue(support.all())
        tiny_masks=[torch.zeros_like(mask) for _,mask in maps]
        tiny_masks[0][0,0,3,10,10]=True
        with torch.no_grad():
            _,support=_masked_corner_features([(x,m) for (x,_),m in zip(maps,tiny_masks)],
                torch.tensor([[3.,10.,10.]],device='cuda'),torch.tensor([0],device='cuda'))
        self.assertEqual(support.tolist(),[[True,False,False]])

    def test_mst_is_explicit_and_root_path_occurs_once(self):
        xyz=torch.tensor([[[0.,0.,0.],[0.,1.,0.],[1.,0.,0.],[1.,1.,0.],
            [100.,0.,0.],[100.,1.,0.],[101.,0.,0.],[101.,1.,0.]]],device='cuda')
        features=torch.zeros((1,8,68),device='cuda')
        features[0,:4,0]=1;features[0,4:,1]=1
        mask=torch.ones((1,8),device='cuda',dtype=torch.bool)
        adjacency,kind,before,count=_build_edges(xyz,features,mask,3,1)
        self.assertFalse(before.any())
        self.assertTrue((kind & 4).any())
        self.assertEqual(count.tolist(),[1])
        self.assertEqual(((kind & 4)!=0).sum().item()//2,7)
        self.assertTrue(_connected(adjacency,mask).all())
        block=_MeanSAGE(128).cuda()
        block.neighbor.weight.data.zero_()
        x=torch.randn(1,8,128,device='cuda')
        actual=block(x,adjacency,mask)
        expected=torch.nn.functional.silu(block.norm(x))
        self.assertTrue(torch.equal(actual,expected))
        self.assertFalse(block.root_weight)
        self.assertFalse(any(name.startswith('root') for name,_ in block.named_parameters()))


if __name__ == '__main__': unittest.main()
