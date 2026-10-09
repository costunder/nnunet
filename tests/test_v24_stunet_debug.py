"""Synthetic UNIT checks; no claims about actual CT ranking performance."""
import ast
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, HeteroData

from hiercp_v1x.v24_stunet import (
    STUNetSmallEncoder, STUNetL0Encoder, FeaturePyramid,
    load_encoder_strict, sample_feature_scale, CHANNELS,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'work/pretrained_l0_latency_DEBUG_20261008/assets'


class STUSamplingDebug(unittest.TestCase):
    def test_full_batched_sampler_values_and_gradient_match_grid_sample(self):
        torch.manual_seed(811)
        a = torch.randn(2, 3, 3, 4, 5, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        grid = torch.tensor([[-1., -1., -1.], [.23,.47,-.11], [1.,1.,1.],
                             [-.2,.12,.44], [1.,-.4,.7]])
        owners = torch.tensor([0, 1, 0, 1, 1])
        input_shape, stride = (7, 9, 11), (2,2,2)
        actual = sample_feature_scale(a, grid, owners, input_shape=input_shape,stride=stride)
        xyz = (grid+1)/2 * torch.tensor([10.,8.,6.]) / 2
        normalized = 2*xyz/torch.tensor([4.,3.,2.])-1
        reference = torch.stack([F.grid_sample(b[owner:owner+1],point.reshape(1,1,1,1,3),
             align_corners=True,padding_mode='border').reshape(3)
             for owner,point in zip(owners.tolist(),normalized)])
        torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-6)
        weights = torch.randn_like(actual)
        (actual*weights).sum().backward(); (reference*weights).sum().backward()
        torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-6)

    def test_singleton_feature_axes_and_repeated_ownership(self):
        a = torch.tensor([[[[[2.,4.]]]]],requires_grad=True)
        grid = torch.tensor([[-1.,0.,1.],[1.,-1.,0.]])
        out = sample_feature_scale(a,grid,torch.zeros(2,dtype=torch.long),
                                   input_shape=(16,16,32),stride=(16,16,32))
        torch.testing.assert_close(out[:,0],torch.tensor([2.,3.9375]))
        out.sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all())


@unittest.skipUnless((ASSETS/'small_ep4k.model').is_file(),'Pinned official checkpoint fixture absent')
class STUEncoderDebug(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_strict_complete_encoder_and_official_feature_parity(self):
        encoder = STUNetSmallEncoder().eval()
        proof = load_encoder_strict(encoder,ASSETS/'small_ep4k.model')
        self.assertEqual(proof['loaded_parameter_numel'],7185744)
        self.assertEqual(proof['loaded_encoder_tensors'],60)
        self.assertEqual(len(proof['ignored_decoder_tensors']),70)
        source = (ASSETS/'stunet_medim_source.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        selected = [node for node in tree.body if isinstance(node,ast.ClassDef)
                    and node.name in ('BasicResBlock','STUNet','Upsample_Layer_nearest','Decoder')]
        selected += [node for node in tree.body if isinstance(node,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='DEFAULT_STRIDES' for t in node.targets)]
        namespace = {'torch':torch,'nn':torch.nn}
        exec(compile(ast.Module(body=selected,type_ignores=[]),'official_stu_UNIT','exec'),namespace)
        official = namespace['STUNet'](input_channels=1,num_classes=105,
            depth=[1]*6,dims=list(CHANNELS),
            pool_op_kernel_sizes=[[2,2,2]]*4+[[1,1,2]],
            conv_kernel_sizes=[[3,3,3]]*6).eval()
        official.conv_blocks_context.load_state_dict(encoder.conv_blocks_context.state_dict(),strict=True)
        inp = torch.randn(2,1,32,32,32)
        with torch.no_grad():
            actual = encoder(inp); x=inp; expected=[]
            for stage in official.conv_blocks_context:
                x=stage(x); expected.append(x)
        self.assertEqual(len(actual),6)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_semantic_readout_and_all_encoder_stages_receive_gradient(self):
        from hiercp.model import LocalTumorContextPyGEncoder
        from hiercp.schema import LOCAL_NODE_TYPES,LOCAL_HANDCRAFTED_DIM,CONTEXT_SHELL_FEATURE_INDEX
        original = LocalTumorContextPyGEncoder(hidden_dim=128,heads=4,layers=3,dropout=.1,
            dense_base_channels=16,dense_feature_dim=32,dense_batch_size=4,
            channels_last_3d=False,checkpoint_local_blocks=True,checkpoint_dense_encoder=True)
        local = STUNetL0Encoder(original,ASSETS/'small_ep4k.model').train()
        graphs=[]
        for graph_id in range(2):
            graph=HeteroData()
            for name in LOCAL_NODE_TYPES:
                graph[name].x=torch.randn(6,LOCAL_HANDCRAFTED_DIM)
                if name in ('source_context','target_context'):
                    graph[name].x[:,CONTEXT_SHELL_FEATURE_INDEX]=torch.tensor([0.,1/3,2/3]*2)
                graph[name].grid=torch.rand(6,3)*2-1
            graphs.append(graph)
        batch=Batch.from_data_list(graphs)
        source=torch.randn(1,5,32,32,32)
        target=torch.randn(2,5,32,32,32)
        source_maps,target_maps=local.encode_dense_maps(source,torch.tensor([0,0]),target)
        self.assertIsInstance(source_maps,FeaturePyramid)
        outputs=local.forward_graph(batch,source_maps,target_maps)
        self.assertEqual(len(outputs),12)
        self.assertTrue(all(tuple(x.shape)==(2,128) for x in outputs.values()))
        loss=sum((value*torch.randn_like(value)).mean() for value in outputs.values())
        loss.backward()
        for name,parameter in local.named_parameters():
            self.assertIsNotNone(parameter.grad,name)
            self.assertTrue(torch.isfinite(parameter.grad).all(),name)
        for stage in local.dense_encoder.conv_blocks_context:
            self.assertGreater(sum(p.grad.abs().sum().item() for p in stage.parameters()),0.)
        before=local.dense_encoder.conv_blocks_context[0][0].conv1.weight.detach().clone()
        optimizer=torch.optim.AdamW(local.parameters(),lr=1e-4)
        optimizer.step()
        self.assertFalse(torch.equal(before,local.dense_encoder.conv_blocks_context[0][0].conv1.weight))
        names=list(dict(local.named_parameters()))
        self.assertFalse(any(name.startswith('blocks.') or 'seg_outputs' in name for name in names))


if __name__=='__main__':
    unittest.main()
