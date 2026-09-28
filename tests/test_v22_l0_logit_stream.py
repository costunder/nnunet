"""DEBUG numerical derivative/parity checks; not medical performance evidence."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import unittest
from contextlib import nullcontext
from unittest.mock import patch
import torch
from torch.nn import functional as F
from tools.v22_l0_logit_stream import EdgeLogit, FusedEdgeLogit, NodeCastAggregation, installed
from hiercp.model import CompatibilityGatedGATv2Conv


class LogitTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_fused_precision_zero_boundary_and_empty_edges(self):
        for dtype in (torch.float32,torch.bfloat16,torch.float64):
            for count in (0,17):
                left=torch.randn(7,4,32,device='cuda',dtype=dtype)
                right=-left[:5].clone()
                edge=torch.stack((torch.arange(count,device='cuda')%5,torch.arange(count,device='cuda')%5))
                att=torch.randn(1,4,32,device='cuda',dtype=torch.float64 if dtype==torch.float64 else torch.float32)
                expected=[]
                for fused in (False,True):
                    a=left.detach().requires_grad_();b=right.detach().requires_grad_();c=att.detach().requires_grad_()
                    if fused:y=FusedEdgeLogit.apply(a,b,edge,c,.2)
                    else:
                        arithmetic=torch.float64 if dtype==torch.float64 else torch.float32
                        joint=b[edge[1]].to(arithmetic)+a[edge[0]].to(arithmetic)
                        y=(F.leaky_relu(joint,.2)*c).sum(-1)
                    y.sum().backward();expected.append((y.detach(),a.grad,b.grad,c.grad))
                for a,b in zip(*expected):self.assertTrue(torch.equal(a,b),(dtype,count))

    def test_aggregation_cast_hoist_double_derivative_and_no_edge_hidden_saved(self):
        edge=torch.tensor([[0,1,0,1,0],[0,1,1,0,0]])
        features=torch.randn(2,2,3,dtype=torch.float64,requires_grad=True)
        attention=torch.randn(5,2,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda f,a:NodeCastAggregation.apply(f,edge,a,2,2),
                                               (features,attention)))
        seen=[]
        with torch.autograd.graph.saved_tensors_hooks(lambda t:(seen.append(tuple(t.shape)) or t),lambda t:t):
            EdgeLogit.apply(features,features,edge,torch.randn(1,2,3,requires_grad=True),.2).sum().backward()
        self.assertNotIn((5,2,3),seen)

    def test_install_restores_after_exception_and_rejects_attributes(self):
        net=CompatibilityGatedGATv2Conv((8,8),4,heads=2,add_self_loops=False)
        original=net._recompute_edge_chunk
        original_forward=net.forward
        import hiercp.model as implementation
        original_aggregation=implementation._StreamedEdgeAggregation
        with self.assertRaisesRegex(RuntimeError,'test'):
            with installed(net,node_cast=True):
                self.assertIs(implementation._StreamedEdgeAggregation,original_aggregation)
                self.assertIs(net.forward.__func__.__globals__['_StreamedEdgeAggregation'],NodeCastAggregation)
                raise RuntimeError('test')
        self.assertEqual(net._recompute_edge_chunk,original)
        self.assertEqual(net.forward,original_forward)
        invalid=CompatibilityGatedGATv2Conv((8,8),4,heads=2,edge_dim=3,add_self_loops=False)
        with self.assertRaisesRegex(ValueError,'no edge attributes'):
            with installed(invalid):pass

    def test_double_numerical_derivative_duplicates_and_empty(self):
        edges = torch.tensor([[0,1,0,1,1],[0,1,1,0,0]])
        args = (torch.randn(2,2,3,dtype=torch.float64,requires_grad=True),
                torch.randn(2,2,3,dtype=torch.float64,requires_grad=True),
                torch.randn(1,2,3,dtype=torch.float64,requires_grad=True))
        self.assertTrue(torch.autograd.gradcheck(lambda a,b,c: EdgeLogit.apply(a,b,edges,c,.2),args))
        y = EdgeLogit.apply(args[0],args[1],edges[:,:0],args[2],.2)
        y.sum().backward()
        for t in args:self.assertTrue(torch.equal(t.grad,torch.zeros_like(t)))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_bf16_multichunk_dropout_all_gradients_exact(self):
        previous=torch.are_deterministic_algorithms_enabled();torch.use_deterministic_algorithms(True)
        try:
            torch.manual_seed(52)
            net=CompatibilityGatedGATv2Conv((128,128),32,heads=4,add_self_loops=False,dropout=.2).cuda()
            x=torch.randn(37,128,device='cuda');z=torch.randn(29,128,device='cuda')
            edges=torch.randint(29,(2,171),device='cuda');outputs=[]
            rng=torch.cuda.get_rng_state()
            with patch('hiercp.model.EDGE_ATTENTION_WORKSPACE_BYTES',8*4*128*31):
                for candidate in ('baseline','direct','fused','combined'):
                    net.zero_grad(set_to_none=True);torch.cuda.set_rng_state(rng)
                    a=x.detach().requires_grad_();b=z.detach().requires_grad_()
                    with installed(net, fused=candidate in ('fused','combined'),node_cast=candidate=='combined') if candidate!='baseline' else nullcontext():
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            y=net((a,b),edges);loss=y.float().square().mean()
                        loss.backward()
                    outputs.append([y.detach(),a.grad,b.grad,*[p.grad.clone() for p in net.parameters()],torch.cuda.get_rng_state()])
            for variant in outputs[1:]:
                for i,(a,b) in enumerate(zip(outputs[0],variant)):
                    self.assertTrue(torch.equal(a,b),f'output/gradient {i}: max diff {(a.float()-b.float()).abs().max()}')
        finally:torch.use_deterministic_algorithms(previous)


if __name__=='__main__':unittest.main()
