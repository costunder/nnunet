"""Mathematical and integration DEBUG; not CP performance tests."""
import unittest
import torch
from torch_geometric.nn import SAGEConv
from l0_sage.encoder import mean_adjacency,GraphSAGEEncoder
from l0_ezsp.data import collate
from tests.test_l0_ezsp_debug import synthetic
from hiercp_v222.v1_cache import configuration
from hiercp_v222.v1_local import V1LocalEncoder
from hiercp_v222.model import PromptGraphModel,supervised_loss
from tools.v222_review_contracts import installed

class Operator(unittest.TestCase):
    def test_relation_roots_and_block_residual_match_explicit_formula(self):
        from collections import Counter
        from hiercp_v22.schema import LOCAL_NODE_TYPES, LOCAL_EDGE_TYPES
        _,base=configuration()
        with installed('stride4'):reference=V1LocalEncoder(base)
        net=GraphSAGEEncoder(reference,seed=42).eval()
        batch=collate([(synthetic(),0)])
        net.adjacency.prepare(batch.graph)
        block=net.encoder.blocks[0]
        x={k:torch.randn(batch.graph[k].num_nodes,128,requires_grad=True) for k in LOCAL_NODE_TYPES}
        edges={e:batch.graph[e].edge_index for e in LOCAL_EDGE_TYPES}
        actual=block(x,edges,{e:None for e in edges})
        expected_messages={k:torch.zeros_like(v) for k,v in x.items()}
        roots=[]
        for e in LOCAL_EDGE_TYPES:
            conv=block.conv.convs[e];source,target=x[e[0]],x[e[2]]
            index=edges[e]
            degree=torch.bincount(index[1],minlength=len(target)).clamp_min(1).to(target.dtype)
            neighbor=torch.zeros_like(target).index_add(0,index[1],source[index[0]])/degree[:,None]
            expected_messages[e[2]]=expected_messages[e[2]]+conv.lin_l(neighbor)+conv.lin_r(target)
            roots.append(conv.lin_r.weight)
        expected={}
        for k in LOCAL_NODE_TYPES:
            # Exactly one explicit block residual, plus one distinct root transform
            # per incoming relation. They are not one shared transform multiplied.
            h=block.message_norm[k](x[k]+expected_messages[k])
            expected[k]=block.ffn_norm[k](h+block.ffn[k](h))
            torch.testing.assert_close(actual[k],expected[k],rtol=2e-5,atol=2e-6)
        inputs=(*x.values(),*roots)
        probes={k:torch.randn_like(v) for k,v in actual.items()}
        ga=torch.autograd.grad(sum((v*probes[k]).sum() for k,v in actual.items()),inputs)
        gb=torch.autograd.grad(sum((v*probes[k]).sum() for k,v in expected.items()),inputs)
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=3e-5,atol=1e-5)
        self.assertEqual(dict(Counter(e[2] for e in LOCAL_EDGE_TYPES)),
            dict(tumor_surface=2,source_context=3,source_liver_surface=2,target_context=4,target_liver_surface=2))
        self.assertEqual(len({id(w) for w in roots}),13)

    def check_operator(self,device):
        torch.manual_seed(4)
        edge=torch.tensor([[0,0,1,2],[0,0,0,1]],device=device)
        xs=torch.randn(3,8,device=device,requires_grad=True);xd=torch.randn(4,8,device=device,requires_grad=True)
        mean=SAGEConv((8,8),8,aggr='mean').to(device)
        sparse=SAGEConv((8,8),8,aggr='sum').to(device);sparse.load_state_dict(mean.state_dict())
        matrix,degree=mean_adjacency(edge,3,4)
        a=mean((xs,xd),edge);ga=torch.autograd.grad(a.square().sum(),(xs,xd,*mean.parameters()))
        b=sparse((xs,xd),matrix);gb=torch.autograd.grad(b.square().sum(),(xs,xd,*sparse.parameters()))
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        for l,r in zip(ga,gb):torch.testing.assert_close(l,r,rtol=1e-5,atol=2e-6)
        self.assertEqual(degree.tolist(),[3.,1.,0.,0.])
        empty,_=mean_adjacency(edge[:,:0],3,4)
        torch.testing.assert_close(mean((xs,xd),edge[:,:0]),sparse((xs,xd),empty))

    def test_cpu_mean_forward_all_gradients(self):self.check_operator('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_mean_forward_all_gradients(self):self.check_operator('cuda')

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class Integration(unittest.TestCase):
    def test_unchanged_weights_coverage_cache_gradient_optimizer(self):
        cfg,base=configuration()
        with installed('stride4'):reference=V1LocalEncoder(base)
        candidate=GraphSAGEEncoder(reference,seed=42)
        original=dict(reference.named_parameters())
        for name,p in candidate.encoder.named_parameters():
            if not (name.startswith('blocks.') and '.conv.' in name):self.assertTrue(torch.equal(p,original[name]),name)
        net=PromptGraphModel(cfg,base,{},local_encoder=candidate).cuda().train()
        batch=collate([(synthetic(),0),(synthetic(),1)]).to('cuda')
        edges={k:batch.graph[k].edge_index.clone() for k in batch.graph.edge_types}
        embeddings=net.local(batch)
        owners=torch.tensor([0,1],device='cuda');classes=owners.clone()
        output=net.predict_embeddings(embeddings,net.prepare_support(embeddings,owners,classes))
        loss=supervised_loss(output,classes);loss.backward()
        self.assertEqual(candidate.adjacency.builds,1)
        for k in edges:torch.testing.assert_close(edges[k],batch.graph[k].edge_index)
        for prefix in ('local.encoder.dense_encoder','local.encoder.blocks.0.conv','local.encoder.blocks.1.conv','local.encoder.blocks.2.conv','l1','l2'):
            grads=[p.grad for n,p in net.named_parameters() if n.startswith(prefix)]
            self.assertTrue(grads and all(g is not None and torch.isfinite(g).all() for g in grads),prefix)
            self.assertTrue(any(bool((g!=0).any()) for g in grads),prefix)
        before=next(candidate.encoder.blocks[0].conv.parameters()).detach().clone()
        torch.optim.AdamW(net.parameters(),lr=1e-4).step()
        self.assertFalse(torch.equal(before,next(candidate.encoder.blocks[0].conv.parameters())))
        with torch.no_grad():net.local(batch)
        self.assertEqual(candidate.adjacency.builds,1)
        with self.assertRaises(RuntimeError):candidate.state_dict()

    def test_missing_coverage_rejected(self):
        _,base=configuration()
        with installed('stride4'):reference=V1LocalEncoder(base)
        net=GraphSAGEEncoder(reference,seed=42).cuda()
        batch=collate([(synthetic(),0)]).to('cuda');del batch.graph.sampled_counts
        with self.assertRaises(ValueError):net(batch)

    def test_empty_target_context_and_single_batch_equivalence(self):
        _,base=configuration()
        with installed('stride4'):reference=V1LocalEncoder(base)
        net=GraphSAGEEncoder(reference,seed=42).cuda().eval()
        item=synthetic();g=item[0];k='target_context'
        g[k].grid=g[k].grid[:0];g[k].shell_id=g[k].shell_id[:0];g[k].num_nodes=0
        for edge in g.edge_types:
            if edge[0]==k or edge[2]==k:g[edge].edge_index=g[edge].edge_index[:,:0]
        from hiercp_v22.schema import LOCAL_NODE_TYPES,LOCAL_EDGE_TYPES
        g.sampled_counts=torch.tensor([[g[t].num_nodes for t in LOCAL_NODE_TYPES]])
        g.relation_edge_counts=torch.tensor([[g[e].edge_index.shape[1] for e in LOCAL_EDGE_TYPES]])
        item[3][k]={name:t[:0] for name,t in item[3][k].items()}
        with torch.no_grad():
            one=net(collate([(item,0)]).to('cuda'))
            two=net(collate([(item,0),(item,1)]).to('cuda'))
        torch.testing.assert_close(two,one.expand(2,-1),rtol=1e-4,atol=1e-5)

    def test_cache_invalidation_and_rng_preservation(self):
        _,base=configuration()
        with installed('stride4'):reference=V1LocalEncoder(base)
        cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
        net=GraphSAGEEncoder(reference,seed=42).cuda().eval()
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()));self.assertTrue(torch.equal(gpu,torch.cuda.get_rng_state()))
        batch=collate([(synthetic(),0)]).to('cuda')
        with torch.no_grad():net(batch)
        edge=batch.graph.edge_types[0]
        batch.graph[edge].edge_index[0,0]=2
        with torch.no_grad():net(batch)
        self.assertEqual(net.adjacency.builds,2)

if __name__=='__main__':unittest.main(verbosity=2)
