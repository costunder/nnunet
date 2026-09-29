"""Explicit synthetic DEBUG tests; not medical validity or production admission."""
import unittest
from unittest.mock import patch
import torch
from torch_geometric.data import HeteroData
from l0_ezsp.config import load_profile
from l0_ezsp.ops import mass_mean,quotient,undirected,mass_readout
from l0_ezsp.partition import Region,partition,CoarseningConstraintError
from l0_ezsp.encoder import EZSPEncoder,build_model
from l0_ezsp.data import collate
from hiercp_v22.schema import LOCAL_NODE_TYPES,LOCAL_EDGE_TYPES
from hiercp_v222.v1_cache import configuration

def synthetic():
    g=HeteroData();meta={}
    for k in LOCAL_NODE_TYPES:
        n=6
        g[k].grid=torch.zeros(n,3)
        g[k].num_nodes=n
        if 'context' in k:g[k].shell_id=torch.arange(n)//2
        meta[k]={'pos_mm':torch.arange(n)[:,None].expand(n,3).float()*.1,'stable_id':torch.arange(n)}
    for e in LOCAL_EDGE_TYPES:
        g[e].edge_index=torch.tensor([[0,1,2,3,4,5,0,2],[1,0,3,2,5,4,2,4]])
    g.sampled_counts=torch.tensor([[6]*len(LOCAL_NODE_TYPES)])
    g.relation_edge_counts=torch.tensor([[8]*len(LOCAL_EDGE_TYPES)])
    # Deliberately synthetic CT; never an actual-data claim.
    return g,torch.randn(1,48,48,48),torch.randn(1,48,48,48),meta

class CPU(unittest.TestCase):
    def test_old_resume_rejected(self):
        from l0_ezsp.identity import require_same_experiment,identity
        cfg=load_profile(reg_scale1=.1,reg_scale2=.1)
        with self.assertRaises(ValueError):require_same_experiment({'local_encoder':'v1_ct_only_context_pair'},cfg)
        require_same_experiment({'l0_ezsp_identity':identity(cfg)},cfg)
        _,base=configuration();encoder=EZSPEncoder(base,cfg)
        state=encoder.state_dict();encoder.load_state_dict(state)
        del state['_extra_state']
        with self.assertRaises(ValueError):encoder.load_state_dict(state,strict=False)

    def test_second_level_mass_and_boundary(self):
        x=torch.tensor([[1.],[3.],[9.],[11.]],requires_grad=True);p=torch.tensor([0,0,1,2])
        first,mass=mass_mean(x,p,torch.ones(4));second,mass2=mass_mean(first,torch.tensor([0,0,1]),mass)
        torch.testing.assert_close(second,torch.tensor([[13/3],[11.]]));self.assertEqual(mass2.tolist(),[3.,1.])
        e=torch.tensor([[0,0,1],[1,2,2]]);parent=torch.tensor([0,0,1])
        q,w=undirected(parent[e],torch.zeros(2,dtype=torch.long),torch.ones(3))
        self.assertEqual(q.tolist(),[[0],[1]]);self.assertEqual(w.tolist(),[2.])
    def test_required_reg(self):
        for v in (None,float('nan'),-1,True):
            with self.assertRaises(ValueError):load_profile(reg_scale1=v,reg_scale2=.1)

    def test_live_mass_and_gradient(self):
        x=torch.tensor([[1.],[3.],[9.]],requires_grad=True);m=torch.tensor([2.,1.,3.]);p=torch.tensor([0,0,1])
        pooled,mass=mass_mean(x,p,m)
        torch.testing.assert_close(pooled,torch.tensor([[5/3],[9.]]));torch.testing.assert_close(mass,torch.tensor([3.,3.]))
        pooled.sum().backward();torch.testing.assert_close(x.grad,torch.tensor([[2/3],[1/3],[1.]]))

    def test_typed_edges_and_witness(self):
        e=torch.tensor([[0,1,2,2],[0,1,0,0]]);p=torch.tensor([0,0,1])
        q,c,w,d=quotient(e,p,p,same_type=False)
        self.assertEqual(q.tolist(),[[0,1],[0,0]]);self.assertEqual(c.tolist(),[2,2]);self.assertEqual(int(d),0)
        torch.testing.assert_close(q,torch.stack((p[e[0,w]],p[e[1,w]])))
        q,c,w,d=quotient(e,p,p,same_type=True);self.assertEqual(int(d),2)
        adj,w=undirected(torch.tensor([[0,1,1,2],[1,0,2,1]]),torch.zeros(3,dtype=torch.long))
        self.assertEqual(w.tolist(),[1.,1.])

    def test_mass_attention(self):
        gate=torch.nn.Linear(1,1,bias=False);gate.weight.data.zero_()
        out=mass_readout(torch.tensor([[2.],[8.]]),torch.tensor([3.,1.]),torch.zeros(2,dtype=torch.long),gate,1)
        torch.testing.assert_close(out,torch.tensor([[3.5]]))

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required for official kernel')
class GPU(unittest.TestCase):
    def test_official_chain_groups_repeat_and_bounds(self):
        device='cuda';x=torch.ones(6,32,device=device)
        e=torch.tensor([[0,1,3,4],[1,2,4,5]],device=device)
        owner=torch.tensor([0,0,0,1,1,1],device=device);s=torch.full_like(owner,-1)
        pos=torch.arange(6,device=device).float()[:,None].expand(-1,3)*.1
        r=Region(owner,s,torch.ones(6,device=device),pos,pos,torch.arange(6,device=device),e,torch.ones(4,device=device))
        cfg=load_profile(reg_scale1=.1,reg_scale2=.1)
        state=torch.cuda.get_rng_state().clone();a=partition(x,r,role='tumor_surface',level=1,cfg=cfg)
        self.assertTrue(torch.equal(state,torch.cuda.get_rng_state()))
        self.assertEqual(sorted(a.region.mass.tolist()),[3.,3.]);self.assertFalse(a.violations)
        b=partition(x,r,role='tumor_surface',level=1,cfg=cfg)
        self.assertTrue(torch.equal(a.parent[:,None]==a.parent[None],b.parent[:,None]==b.parent[None]))
        perm=torch.tensor([2,0,1,5,3,4],device=device);inv=perm.argsort()
        rp=Region(owner[perm],s[perm],r.mass[perm],pos[perm],pos[perm],r.stable_id[perm],inv[e],r.boundary)
        c=partition(x[perm],rp,role='tumor_surface',level=1,cfg=cfg);cp=c.parent[inv]
        self.assertTrue(torch.equal(a.parent[:,None]==a.parent[None],cp[:,None]==cp[None]))
        # Same first component alone versus disjoint-union batch.
        single=Region(owner[:3],s[:3],r.mass[:3],pos[:3],pos[:3],r.stable_id[:3],e[:,:2],r.boundary[:2])
        one=partition(x[:3],single,role='tumor_surface',level=1,cfg=cfg)
        self.assertEqual(one.region.mass.tolist(),[3.])
        r.upper=r.upper+20
        rejected=partition(x,r,role='tumor_surface',level=1,cfg=cfg)
        self.assertIn('bbox',rejected.violations)

    def test_shell_split_isolated_and_empty(self):
        x=torch.ones(5,32,device='cuda');owner=torch.zeros(5,dtype=torch.long,device='cuda')
        shell=torch.tensor([0,0,1,1,2],device='cuda');pos=torch.zeros(5,3,device='cuda')
        e,w=undirected(torch.tensor([[0,1,1,2,3],[1,0,2,3,2]],device='cuda'),owner*4+shell+1)
        r=Region(owner,shell,torch.ones(5,device='cuda'),pos,pos,torch.arange(5,device='cuda'),e,w)
        p=partition(x,r,role='source_context',level=1,cfg=load_profile(reg_scale1=.1,reg_scale2=.1))
        self.assertEqual(sorted(p.region.mass.tolist()),[1.,2.,2.]);self.assertFalse(p.violations)
        empty=Region(owner[:0],shell[:0],r.mass[:0],pos[:0],pos[:0],r.stable_id[:0],e[:,:0],w[:0])
        p=partition(x[:0],empty,role='target_context',level=1,cfg=load_profile(reg_scale1=.1,reg_scale2=.1))
        self.assertEqual(p.stats['output_nodes'],0)

    def test_full_encoder_gradient_and_no_backward_repartition(self):
        torch.manual_seed(7);cfg,base=configuration()
        net=build_model(cfg,base,load_profile(reg_scale1=.1,reg_scale2=.1)).cuda().train()
        batch=collate([(synthetic(),0),(synthetic(),1)]).to('cuda')
        from l0_ezsp import encoder
        with patch.object(encoder,'partition',wraps=partition) as counted:
            # Real L0 -> unchanged L1/L2 -> unchanged supervised loss.
            support=net.local(batch)
            owners=torch.tensor([0,1],device='cuda');classes=torch.tensor([0,1],device='cuda')
            out=net.predict_embeddings(support,net.prepare_support(support,owners,classes))
            from hiercp_v222.model import supervised_loss
            loss=supervised_loss(out,classes)
            forward_calls=counted.call_count;loss.backward()
            self.assertEqual(counted.call_count,forward_calls);self.assertEqual(forward_calls,10)
        self.assertEqual(tuple(support.shape),(2,128))
        for prefix in ('local.dense_encoder','local.blocks.0','local.blocks.1','local.blocks.2','l1','l2'):
            grads=[p.grad for name,p in net.named_parameters() if name.startswith(prefix)]
            self.assertTrue(grads and any(g is not None and bool(torch.isfinite(g).all()) and bool((g!=0).any()) for g in grads),prefix)
        optimizer=torch.optim.Adam(net.parameters(),lr=1e-4)
        before=next(net.local.dense_encoder.parameters()).detach().clone();optimizer.step()
        self.assertFalse(torch.equal(before,next(net.local.dense_encoder.parameters())))
        # Admission happens before any GAT, and never silently falls back.
        batch.sidecar['tumor_surface']['pos_mm']*=10000
        with patch.object(net.local.blocks[0],'forward',wraps=net.local.blocks[0].forward) as block:
            with self.assertRaises(CoarseningConstraintError):net.local(batch)
            self.assertEqual(block.call_count,0)

if __name__=='__main__':unittest.main(verbosity=2)
