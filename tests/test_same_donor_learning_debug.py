"""Synthetic operator/contract tests only; not CT performance measurements."""
import unittest
from types import SimpleNamespace
from collections import Counter
from unittest.mock import patch
import torch
from torch.nn import functional as F
from l0_regions.donor_learning import groups,LiveContext,forward_loss,configuration,selection
from l0_regions.ram_pressure import trim,guard
from tools.v222_runtime_cache import TensorCache

def rows(p,n,case='a'):
    return [dict(id=f'{case}:{i}',case_id=case,patient_group=case,donor_case_id='donor',donor_component=1,
        donor_group='donor',target=int(i<p),bounds=dict(edges=i+1)) for i in range(p+n)]

class Checks(unittest.TestCase):
    def test_assignment_class_blind_train_only_and_patient_exclusion(self):
        from l0_regions.donor_data import assignment
        rs=rows(2,4);rs=[dict(r,component=None,center=[i,2,3]) for i,r in enumerate(rs)]
        meta=dict(records=rs,donor_pool=[dict(case_id='a',component_id=1),dict(case_id='donor',component_id=2)],
            identities=dict(cases={'a':dict(patient_group='a'),'donor':dict(patient_group='donor')}),split=dict(inner_train=['a','donor']))
        first=assignment(meta,42)
        self.assertEqual({r['donor_case_id'] for r in first},{'donor'})
        for r in meta['records']:r['target']=1-r['target']
        second=assignment(meta,42)
        self.assertEqual([(r['donor_case_id'],r['donor_component']) for r in first],[(r['donor_case_id'],r['donor_component']) for r in second])
        meta['split']['inner_train']=['a']
        with self.assertRaisesRegex(ValueError,'Train-only'):assignment(meta,42)

    def test_full_pair_coverage_with_physical_32_and_zero_positive_case(self):
        rs=rows(21,128)+rows(0,128,'b');ds=SimpleNamespace(rows=rs)
        order=list(groups(ds,32,42,0));uses=Counter(i for ids in order for i in ids)
        pairs=Counter((p,u) for ids in order for p in ids if rs[p]['target'] for u in ids if not rs[u]['target'])
        self.assertEqual(set(uses),set(range(len(rs))))
        self.assertEqual(set(pairs),{(p,u) for p in range(21) for u in range(21,149)})
        self.assertEqual(set(pairs.values()),{1});self.assertEqual(max(map(len,order)),32)
        self.assertTrue(all(len(ids)<=32 for ids in order))
        self.assertEqual(Counter(map(tuple,order)),Counter(map(tuple,groups(ds,32,42,1))))

    def test_different_donor_rejected(self):
        rs=rows(2,4);rs[1]['donor_component']=2
        with self.assertRaisesRegex(ValueError,'different donors'):list(groups(SimpleNamespace(rows=rs),4))

    def test_tiled_gradient_equals_global_objective_and_live_both_sides(self):
        rs=rows(3,9)+rows(0,4,'b');ds=SimpleNamespace(rows=rs);ctx=LiveContext(ds,4)
        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            torch.manual_seed(42);x=torch.randn(len(rs),2,device=device,requires_grad=True)
            class Net:
                def local(self,q):return q
                def prepare_support(self,*args,**kw):return None
                def predict_embeddings(self,e,state):return dict(logits=e,alignment_loss_weight=0.,alignment_loss=e.sum()*0)
            losses=[]
            for ids in ctx.order:
                y=torch.tensor([rs[i]['target'] for i in ids],device=device)
                loss,_=forward_loss(Net(),x[ids],(),None,y,None,ctx,configuration(),indices=ids);losses.append(loss)
            tiled=torch.stack(losses).mean();score=x[:,1]-x[:,0]
            truth=torch.tensor([r['target'] for r in rs],device=device)
            fullrank=F.softplus(-(score[:3,None]-score[None,3:12])).mean()
            ce=F.cross_entropy(x,truth,reduction='none');full=fullrank+.5*ce[truth==1].mean()+.5*ce[truth==0].mean()
            ga=torch.autograd.grad(tiled,x,retain_graph=True)[0];gb=torch.autograd.grad(full,x)[0]
            torch.testing.assert_close(tiled,full);torch.testing.assert_close(ga,gb)
            self.assertTrue(bool((ga.norm(dim=1)>0).all()))

    def test_best_prefers_mrr_even_with_lower_pairwise_loss(self):
        a=dict(ranking_mrr=.189,ranking_recall_at_1=.007,ranking_pairwise_loss=.6903)
        b=dict(ranking_mrr=.178,ranking_recall_at_1=.007,ranking_pairwise_loss=.6871)
        self.assertGreater(selection(a),selection(b))

    def test_cache_reclaim_preserves_alias_accounting_and_live_objects(self):
        cache=TensorCache(4096);x=torch.arange(16.)
        cache.get(('file',1),lambda:x);cache.get(('same_donor_bound_batch',1),lambda:x)
        self.assertEqual(cache.bytes,64)
        trim(cache,prefixes=('same_donor_bound_batch',))
        self.assertEqual(cache.bytes,64);self.assertTrue(torch.equal(x,torch.arange(16.)))
        trim(cache,bytes_to_release=64);self.assertEqual(cache.bytes,0);self.assertFalse(cache.references)

    def test_rss_guard_reclaims_and_still_rejects_unrecoverable_pressure(self):
        cache=TensorCache(4096);cache.get(('file',1),lambda:torch.ones(16))
        with patch('l0_regions.ram_pressure.psutil.Process') as process:
            process.return_value.memory_info.return_value.rss=5000
            report=guard(cache,8192);self.assertEqual(report['evicted'],1)
            process.return_value.memory_info.return_value.rss=9000
            with self.assertRaisesRegex(MemoryError,'after cache reclamation'):guard(cache,8192)

if __name__=='__main__':unittest.main(verbosity=2)
