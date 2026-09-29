"""Single-scale CUDA plumbing, topology preservation and gradient regressions."""
import copy
import unittest
from unittest.mock import patch
import torch
from tests import test_l0_regions_debug as legacy
from l0_regions.preparation import single_profile,prepare,OfflinePartition
from l0_regions.data import collate,validate_item,seal_item
from l0_regions.encoder import RegionSAGEEncoder
from hiercp_v222.model import PromptGraphModel,supervised_loss


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class SingleScale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        legacy.FixedHierarchy.setUpClass()
        f=legacy.FixedHierarchy;cls.f=f
        cls.profile=single_profile(.02)
        bindings=copy.deepcopy(f.bindings)
        for b in bindings:b['profile']=cls.profile;b['scale2_features']='REMOVED'
        cls.calls=[];original=OfflinePartition._coarsen
        def counted(self,x,regions,edges,level,count):
            cls.calls.append(level)
            return original(self,x,regions,edges,level,count)
        with patch.object(OfflinePartition,'_coarsen',counted):
            cls.items,cls.audit=prepare(f.fine,copy.deepcopy(f.reference).cuda().eval().requires_grad_(False),
                cls.profile,bindings,f.budget,allow_unvalidated_profile=True)

    def test_no_second_partition_or_parent(self):
        self.assertEqual(self.calls,[1])
        self.assertIsNone(self.profile['partition']['reg_scale2'])
        self.assertEqual(self.audit['scale2'],{'status':'REMOVED_BY_DESIGN'})
        for item in self.items:
            self.assertEqual(len(item['scales']),1)
            self.assertEqual(len(item['edges']),1)
            self.assertTrue(all('parent' not in r for r in item['scales'][0].values()))

    def test_first_partition_and_relations_unchanged(self):
        for old,new in zip(self.f.items,self.items):
            for role in old['fine']:
                torch.testing.assert_close(old['fine'][role]['parent'],new['fine'][role]['parent'],rtol=0,atol=0)
                for key in new['scales'][0][role]:
                    torch.testing.assert_close(old['scales'][0][role][key],new['scales'][0][role][key],rtol=0,atol=0)
            for rel in old['edges'][0]:
                self.assertTrue(torch.equal(old['edges'][0][rel],new['edges'][0][rel]))

    def test_unused_second_assignment_rejected(self):
        item=copy.deepcopy(self.items[0])
        item['scales'][0]['tumor_surface']['parent']=torch.zeros_like(item['scales'][0]['tumor_surface']['shell'])
        seal_item(item)
        with self.assertRaisesRegex(ValueError,'unused parent'):validate_item(item)

    def test_compact_audit_cannot_hide_missing_groups_or_mutation(self):
        item=copy.deepcopy(self.items[0])
        role=next(r for r,s in item['audit']['scale1']['roles'].items() if s['group_diagnostics'])
        item['audit']['scale1']['roles'][role]['group_diagnostics'].clear()
        with self.assertRaisesRegex(ValueError,'receipt changed'):validate_item(item)
        seal_item(item)
        with self.assertRaisesRegex(ValueError,'cluster coverage'):validate_item(item)

    def test_all_three_sage_layers_and_cnn_update(self):
        local=RegionSAGEEncoder(self.f.reference,seed=42,resource_budget=self.f.budget,
            allow_unvalidated_profile=True,levels=1).cuda()
        net=PromptGraphModel(self.f.cfg,self.f.base,{},local_encoder=local).cuda().train()
        batch=collate(self.items,[0,1]).to('cuda')
        before=next(local.dense_encoder.parameters()).detach().clone()
        with patch('l0_ezsp.encoder.partition',side_effect=AssertionError('No partition during training')):
            embeddings=local(batch);self.assertEqual(tuple(embeddings.shape),(2,128))
            classes=torch.tensor([0,1],device='cuda')
            out=net.predict_embeddings(embeddings,net.prepare_support(embeddings,classes,classes))
            supervised_loss(out,classes).backward()
        for prefix in ('local.core.dense_encoder','local.core.blocks.0.conv','local.core.blocks.1.conv','local.core.blocks.2.conv','l1','l2'):
            grads=[p.grad for n,p in net.named_parameters() if n.startswith(prefix)]
            self.assertTrue(grads and all(g is not None and torch.isfinite(g).all() for g in grads),prefix)
            self.assertTrue(any(bool((g!=0).any()) for g in grads),prefix)
        torch.optim.AdamW(net.parameters(),lr=1e-4).step()
        self.assertFalse(torch.equal(before,next(local.dense_encoder.parameters())))
        self.assertEqual(len(local.adjacencies),1)


if __name__=='__main__':unittest.main(verbosity=2)
