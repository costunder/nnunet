"""Synthetic DEBUG only: actual task IDs, U masking and evidence interventions."""
import copy
import json
import time
import unittest
import torch
from hiercp_v22.contracts import load_config,validate_split,verify_v1
from hiercp_v22.model import PromptGraphModel,prompt_loss,observed_evidence_loss,transport_positive,context_descriptor,DESCRIPTOR_DIM
from hiercp_v22.data import LocalBatch
from hiercp_v22.training import attach_context
from test_hierarchy_model_debug import debug_batch

def memory_fixture():
    return dict(case_ids=['A','B','C'],embeddings=torch.randn(12,128),owners=torch.arange(3).repeat_interleave(4),
        evidence=torch.tensor([1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1]),
        descriptors=torch.randn(12,DESCRIPTOR_DIM),donor_allowed=torch.tensor([True,True,True]))

class PromptGraphV22AlignmentDebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads(); torch.set_num_threads(4); cls.cfg,cls.base=load_config()
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)
    def setUp(self):
        torch.manual_seed(611); self.memory=memory_fixture()
        self.model=PromptGraphModel(self.cfg,self.base,patient_ids=['A','B','C'],with_local=False).eval()
        self.query=torch.randn(4,128); self.owners=torch.tensor([0,1,1,2])
    def forward(self,memory=None):
        return self.model.forward_tasks(memory or self.memory,self.query,self.owners)

    def test_independent_trainable_patient_labels(self):
        self.assertFalse(torch.equal(self.model.patient_labels[0],self.model.patient_labels[1]))
        self.assertIn('patient_labels',dict(self.model.named_parameters()))
        torch.testing.assert_close(self.model.labels_for(['unseen']),self.model.labels_for(['unseen']))
        self.assertNotIn('unseen',self.model.patient_lookup)

    def test_unknown_has_zero_negative_supervision_gradient(self):
        scores=torch.tensor([.4,.6,.7],requires_grad=True)
        loss,count=observed_evidence_loss(scores,torch.tensor([1,0,-1]),torch.ones(3,dtype=torch.bool))
        loss.backward(); self.assertEqual(int(count),2); self.assertEqual(float(scores.grad[2]),0)
        self.assertLess(float(scores.grad[0]),0); self.assertGreater(float(scores.grad[1]),0)

    def test_no_positive_patient_receives_external_evidence(self):
        with torch.no_grad(): out=self.forward()
        self.assertEqual(float(out['positive'][1].sum()),0)
        self.assertGreater(float(out['transported'][1].sum()),0)
        self.assertTrue(bool(out['available'].all()))
        changed=copy.deepcopy(self.memory); changed['evidence'][0]=-1
        with torch.no_grad(): altered=self.forward(changed)
        self.assertFalse(torch.allclose(out['transported'][1],altered['transported'][1]))
        self.assertFalse(torch.allclose(out['scores'][1:3],altered['scores'][1:3],rtol=1e-6,atol=1e-7))
        self.assertTrue(bool((self.memory['evidence'][4:8]==-1).all()))

    def test_exact_correspondence_transfers_T_but_not_U_or_own_T(self):
        assignment=torch.eye(2).repeat(3,1); owners=torch.tensor([0,0,1,1,2,2])
        evidence=torch.tensor([1,-1,-1,-1,-1,1])
        matrix=torch.eye(2).expand(3,3,2,2).clone()
        transported,available,_=transport_positive(assignment,owners,evidence,matrix,torch.tensor([True,True,False]))
        torch.testing.assert_close(transported[1],torch.tensor([1.,0.]))
        self.assertFalse(bool(available[0])); self.assertTrue(bool(available[1]))
        matrix[0,1]=matrix[0,1].flip(-1)
        changed,_,_=transport_positive(assignment,owners,evidence,matrix,torch.tensor([True,True,False]))
        torch.testing.assert_close(changed[1],torch.tensor([0.,1.]))

    def test_own_patient_observation_cannot_supply_own_score(self):
        with torch.no_grad(): baseline=self.forward()
        changed=copy.deepcopy(self.memory); changed['evidence'][:4]=-1
        with torch.no_grad(): altered=self.forward(changed)
        torch.testing.assert_close(baseline['scores'][0],altered['scores'][0])
        torch.testing.assert_close(baseline['labels'],altered['labels'])

    def test_query_does_not_modify_support_or_other_queries(self):
        with torch.no_grad(): baseline=self.forward()
        self.query[0]+=torch.randn(128)*20
        with torch.no_grad(): changed=self.forward()
        torch.testing.assert_close(baseline['labels'],changed['labels'])
        torch.testing.assert_close(baseline['correspondence'],changed['correspondence'])
        torch.testing.assert_close(baseline['scores'][1:],changed['scores'][1:])

    def test_L1_patient_isolation(self):
        with torch.no_grad(): baseline=self.forward()
        changed=copy.deepcopy(self.memory); changed['embeddings'][4:8]+=10*torch.randn(4,128)
        with torch.no_grad(): altered=self.forward(changed)
        torch.testing.assert_close(baseline['labels'][0],altered['labels'][0])
        self.assertFalse(torch.allclose(baseline['labels'][1],altered['labels'][1]))

    def test_independent_label_permutation_and_query_batch_equivalence(self):
        with torch.no_grad():
            expected=self.forward(); permutation=[torch.randperm(16) for _ in range(3)]
            for i,p in enumerate(permutation): self.model.patient_labels[i].copy_(self.model.patient_labels[i,p].clone())
            actual=self.forward()
            one=self.model.forward_tasks(self.memory,self.query[:1],self.owners[:1])
        torch.testing.assert_close(expected['scores'],actual['scores'],rtol=3e-5,atol=3e-6)
        torch.testing.assert_close(actual['scores'][:1],one['scores'],rtol=3e-5,atol=3e-6)

    def test_unseen_U_task_replaces_all_own_T(self):
        joined,owner=attach_context(self.memory,torch.randn(5,128),torch.randn(5,DESCRIPTOR_DIM),'A')
        self.assertEqual(joined['case_ids'].count('A'),1)
        self.assertTrue(bool((joined['evidence'][joined['owners']==owner]==-1).all()))
        self.assertFalse(bool(joined['donor_allowed'][owner]))

    def test_frozen_task_cache_preserves_exact_scoring(self):
        with torch.no_grad():
            expected=self.forward(); state=self.model.prepare_task_state(self.memory)
            actual=self.model.forward_tasks(self.memory,self.query,self.owners,state=state)
        torch.testing.assert_close(expected['scores'],actual['scores'])

    def test_cpu_bfloat16_patient_alignment_path(self):
        with torch.autocast('cpu',dtype=torch.bfloat16):
            out=self.forward(); loss,_=prompt_loss(out,torch.randn(4,DESCRIPTOR_DIM),torch.tensor([1,-1,0,1]),self.cfg['loss_weights'],self.cfg['temperature'])
        loss.backward(); self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA regression requires an actual GPU')
    def test_cuda_bfloat16_patient_alignment_gradient(self):
        model=self.model.to('cuda').train()
        memory={k:v.cuda() if torch.is_tensor(v) else v for k,v in self.memory.items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model.forward_tasks(memory,self.query.cuda(),self.owners.cuda())
            loss,_=prompt_loss(out,torch.randn(4,DESCRIPTOR_DIM,device='cuda'),torch.tensor([1,-1,0,1],device='cuda'),self.cfg['loss_weights'],self.cfg['temperature'])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_v1_preserved_and_no_view_task_contract(self):
        self.assertEqual(verify_v1(),'74dcc2cf03d2d40d1f582223321d96004333f661')
        self.assertNotIn('task_views',self.cfg); self.assertNotIn('cluster_iterations',self.cfg)
        self.assertEqual(self.cfg['task_definition'],'actual_patient_id')
        good=dict(inner_train=['a'],inner_val=['b'],outer_train=['a','b'],outer_val=['c'])
        validate_split(good)
        with self.assertRaises(ValueError): validate_split(dict(good,outer_val=['a']))

if __name__=='__main__': unittest.main()
