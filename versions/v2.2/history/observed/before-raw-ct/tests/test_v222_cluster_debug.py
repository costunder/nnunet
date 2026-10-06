"""Synthetic adversarial clustering DEBUG; no clinical performance claims."""
import copy
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from hiercp_v222.clustering import select_partition, fit_prototypes, alignment_loss, prototype_logits
from hiercp_v222.data import support_for_query
from hiercp_v222.contracts import load_config
from hiercp_v222.model import PromptGraphModel, supervised_loss
from test_v222_leakage_debug import debug_contract


def modes():
    rng = np.random.default_rng(123)
    # Three well-separated modes per class, six distinct patients per mode.
    x = np.eye(6)[np.tile(np.arange(3),6)] + rng.normal(0,.015,(18,6))
    return torch.tensor(np.stack((x,np.roll(x,3,axis=1)),axis=1),dtype=torch.float32)


class ClusterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_resolves_multiple_modes_without_fixed_K(self):
        for k in (2,3,4):
            rng=np.random.default_rng(k)
            ids=np.repeat(np.arange(k),6)
            x=np.eye(k)[ids]+rng.normal(0,.01,(len(ids),k))
            labels,audit=select_partition(x)
            self.assertEqual(audit['k'],k)
            self.assertGreater(audit['silhouette'],.9)
            self.assertTrue(np.array_equal(labels[:,None]==labels,ids[:,None]==ids))
            self.assertEqual([r['k'] for r in audit['candidates']],list(range(2,len(x)//2+1)))

    def test_singleton_outlier_retained_but_not_own_mode(self):
        x=np.array([[1.,0],[1.,.01],[1.,-.01],[1.,.02],[-1.,0]])
        labels,audit=select_partition(x)
        self.assertEqual(len(labels),5)
        self.assertEqual(audit['k'],1)
        self.assertEqual(audit['reason'],'no_positive_silhouette_non_singleton_cut')

    def test_unresolved_identical_and_missing_class_evidence(self):
        labels,audit=select_partition(np.ones((8,4)))
        self.assertEqual(audit['k'],1)
        self.assertEqual(audit['reason'],'identical_directions_no_resolved_modes')
        local=modes()[:3]
        owners=torch.tensor([0,0,1,1,2]);classes=torch.tensor([0,1,0,1,0])
        plan=fit_prototypes(local,owners,classes)
        self.assertNotIn(5,plan['active'].tolist())
        self.assertEqual(plan['audit']['missing_class_labels_excluded'],1)
        self.assertEqual(plan['audit']['classes'][1]['patients'],2)
        for bad in (np.zeros((4,3)),np.full((4,3),np.nan)):
            with self.assertRaises(ValueError):select_partition(bad)

    def test_patient_permutation_and_class_permutation_equivariance(self):
        local=modes();owners=torch.arange(18).repeat_interleave(2);classes=torch.arange(2).repeat(18)
        plan=fit_prototypes(local,owners,classes)
        q=torch.randn(7,6);reference=prototype_logits(q,local,plan,.2)
        order=torch.randperm(18);other=local[order]
        swapped=fit_prototypes(other,owners,classes)
        torch.testing.assert_close(reference,prototype_logits(q,other,swapped,.2))
        other=local.flip(1);swapped=fit_prototypes(other,owners,classes)
        torch.testing.assert_close(reference.flip(1),prototype_logits(q,other,swapped,.2))

    def test_alignment_objective_and_prototypes_change_scores(self):
        local=modes().requires_grad_();owners=torch.arange(18).repeat_interleave(2);classes=torch.arange(2).repeat(18)
        plan=fit_prototypes(local,owners,classes)
        self.assertFalse(plan['centers'].requires_grad)
        self.assertEqual(plan['audit']['prototype_count'],6)
        correct=alignment_loss(local,plan,.2)
        wrong=alignment_loss(local.flip(1),plan,.2)
        collapsed=alignment_loss(torch.ones_like(local),plan,.2)
        self.assertLess(float(correct.detach()),float(wrong.detach()));self.assertLess(float(correct.detach()),float(collapsed))
        # Learn the alignment of deliberately wrong vectors to the fixed teacher.
        live=torch.nn.Parameter(local.detach().flip(1).clone())
        opt=torch.optim.Adam([live],lr=.1)
        before=float(alignment_loss(live,plan,.2).detach())
        for _ in range(30):
            opt.zero_grad();loss=alignment_loss(live,plan,.2);loss.backward();opt.step()
        self.assertLess(float(alignment_loss(live,plan,.2).detach()),before/2)
        self.assertIsNone(local.grad)
        q=local.detach()[::3,1]
        score=prototype_logits(q,local,plan,.2)
        changed=prototype_logits(q,local.flip(1),plan,.2)
        self.assertFalse(torch.allclose(score,changed))

    def test_query_patient_intervention_cannot_change_fit(self):
        cfg,base=load_config();torch.manual_seed(42)
        model=PromptGraphModel(cfg,base,debug_contract()).eval()
        memory=dict(embeddings=torch.randn(20,128),owners=torch.arange(5).repeat_interleave(4),
                    classes=torch.tensor([0,1,0,1]*5),patient_groups=['A','A','B','C','D'])
        support=support_for_query(memory,'A');plan=model.fit_support_clusters(*support)
        changed=copy.deepcopy(memory);changed['embeddings'][:8]=100000;changed['classes'][:8]=1
        other=model.fit_support_clusters(*support_for_query(changed,'A'))
        torch.testing.assert_close(plan['centers'],other['centers'],rtol=0,atol=0)
        torch.testing.assert_close(plan['assignment'],other['assignment'],rtol=0,atol=0)
        self.assertEqual(plan['audit']['classes'],other['audit']['classes'])
        self.assertFalse(plan['centers'].requires_grad)
        # Query features/targets are not accepted by the fitting API at all.
        with self.assertRaises(TypeError):model.fit_support_clusters(*support,query_targets=torch.ones(2))
        with self.assertRaises(ValueError):model.prepare_support(support[0]+1,*support[1:],cluster_plan=plan)

    def test_fixed_episode_teacher_and_both_loss_gradient_paths(self):
        cfg,base=load_config();torch.manual_seed(7)
        model=PromptGraphModel(cfg,base,debug_contract()).train()
        support=(torch.randn(24,128),torch.arange(6).repeat_interleave(4),torch.tensor([0,1,0,1]*6))
        plan=model.fit_support_clusters(*support);centers=plan['centers'].clone()
        self.assertTrue(model.training)
        state=model.prepare_support(*support,cluster_plan=plan)
        output=model.predict_embeddings(torch.randn(4,128),state)
        loss=supervised_loss(output,torch.tensor([0,1,0,1]))
        torch.testing.assert_close(loss,F.cross_entropy(output['logits'],torch.tensor([0,1,0,1]))+output['alignment_loss'])
        for objective in (F.cross_entropy(output['logits'],torch.tensor([0,1,0,1])),output['alignment_loss']):
            model.zero_grad(set_to_none=True);objective.backward(retain_graph=True)
            for prefix in ('l1.0','l1.1','l2.0','l2.1','label_seed'):
                params=[p for n,p in model.named_parameters() if n.startswith(prefix)]
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in params),prefix)
        torch.testing.assert_close(plan['centers'],centers,rtol=0,atol=0)

    def test_cluster_count_alone_does_not_change_class_prior(self):
        # Splitting identical class-0 vectors into duplicated identical centers
        # cannot increase class-0 evidence merely by increasing K.
        local=torch.tensor([[[1.,0],[0,1]],[[1.,0],[0,1]],[[1.,0],[0,1]],[[1.,0],[0,1]]])
        owners=torch.arange(4).repeat_interleave(2);classes=torch.arange(2).repeat(4)
        plan=fit_prototypes(local,owners,classes);q=torch.randn(5,2)
        reference=prototype_logits(q,local,plan,.2)
        modified=dict(plan,prototype_classes=torch.tensor([0,0,1]),mass=torch.tensor([2.,2.,4.]),
                      membership=F.one_hot(torch.tensor([0,2,0,2,1,2,1,2]),num_classes=3).float(),
                      log_prior=torch.tensor([.5,.5,1.]).log())
        torch.testing.assert_close(reference,prototype_logits(q,local,modified,.2))


if __name__=='__main__':unittest.main()
