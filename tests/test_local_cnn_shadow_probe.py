"""Explicit UNIT operator inputs; no CT learning or production accuracy claim."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from hiercp_v222.v1_execution import rng_state, restore_rng
from l0_regions.donor_learning import LiveContext, configuration, forward_loss
from l0_regions.training import hash_state
from tools.local_cnn_shadow_probe import direction, objective_terms, shadow_update


class UnitLocal(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cnn = nn.Linear(dim, dim)
        self.project = nn.Linear(dim, dim)
        self.fuse = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(.2))
    def forward(self, x):
        return self.fuse(torch.tanh(self.project(torch.tanh(self.cnn(x)))))


class UnitModel(nn.Module):
    """Small algebraic unit fixture, never a candidate production model."""
    def __init__(self):
        super().__init__()
        self.local = UnitLocal(8)
        self.l1 = nn.Sequential(nn.Linear(8, 8), nn.Dropout(.2), nn.Tanh())
        self.l2 = nn.Linear(8, 2)
    def prepare_support(self, x, owner, target, cluster_plan):
        return dict(features=x, targets=target)
    def predict_embeddings(self, embedding, state):
        return dict(logits=self.l2(self.l1(embedding)), alignment_loss_weight=1.,
                    alignment_loss=F.cross_entropy(self.l2(self.l1(state['features'])), state['targets']))


class UnitConvLocal(nn.Module):
    """Conv3D operator layout fixture, not the production CT architecture."""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv3d(2, 8, 3, padding=1)
        self.norm = nn.LayerNorm(8)
        self.project = nn.Linear(8, 8)
        self.fuse = nn.Sequential(nn.Linear(8, 8), nn.Dropout(.2))
    def forward(self, x):
        x = self.cnn(x)
        x = self.norm(x.movedim(1, -1)).movedim(-1, 1)
        return self.fuse(torch.tanh(self.project(x.mean((2, 3, 4)))))


class UnitConvModel(UnitModel):
    def __init__(self):
        super().__init__()
        self.local = UnitConvLocal()


class UnitBudget:
    def __init__(self):
        self.checks = 0
    def check(self):
        self.checks += 1


def rows():
    return [dict(id=f'UNIT{i}', case_id='UNIT recipient', patient_group='UNIT p',
                 donor_case_id='UNIT donor', donor_component=1, donor_group='UNIT d',
                 target=target, bounds=dict(edges=i)) for i, target in enumerate([1, 1, 0, 0])]


def fixture(factory, dim, device):
    torch.set_num_threads(4)
    torch.manual_seed(197)
    net = factory().to(device).train()
    data = rows()
    context = LiveContext(SimpleNamespace(rows=data), 4)
    query = torch.randn(4, dim, device=device)
    support = (torch.randn(12, dim, device=device), torch.arange(3, device=device).repeat_interleave(4),
               torch.tensor([0, 1, 0, 1] * 3, device=device))
    plan = net.fit_support_clusters(*support) if hasattr(net, 'fit_support_clusters') else {}
    optimizer = torch.optim.AdamW(net.parameters(), lr=.01, weight_decay=.013, fused=False)
    terms = objective_terms(net, query, support, plan, data, list(range(4)), context, configuration())
    sum(terms.values()).backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), .5, error_if_nonfinite=True)
    optimizer.step()
    saved = dict(model=copy.deepcopy(net.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                 rng=copy.deepcopy(rng_state()), state=dict(batch=4, step=1),
                 identity=dict(precision='FP32', ranking=configuration(),
                               base=dict(training=dict(grad_clip=.5))))
    return saved, query, support, plan, data, context


class Checks(unittest.TestCase):
    def test_signed_conflict_projection(self):
        reference = torch.tensor([2., 0.])
        value = direction(reference, torch.tensor([-6., 4.]))
        self.assertEqual(value['signed_projection'], -3)
        self.assertEqual(value['opposing_projection'], 3)
        self.assertLess(value['cosine'], 0)
        self.assertIsNone(direction(torch.zeros(2), reference)['cosine'])
        self.assertEqual(direction(reference, torch.zeros(2))['opposing_projection'], 0)

    def test_same_adam_moments_none_gradients_and_no_saved_or_rng_mutation(self):
        saved, query, support, plan, data, context = fixture(UnitModel, 8, 'cpu')
        original = hash_state(saved)
        before = hash_state(rng_state())
        budget = UnitBudget()
        report = shadow_update(UnitModel, saved, query, support, plan, data, list(range(4)), context, budget)
        self.assertEqual(hash_state(saved), original)
        self.assertEqual(hash_state(rng_state()), before)
        self.assertTrue(report['decomposition_full_backward_verified'])
        self.assertEqual(report['branches']['full']['adam_step_before_min'], 1)
        self.assertEqual(report['branches']['full']['adam_step_after_min'], 2)
        self.assertEqual(report['branches']['alignment']['modules']['CNN']['delta_norm'], 0)
        self.assertEqual(report['branches']['alignment']['modules']['readout_fusion']['delta_norm'], 0)
        self.assertTrue(all(n.startswith('local.') for n in report['branches']['alignment']['unused_gradient_parameters']))
        self.assertGreater(budget.checks, 4)

    def test_full_update_matches_production_loss_and_saved_adam_state(self):
        factory = UnitModel
        saved, query, support, plan, data, context = fixture(factory, 8, 'cpu')
        report = shadow_update(factory, saved, query, support, plan, data, list(range(4)), context, UnitBudget())
        net = factory().train()
        net.load_state_dict(saved['model'])
        optimizer = torch.optim.AdamW(net.parameters(), lr=.01, weight_decay=.013, fused=False)
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        old = [p.detach().clone() for p in net.parameters()]
        restore_rng(saved['rng'])
        loss, _ = forward_loss(net, query, support, plan, torch.tensor([1, 1, 0, 0]),
                               torch.ones(2), context, configuration(), indices=list(range(4)))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), .5, error_if_nonfinite=True)
        optimizer.step()
        delta = torch.cat([(p.detach() - initial).flatten() for p, initial in zip(net.parameters(), old)])
        self.assertAlmostEqual(float(loss.detach()), report['losses']['full'], places=6)
        self.assertAlmostEqual(float(norm), report['branches']['full']['gradient_norm_before_clip'], places=6)
        self.assertAlmostEqual(float(delta.norm()), report['branches']['full']['modules']['global']['delta_norm'], places=6)
        # Primed moments must matter: a fresh Adam optimizer gives another step.
        fresh = factory().train();fresh.load_state_dict(saved['model']);restore_rng(saved['rng'])
        fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=.01, weight_decay=.013, fused=False)
        fresh_loss, _ = forward_loss(fresh, query, support, plan, torch.tensor([1, 1, 0, 0]),
                                     torch.ones(2), context, configuration(), indices=list(range(4)))
        fresh_loss.backward();torch.nn.utils.clip_grad_norm_(fresh.parameters(), .5);fresh_optimizer.step()
        self.assertGreater(float(torch.cat([(a-b).detach().flatten() for a,b in zip(net.parameters(),fresh.parameters())]).norm()), 1e-5)

    def test_failure_restores_caller_rng(self):
        saved, query, support, plan, data, context = fixture(UnitModel, 8, 'cpu')
        digest = hash_state(rng_state())
        def failing_factory():
            torch.rand(10)
            raise MemoryError('UNIT explicit budget rejection')
        with self.assertRaisesRegex(MemoryError, 'explicit budget'):
            shadow_update(failing_factory, saved, query, support, plan, data, list(range(4)), context, UnitBudget())
        self.assertEqual(hash_state(rng_state()), digest)

    def test_cuda_train_dropout_full_objective_operator_parity(self):
        if not torch.cuda.is_available():
            self.skipTest('Explicit CUDA operator test requires GPU')
        from hiercp_v222.model import PromptGraphModel
        cfg = json.loads(Path('config/prompt_graph_v222_v1_l0.json').read_text())
        base = json.loads(Path('config/train.json').read_text())
        def factory():
            net = PromptGraphModel(cfg, base, {}, local_encoder=UnitLocal(128)).cuda()
            net.checkpoint_support = False
            return net
        saved, query, support, plan, data, context = fixture(factory, 128, 'cuda')
        report = shadow_update(factory, saved, query, support, plan, data, list(range(4)), context, UnitBudget())
        net = factory().train();net.load_state_dict(saved['model']);restore_rng(saved['rng'])
        terms = objective_terms(net, query, support, plan, data, list(range(4)), context, configuration())
        restore_rng(saved['rng'])
        production, _ = forward_loss(net, query, support, plan, torch.tensor([1,1,0,0],device='cuda'),
                                    torch.ones(2,device='cuda'), context, configuration(), indices=list(range(4)))
        torch.testing.assert_close(sum(terms.values()), production, atol=1e-6, rtol=1e-6)
        self.assertTrue(report['decomposition_full_backward_verified'])
        self.assertGreater(report['gradients']['CNN']['norms']['ranking'], 0)
        self.assertEqual(report['branches']['alignment']['modules']['CNN']['delta_norm'], 0)
        self.assertEqual(report['branches']['full']['adam_step_after_min'], 2)

    def test_cuda_fused_adam_preserves_saved_conv_layout_and_derivative_layout(self):
        if not torch.cuda.is_available():
            self.skipTest('Explicit fused CUDA layout test requires GPU')
        from tools.local_cnn_shadow_probe import _preserve_parameter_layouts
        torch.manual_seed(181)
        factory = lambda: UnitConvModel().cuda()
        model = factory().train()
        # A saved non-default layout must survive reconstruction. Conv3D plus
        # movedim/LayerNorm also exercises autograd.grad derivative strides.
        model.local.cnn.weight.data = model.local.cnn.weight.data.contiguous(memory_format=torch.channels_last_3d)
        query = torch.randn(4, 2, 5, 5, 5, device='cuda')
        support = (torch.randn(12, 8, device='cuda'), torch.arange(3,device='cuda').repeat_interleave(4),
                   torch.tensor([0,1,0,1]*3,device='cuda'))
        data=rows();context=LiveContext(SimpleNamespace(rows=data),4);plan={}
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.013,fused=True)
        terms=objective_terms(model,query,support,plan,data,list(range(4)),context,configuration())
        sum(terms.values()).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
        saved=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
                   rng=copy.deepcopy(rng_state()),state=dict(batch=4,step=1),identity=dict(precision='FP32',
                   ranking=configuration(),base=dict(training=dict(grad_clip=.5))))
        report=shadow_update(factory,saved,query,support,plan,data,list(range(4)),context,UnitBudget())
        self.assertIn('local.cnn.weight',report['saved_parameter_layouts_restored'])
        direct=factory().train();_preserve_parameter_layouts(direct,saved['model']);direct.load_state_dict(saved['model'])
        direct_optimizer=torch.optim.AdamW(direct.parameters(),lr=.01,weight_decay=.013,fused=True)
        direct_optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        old=[p.detach().clone() for p in direct.parameters()];restore_rng(saved['rng'])
        loss,_=forward_loss(direct,query,support,plan,torch.tensor([1,1,0,0],device='cuda'),
                            torch.ones(2,device='cuda'),context,configuration(),indices=list(range(4)))
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(direct.parameters(),.5);direct_optimizer.step()
        delta=torch.cat([(p.detach()-initial).flatten() for p,initial in zip(direct.parameters(),old)])
        self.assertAlmostEqual(float(loss.detach()),report['losses']['full'],places=6)
        self.assertAlmostEqual(float(norm),report['branches']['full']['gradient_norm_before_clip'],places=5)
        self.assertAlmostEqual(float(delta.norm()),report['branches']['full']['modules']['global']['delta_norm'],places=6)
        self.assertEqual(report['branches']['full']['adam_step_after_min'],2)


if __name__ == '__main__':
    unittest.main()
