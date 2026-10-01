"""UNIT objective/AdamW checks, with an explicit device and no CT claims.

NNUNET_TEST_DEVICE=cuda requests CUDA execution and errors if unavailable. It
never switches a requested CUDA check to CPU. CPU is an explicit unit fixture,
not a fallback for the separately required actual-CT GPU integration.
"""
import copy
import json
import os
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from hiercp_v222.v1_execution import rng_state
from l0_regions.training import hash_state
from tools.local_cnn_reference_objective_probe import objective_probe


DEVICE = torch.device(os.environ.get('NNUNET_TEST_DEVICE', 'cpu'))
if DEVICE.type == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('NNUNET_TEST_DEVICE=cuda requires an available CUDA device; CPU fallback forbidden')


class UnitObjectiveModel(nn.Module):
    """Labelled small UNIT fixture; no training-data or architecture substitute."""
    def __init__(self):
        super().__init__()
        self.local = nn.Module()
        self.local.cnn = nn.Linear(3, 4, bias=False)
        self.local.project = nn.Linear(4, 4)
        self.local.fuse = nn.Linear(4, 4)
        self.l1 = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(.2))])
        self.label_seed = nn.Parameter(torch.ones(4)*.1)
        self.l2 = nn.ModuleList([nn.Linear(4, 2)])
        self.l2_updates = nn.ModuleList([nn.Linear(2, 2)])
        self.forward_calls = 0

    def forward(self, values):
        self.forward_calls += 1
        z = self.local.cnn(values)
        z = self.local.fuse(F.silu(self.local.project(z)))
        z = self.l1[0](z) + self.label_seed
        return self.l2_updates[0](self.l2[0](z))


def fixture():
    torch.manual_seed(42)
    net = UnitObjectiveModel().to(DEVICE)
    named = [(name, parameter) for name, parameter in net.named_parameters() if parameter.requires_grad]
    midpoint = len(named)//2
    optimizer = torch.optim.AdamW([
        dict(params=[p for _, p in named[:midpoint]], lr=.007, betas=(.8, .94), weight_decay=.03),
        dict(params=[p for _, p in named[midpoint:]], lr=.011, betas=(.9, .98), weight_decay=.02),
    ], fused=DEVICE.type == 'cuda', amsgrad=True)
    values = torch.arange(18, dtype=torch.float32, device=DEVICE).reshape(6, 3)/11
    targets = torch.tensor([1, 1, 0, 0, 0, 0], device=DEVICE)
    return net, named, optimizer, values, targets


def terms_for(net, values, targets):
    logits = net(values)
    score = logits[:, 1]-logits[:, 0]
    ranking = F.softplus(score[None, targets == 0]-score[targets == 1, None]).mean()
    ce = .7*F.cross_entropy(logits, targets)
    align = .13*logits.square().mean()
    return dict(ranking=ranking, observation_ce=ce, alignment=align, full=ranking+ce+align)


def actual_step(named, optimizer, terms, grad_clip, report):
    initial = {name: p.detach().clone() for name, p in named}
    optimizer.zero_grad(set_to_none=True)
    terms['full'].backward()
    torch.nn.utils.clip_grad_norm_([p for _, p in named], grad_clip, error_if_nonfinite=True)
    clipped = hash_state({name: p.grad for name, p in named})
    optimizer.step()
    delta = {name: p.detach()-initial[name] for name, p in named}
    return dict(delta=hash_state(delta), clipped=clipped,
                parameters=hash_state({name: p.detach() for name, p in named}),
                optimizer=hash_state(optimizer.state_dict()))


class ObjectiveProbeChecks(unittest.TestCase):
    def assertPrediction(self, actual, report):
        self.assertEqual(actual['delta'], report['full_delta_sha256'])
        self.assertEqual(actual['clipped'], report['full_clipped_gradient_sha256'])
        self.assertEqual(actual['parameters'], report['full_post_parameter_sha256'])
        self.assertEqual(actual['optimizer'], report['full_post_optimizer_sha256'])

    def test_one_forward_preserves_grad_rng_bn_optimizer_and_predicts_actual_full(self):
        net, named, optimizer, values, targets = fixture()
        terms = terms_for(net, values, targets)
        for _, parameter in named:
            parameter.grad = torch.full_like(parameter, 19.)
        grad_refs = [p.grad for _, p in named]
        snapshot = hash_state(dict(model=net.state_dict(), optimizer=optimizer.state_dict(),
                                  grads={name: p.grad for name, p in named}, rng=rng_state()))
        report = objective_probe(net, named, terms, optimizer, .4)
        self.assertEqual(net.forward_calls, 1)
        self.assertEqual(snapshot, hash_state(dict(model=net.state_dict(), optimizer=optimizer.state_dict(),
                                  grads={name: p.grad for name, p in named}, rng=rng_state())))
        self.assertTrue(all(p.grad is ref for (_, p), ref in zip(named, grad_refs)))
        self.assertEqual(report['device'], str(next(net.parameters()).device))
        self.assertEqual(report['additional_model_forwards'], 0)
        self.assertEqual(report['simultaneous_shadow_optimizers'], 1)
        self.assertTrue(report['original_parameters_gradients_bn_rng_optimizer_preserved'])
        self.assertFalse(report['optimizer_history']['moments_reset'])
        for module in ('CNN', 'readout_fusion', 'L1', 'L2'):
            self.assertGreater(report['module_gradients'][module]['terms']['ranking']['norm'], 0)
            self.assertIsNotNone(report['module_gradients'][module]['cosines']['ranking_vs_full']['cosine'])
        self.assertLess(report['shadow_updates']['full']['clipping_factor'], 1.)
        json.dumps(report, allow_nan=False)
        self.assertPrediction(actual_step(named, optimizer, terms, .4, report), report)

    def test_later_shadow_steps_reuse_existing_moments_and_all_parameter_groups(self):
        net, named, optimizer, values, targets = fixture()
        reports = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            terms = terms_for(net, values, targets)
            report = objective_probe(net, named, terms, optimizer, .3)
            self.assertEqual(report['optimizer_history']['state_entries'], 0 if step == 0 else len(named))
            self.assertEqual(report['optimizer_history']['minimum_parameter_step'], None if step == 0 else step)
            self.assertPrediction(actual_step(named, optimizer, terms, .3, report), report)
            reports.append(report)
        self.assertNotEqual(reports[0]['full_delta_sha256'], reports[1]['full_delta_sha256'])
        self.assertEqual(net.forward_calls, 3)

    def test_analytic_term_norm_cosines_and_actual_delta_direction(self):
        net = nn.Module()
        net.local = nn.Module()
        net.local.cnn = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            net.local.cnn.weight.copy_(torch.tensor([[1., -2.]]))
        net.to(DEVICE)
        named = list(net.named_parameters())
        p = named[0][1]
        optimizer = torch.optim.AdamW([p], lr=.01, weight_decay=0., fused=DEVICE.type == 'cuda')
        rank = (p*torch.tensor([[3., 4.]], device=DEVICE)).sum()
        ce = -(p*torch.tensor([[3., 4.]], device=DEVICE)).sum()
        align = (p*torch.tensor([[1., 0.]], device=DEVICE)).sum()
        terms = dict(ranking=rank, observation_ce=ce, alignment=align, full=rank+ce+align)
        report = objective_probe(net, named, terms, optimizer, 100.)
        module = report['module_gradients']['CNN']
        self.assertAlmostEqual(module['terms']['ranking']['norm'], 5.)
        self.assertAlmostEqual(module['cosines']['ranking_vs_observation_ce']['cosine'], -1.)
        self.assertAlmostEqual(module['cosines']['ranking_vs_alignment']['cosine'], .6)
        self.assertAlmostEqual(module['cosines']['ranking_vs_full']['cosine'], .6)
        self.assertAlmostEqual(report['delta_cosines']['CNN']['rank_only_vs_full']['cosine'], 2**-.5, places=5)
        self.assertPrediction(actual_step(named, optimizer, terms, 100., report), report)

    def test_missing_and_zero_derivatives_have_explicit_undefined_cosines(self):
        net = nn.Module()
        net.local = nn.Module()
        net.local.cnn = nn.Linear(2, 1, bias=False)
        net.l2 = nn.ModuleList([nn.Linear(2, 1, bias=False)])
        net.to(DEVICE)
        named = list(net.named_parameters())
        cnn, l2 = [p for _, p in named]
        optimizer = torch.optim.AdamW([cnn, l2], lr=.01, weight_decay=.1, fused=DEVICE.type == 'cuda')
        rank = (cnn*0).sum()
        ce = cnn.square().sum()+l2.square().sum()
        align = torch.zeros((), device=DEVICE)  # Explicit disconnected UNIT constant.
        terms = dict(ranking=rank, observation_ce=ce, alignment=align, full=rank+ce+align)
        report = objective_probe(net, named, terms, optimizer, 1.)
        self.assertEqual(report['module_gradients']['CNN']['terms']['ranking']['norm'], 0.)
        self.assertIsNone(report['module_gradients']['L2']['terms']['ranking']['norm'])
        self.assertEqual(report['module_gradients']['CNN']['cosines']['ranking_vs_full']['reason'], 'left_zero_direction')
        self.assertEqual(report['module_gradients']['L2']['cosines']['ranking_vs_full']['reason'], 'left_no_parameter_dependency')
        self.assertEqual(report['module_gradients']['L1']['terms']['ranking']['reason'], 'module_has_no_parameters')
        self.assertEqual(report['shadow_updates']['rank_only']['parameters_skipped_for_missing_derivative'], 1)
        self.assertEqual(report['shadow_updates']['rank_only']['module_delta_norms']['L2']['norm'], 0.)
        self.assertPrediction(actual_step(named, optimizer, terms, 1., report), report)

    def test_requested_parameter_binding_and_original_optimizer_are_admitted(self):
        net, named, optimizer, values, targets = fixture()
        terms = terms_for(net, values, targets)
        with self.assertRaisesRegex(ValueError, 'binding'):
            objective_probe(net, named[:-1], terms, optimizer, 1.)
        wrong = torch.optim.AdamW([p for _, p in named[:-1]])
        with self.assertRaisesRegex(ValueError, 'exactly'):
            objective_probe(net, named, terms, wrong, 1.)
        with self.assertRaisesRegex(ValueError, 'substitute'):
            objective_probe(net, named, terms, torch.optim.SGD([p for _, p in named], lr=.1), 1.)
        for clip in (0., -1., float('nan'), float('inf')):
            with self.subTest(clip=clip), self.assertRaises(ValueError):
                objective_probe(net, named, terms, optimizer, clip)

    def test_nonfinite_losses_and_derivatives_raise_and_preserve_original_state(self):
        net, named, optimizer, values, targets = fixture()
        terms = terms_for(net, values, targets)
        bad = dict(terms, full=terms['full']*float('inf'))
        with self.assertRaises(FloatingPointError):
            objective_probe(net, named, bad, optimizer, 1.)
        p = named[0][1]
        rank = (p-p.detach()).sum().sqrt()  # Finite zero loss, infinite derivative.
        ce = p.square().sum()
        align = p.sum()*0
        bad = dict(ranking=rank, observation_ce=ce, alignment=align, full=rank+ce+align)
        before = hash_state(dict(model=net.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state()))
        with self.assertRaises(FloatingPointError):
            objective_probe(net, named, bad, optimizer, 1.)
        self.assertEqual(before, hash_state(dict(model=net.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state())))

    def test_fused_stride_layout_and_later_moments_are_preserved(self):
        net = nn.Module()
        net.local = nn.Module()
        net.local.cnn = nn.Conv3d(2, 3, 3, bias=False)
        net.to(DEVICE, memory_format=torch.channels_last_3d)
        named = list(net.named_parameters())
        p = named[0][1]
        optimizer = torch.optim.AdamW([p], lr=.003, fused=DEVICE.type == 'cuda')
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            rank = (p-.2).square().mean()
            ce = .3*p.square().mean()
            align = .04*p.sin().sum()
            terms = dict(ranking=rank, observation_ce=ce, alignment=align, full=rank+ce+align)
            stride = p.stride()
            report = objective_probe(net, named, terms, optimizer, .01)
            self.assertPrediction(actual_step(named, optimizer, terms, .01, report), report)
            self.assertEqual(p.stride(), stride)
            self.assertEqual(optimizer.state[p]['exp_avg'].stride(), stride)
            self.assertEqual(optimizer.state[p]['exp_avg_sq'].stride(), stride)


if __name__ == '__main__':
    unittest.main()
