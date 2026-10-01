"""CUDA UNIT finite-shadow checks; synthetic inputs, never CT accuracy claims."""
import inspect
import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn

from hiercp_v222.v1_execution import rng_state
from l0_regions.donor_learning import configuration, forward_loss
from l0_regions.training import hash_state
from tests.test_local_cnn_reference_mode_probe import reference_fixture as original_reference_fixture, snapshot
from tests.test_local_cnn_interaction_updates import Budget
from tools.local_cnn_reference_finite_shadow import FiniteShadowScores
from tools.local_cnn_reference_objective_probe import objective_probe
from tools.local_cnn_reference_runtime import probe_updates


CUDA = os.environ.get('NNUNET_TEST_DEVICE', 'cuda') == 'cuda' and torch.cuda.is_available()
if os.environ.get('NNUNET_TEST_DEVICE', 'cuda') == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('Requested finite shadow CUDA UNIT checks require CUDA; CPU fallback forbidden')


def reference_fixture():
    result = original_reference_fixture()
    # Geometry belongs to this UNIT metric fixture only. The existing mode
    # fixture has no candidate-order metric and thus does not carry centers.
    for index, row in enumerate(result[6].rows):
        row['center'] = [index, 0, 0]
    return result


def shadow_metadata(parameters):
    return dict(post_parameter_sha256=hash_state(parameters), delta_sha256='UNIT explicit mapping',
        clipped_gradient_sha256='UNIT explicit mapping', clipping_factor=1.)


@unittest.skipUnless(CUDA, 'Explicit CUDA required; no CPU model execution fallback')
class FiniteShadowChecks(unittest.TestCase):
    def setUp(self):
        self.backend = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def tearDown(self):
        enabled, warn, cudnn, benchmark, matmul_tf32, cudnn_tf32 = self.backend
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
        torch.backends.cudnn.deterministic = cudnn
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32

    def test_all_arms_freeze_pre_forward_buffers_and_reencode_actual_CNN(self):
        _, net, query, support, plan, truth, context, ids, _ = reference_fixture()
        net.train()
        for module in net.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.running_mean.add_(.7)
                module.running_var.mul_(1.8)
                module.num_batches_tracked.add_(11)
        initial = snapshot(net, query, support, plan, truth, context)
        cnn_values = []
        actual = type(net.local).forward
        def observe(local, native):
            self.assertIs(native, query)
            cnn_values.append(hash_state(local.cnn.state_dict()))
            return actual(local, native)
        with patch.object(type(net.local), 'forward', observe), patch.object(
                type(net), 'fit_support_clusters', side_effect=AssertionError('teacher must stay fixed')):
            scorer = FiniteShadowScores(net, query, support, plan, truth, context, ids, Budget())
            self.assertEqual(initial, snapshot(net, query, support, plan, truth, context))
            fixed_buffer_hash = scorer.initial_buffers
            # Simulate native training-forward BN changes. These remain live,
            # but every deterministic counterfactual reads the earlier snapshot.
            for module in net.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.running_mean.add_(3.)
                    module.running_var.mul_(2.)
                    module.num_batches_tracked.add_(1)
            changed = snapshot(net, query, support, plan, truth, context)
            same = {name:p.detach().clone() for name,p in net.named_parameters() if p.requires_grad}
            different = {name:value.clone() for name,value in same.items()}
            first = next(name for name in different if name.startswith('local.cnn.'))
            different[first].add_(.1)
            scorer.score_shadow('rank_only', different, shadow_metadata(different))
            scorer.score_shadow('full', same, shadow_metadata(same))
            report = scorer.report()
        self.assertEqual(changed, snapshot(net, query, support, plan, truth, context))
        self.assertEqual(len(cnn_values), 3)
        self.assertEqual(cnn_values[0], cnn_values[2])
        self.assertNotEqual(cnn_values[0], cnn_values[1])
        self.assertEqual(report['arms']['no_change']['score'], report['arms']['full']['score'])
        self.assertEqual(report['arms']['full']['score_change_from_no_change']['rms'], 0.)
        for arm in report['arms'].values():
            self.assertEqual(arm['buffer_sha256'], fixed_buffer_hash)
            self.assertEqual(arm['fixed_teacher_sha256'], hash_state(plan))
            self.assertTrue(arm['native_CNN_recomputed'])
            self.assertFalse(arm['cached_pre_step_L0_used'])
            self.assertGreaterEqual(arm['total_control_seconds'], arm['synchronized_scoring_seconds'])
        self.assertEqual(report['scoring_forwards'], 3)
        self.assertEqual(report['production_optimizer_updates'], 0)
        json.dumps(report, allow_nan=False)

    def test_existing_first_and_later_Adam_shadows_match_original_full_step(self):
        _, net, query, support, plan, truth, context, ids, training = reference_fixture()
        net.train()
        named = [(name,p) for name,p in net.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([p for _,p in named], lr=training['lr'],
            weight_decay=training['weight_decay'], fused=training['fused_optimizer'])
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            scorer = FiniteShadowScores(net, query, support, plan, truth, context, ids, Budget())
            loss, terms = forward_loss(net, query, support, plan, truth, None, context, configuration(), indices=ids)
            c = configuration()
            weighted = dict(ranking=terms['ranking_loss']*c['ranking_weight'],
                observation_ce=terms['observation_auxiliary_loss']*c['observation_auxiliary_weight'],
                alignment=terms['alignment_loss']*net.alignment_loss_weight, full=loss)
            before = snapshot(net, query, support, plan, truth, context)
            optimizer_hash = hash_state(optimizer.state_dict())
            initial = {name:p.detach().clone() for name,p in named}
            direction = objective_probe(net, named, weighted, optimizer, training['grad_clip'],
                shadow_score_callback=scorer.score_shadow)
            self.assertEqual(before, snapshot(net, query, support, plan, truth, context))
            self.assertEqual(optimizer_hash, hash_state(optimizer.state_dict()))
            self.assertEqual(direction['optimizer_history']['minimum_parameter_step'], None if step == 0 else step)
            finite = scorer.report()
            for name in ('rank_only', 'full'):
                self.assertEqual(finite['arms'][name]['parameter_sha256'],
                    direction['shadow_updates'][name]['post_parameter_sha256'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _,p in named], training['grad_clip'], error_if_nonfinite=True)
            clipped = hash_state({name:p.grad for name,p in named})
            optimizer.step()
            self.assertEqual(hash_state({name:p.detach()-initial[name] for name,p in named}), direction['full_delta_sha256'])
            self.assertEqual(hash_state({name:p.detach() for name,p in named}), direction['full_post_parameter_sha256'])
            self.assertEqual(clipped, direction['full_clipped_gradient_sha256'])
            self.assertEqual(hash_state(optimizer.state_dict()), direction['full_post_optimizer_sha256'])

    def test_failure_preserves_live_state_and_reports_no_success(self):
        _, net, query, support, plan, truth, context, ids, _ = reference_fixture()
        scorer = FiniteShadowScores(net, query, support, plan, truth, context, ids, Budget())
        next(net.parameters()).grad = torch.ones_like(next(net.parameters()))
        before = snapshot(net, query, support, plan, truth, context)
        same = {name:p.detach().clone() for name,p in net.named_parameters() if p.requires_grad}
        with patch('tools.local_cnn_reference_finite_shadow.forward_loss', side_effect=RuntimeError('UNIT readout failure')):
            with self.assertRaisesRegex(RuntimeError, 'UNIT readout failure'):
                scorer.score_shadow('rank_only', same, shadow_metadata(same))
        self.assertEqual(before, snapshot(net, query, support, plan, truth, context))
        with self.assertRaisesRegex(ValueError, 'all three measured'):
            scorer.report()

    def test_default_is_disabled_and_mapping_truth_resource_are_not_bypassed(self):
        self.assertIsNone(inspect.signature(objective_probe).parameters['shadow_score_callback'].default)
        self.assertFalse(inspect.signature(probe_updates).parameters['finite_shadow_score'].default)
        _, net, query, support, plan, truth, context, ids, _ = reference_fixture()
        before = snapshot(net, query, support, plan, truth, context)
        with self.assertRaisesRegex(ValueError, 'truth binding'):
            FiniteShadowScores(net, query, support, plan, 1-truth, context, ids, Budget())
        with self.assertRaisesRegex(MemoryError, 'UNIT explicit resource rejection'):
            FiniteShadowScores(net, query, support, plan, truth, context, ids, Budget(fail=1))
        self.assertEqual(before, snapshot(net, query, support, plan, truth, context))
        scorer = FiniteShadowScores(net, query, support, plan, truth, context, ids, Budget())
        same = {name:p.detach().clone() for name,p in net.named_parameters() if p.requires_grad}
        bad = dict(same)
        bad.pop(next(iter(bad)))
        with self.assertRaisesRegex(ValueError, 'complete shadow parameter'):
            scorer.score_shadow('rank_only', bad, shadow_metadata(bad))
        self.assertEqual(before, snapshot(net, query, support, plan, truth, context))


if __name__ == '__main__':
    unittest.main()
