"""CUDA UNIT controls only; synthetic native-shaped inputs, no CT accuracy claim.

The full 32-query/128D/four-head/two-L1/two-L2 contracts are retained. Small
synthetic image shapes are explicitly test fixtures, never production defaults.
No CPU training fallback is used when the requested CUDA device is unavailable.
"""
import os
import unittest
from unittest.mock import patch

import torch

from hiercp_v222.v1_execution import rng_state
from l0_regions.donor_learning import forward_loss
from l0_regions.training import hash_state
from tools.local_cnn_reference_mode_probe import MODES, mode_probe
from tools.local_cnn_reference_transfer import clone_reference_control
from tests.test_local_cnn_interaction_updates import fixture, Budget


DEVICE = os.environ.get('NNUNET_TEST_DEVICE', 'cuda')


def reference_fixture():
    legacy, settings, support = fixture(DEVICE)
    reference, _ = clone_reference_control(legacy, policy='affine_relations_zero_out_bias')
    ids = list(settings['loss_context'].order[0])
    query = settings['batch_provider'](ids)
    plan = reference.fit_support_clusters(*support)
    targets = torch.tensor([settings['loss_context'].rows[i]['target'] for i in ids],
                           device=DEVICE, dtype=torch.long)
    training = {key: settings[key] for key in ('lr', 'weight_decay', 'grad_clip', 'fused_optimizer')}
    return legacy, reference, query, support, plan, targets, settings['loss_context'], ids, training


def snapshot(net, query, support, plan, targets, context):
    return dict(state=hash_state(net.state_dict()),
        gradients=hash_state({n: p.grad for n, p in net.named_parameters()}),
        modes=tuple(m.training for m in net.modules()), rng=hash_state(rng_state()),
        inputs=hash_state(dict(query=vars(query), support=support, plan=plan, targets=targets)),
        context=hash_state(dict(rows=context.rows, order=context.order, uses=dict(context.uses),
            counts=dict(context.counts), steps=context.steps, pairs=context.pairs, audit=context.audit)))


@unittest.skipUnless(DEVICE == 'cuda' and torch.cuda.is_available(),
                     'CUDA UNIT device required; no CPU training fallback')
class ReferenceModeProbeChecks(unittest.TestCase):
    def test_orthogonal_modes_gradient_route_and_single_original_forwards(self):
        _, net, query, support, plan, targets, context, ids, training = reference_fixture()
        net.train()
        net.l2[0].eval()
        net.l1[1].bn.eval()
        next(net.parameters()).grad = torch.ones_like(next(net.parameters()))
        old = snapshot(net, query, support, plan, targets, context)
        with patch('tools.local_cnn_reference_mode_probe.forward_loss', wraps=forward_loss) as calls:
            report = mode_probe(net, query, support, plan, targets, context, ids, training, Budget())
        self.assertEqual(old, snapshot(net, query, support, plan, targets, context))
        self.assertEqual(calls.call_count, 6)
        for call in calls.call_args_list:
            self.assertIs(call.args[1], query)
            self.assertIs(call.args[2], support)
            self.assertIs(call.args[3], plan)
            self.assertIs(call.args[6], context)
            self.assertEqual(call.kwargs['indices'], ids)
        self.assertEqual([row['mode'] for row in report['modes']], [row[0] for row in MODES])
        self.assertEqual(report['physical_batch'], 32)
        self.assertGreater(report['ranking_pairs'], 0)
        for row in report['modes']:
            self.assertEqual(row['optimizer_updates'], 0)
            for value in row['dropout_modules'].values():
                self.assertEqual(value['training'], row['dropout_training'])
            for name, before in row['batch_norm_before'].items():
                after = row['batch_norm_after'][name]
                self.assertEqual(after['forward_calls'], 1)
                self.assertEqual(after['num_batches_tracked']-before['num_batches_tracked'],
                                 int(row['bn_training']))
                self.assertEqual(after['batch_statistics_used'], row['bn_training'])
                if not row['bn_training']:
                    self.assertEqual(before['buffers_sha256'], after['buffers_sha256'])
            alignment = row['per_loss_query_CNN_gradients']['alignment']
            self.assertEqual(alignment['query_gradient_route_active'], row['bn_training'])
            self.assertEqual(alignment['CNN_gradient_route_active'], row['bn_training'])
            rank = row['per_loss_query_CNN_gradients']['ranking']
            self.assertGreater(rank['query_embedding_gradient_norm'], 0)
            self.assertGreater(rank['CNN_parameter_gradient_norm'], 0)
        identities = {(r['parameter_sha256'], r['teacher_plan_sha256'],
                       r['input_sha256'], r['initial_rng_sha256']) for r in report['modes']}
        self.assertEqual(len(identities), 1)
        self.assertTrue(report['same_initial_weights_teacher_tile_and_rng'])
        self.assertTrue(report['original_state_gradients_modes_rng_and_inputs_preserved'])
        self.assertFalse(report['production_checkpoint_written'])
        self.assertEqual(report['production_optimizer_updates'], 0)

        update = report['frozen_BN_dropout_off_full_update']
        self.assertEqual(update['optimizer_updates'], 1)
        self.assertEqual(update['optimizer']['lr'], training['lr'])
        self.assertEqual(update['optimizer']['weight_decay'], training['weight_decay'])
        self.assertEqual(update['optimizer']['grad_clip'], training['grad_clip'])
        self.assertEqual(update['optimizer']['fused'], training['fused_optimizer'])
        for label in ('CNN', 'L1', 'L2'):
            self.assertGreater(update['parameter_delta_norms'][label], 0)
        for name, before in update['batch_norm_before'].items():
            after = update['batch_norm_after'][name]
            self.assertEqual(after['forward_calls'], 2)
            self.assertEqual(after['buffers_sha256'], before['buffers_sha256'])
        self.assertAlmostEqual(update['before']['full_loss'], report['modes'][0]['full_loss'], places=5)
        self.assertEqual(update['teacher_plan_sha256'], report['modes'][0]['teacher_plan_sha256'])

    def test_resource_rejection_preserves_original_everything(self):
        _, net, query, support, plan, targets, context, ids, training = reference_fixture()
        old = snapshot(net, query, support, plan, targets, context)
        with self.assertRaisesRegex(MemoryError, 'UNIT explicit resource rejection'):
            mode_probe(net, query, support, plan, targets, context, ids, training, Budget(fail=2))
        self.assertEqual(old, snapshot(net, query, support, plan, targets, context))

    def test_legacy_and_unbound_tile_are_rejected_without_forward(self):
        legacy, net, query, support, plan, targets, context, ids, training = reference_fixture()
        with patch('tools.local_cnn_reference_mode_probe.forward_loss', side_effect=AssertionError('unexpected forward')):
            with self.assertRaisesRegex(ValueError, 'explicit reference clone'):
                mode_probe(legacy, query, support, plan, targets, context, ids, training, Budget())
            with self.assertRaisesRegex(ValueError, 'truth binding'):
                mode_probe(net, query, support, plan, 1-targets, context, ids, training, Budget())
            with self.assertRaisesRegex(ValueError, 'full physical'):
                mode_probe(net, query, support, plan, targets[:16], context, ids[:16], training, Budget())


if __name__ == '__main__':
    unittest.main()
