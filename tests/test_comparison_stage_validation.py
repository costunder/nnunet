"""Explicit CPU UNIT fixtures for the real resumable stage evaluator.

Synthetic inputs and mocked device/data boundaries are not training or an
actual-CT evaluation. Metric computation, cursor, RNG and disk reports are real.
"""
from contextlib import contextmanager, ExitStack, redirect_stderr, redirect_stdout
import copy
from io import StringIO
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from hiercp_v1x import comparison_stage_validation as stage
from hiercp_v1x import comparison_runtime
from hiercp_v1x.u_bridge_training import capture_rng, digest


class StageValidationUnitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix='UNIT_stage_validation_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.keys = ['P', 'U:3', 'U:91']
        self.ids = [10, 11, 12]
        self.lookup = {i: dict(index=i, id=f'UNIT_source_{i}', case_id='UNIT_a' if i < 12 else 'UNIT_b',
            source_component=i, positive_center=[0, 1, 2], native_centers=[[j, 1, 2] for j in range(128)])
            for i in self.ids}
        self.state = dict(epoch=8, updates=1208, phase='validation',
                          validation_position=len(self.ids),
                          validation_rows=[{'sample_index': i} for i in self.ids],
                          best={'epoch': 7, 'update': 1057, 'selection_key': [.8, .7, -.2]},
                          history=[{'epoch': 7}], seen_comparisons={'0': ['U:3']})
        self.policy = {'policy_sha256': 'UNIT_policy_sha'}
        self.saved = []
        self.plans = []
        self.loads = []
        self.scored = []
        self.device_rng = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
        self.provider = object()
        test = self

        class UnitNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.]))

            def score_inference_chunked(self, batch, *, local_chunk_size):
                test.assertFalse(self.training)
                test.assertFalse(torch.is_grad_enabled())
                test.assertEqual(local_chunk_size, 8)
                test.assertEqual(batch.graph_keys, test.keys)
                test.scored.extend(batch.bridge_indices)
                test.consume_rng()
                values = {10: [3., 1., 0.], 11: [1., 2., 0.], 12: [0., 0., 0.]}
                return [torch.tensor(values[i]) for i in batch.bridge_indices]

        self.net = UnitNet()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=.001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=40)

    def consume_rng(self):
        random.random()
        np.random.random()
        torch.rand(3)
        self.device_rng = [self.device_rng[0] + 1]

    @contextmanager
    def candidate_plan(self, provider, arm, policy_sha, keys, *, mode, epoch):
        self.assertIs(provider, self.provider)
        self.assertEqual((arm, policy_sha, mode, epoch),
                         ('native', self.policy['policy_sha256'], 'stage_validation', 0))
        self.assertEqual(list(keys), self.keys)
        self.plans.append(list(keys))
        yield SimpleNamespace(candidate_keys=lambda index, arm, epoch, full: list(keys), keys=list(keys))

    def prefetch(self, provider, remaining, arm, epoch, **kwargs):
        self.assertEqual(kwargs['full'], False)
        self.assertEqual(kwargs['training'], False)
        self.assertEqual(kwargs['pin_memory'], False)
        self.assertEqual(kwargs['with_timing'], True)
        self.assertEqual((arm, epoch), ('native', 0))
        for ids in remaining:
            self.loads.append(list(ids))
            self.consume_rng()
            batch = SimpleNamespace(counts=[len(provider.keys)] * len(ids), local_batch_view2=object(),
                                    bridge_indices=tuple(ids), graph_keys=list(provider.keys))
            yield batch, dict(loader_seconds=.001, loader_wait_seconds=.0001)

    def checkpoint(self):
        self.saved.append(copy.deepcopy(self.state))
        return .0001

    @contextmanager
    def boundaries(self, *, prefetch=None):
        # Even CUDA RNG is a deterministic UNIT fixture; no device is touched.
        with ExitStack() as stack:
            stack.enter_context(patch.object(stage, 'candidate_plan', self.candidate_plan))
            stack.enter_context(patch.object(comparison_runtime, '_prefetch', prefetch or self.prefetch))
            stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
            stack.enter_context(patch.object(torch.cuda, 'get_rng_state_all',
                                             side_effect=lambda: [x.clone() for x in self.device_rng]))
            stack.enter_context(patch.object(torch.cuda, 'set_rng_state_all',
                                             side_effect=lambda x: setattr(self, 'device_rng', [y.clone() for y in x])))
            for method in ('reset_peak_memory_stats', 'synchronize'):
                stack.enter_context(patch.object(torch.cuda, method))
            stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
            stack.enter_context(redirect_stderr(StringIO()))
            stack.enter_context(redirect_stdout(StringIO()))
            yield

    def evaluate(self, pause=lambda: False):
        return stage.evaluate_stage_validation(net=self.net, provider=self.provider, state=self.state,
            arm='native', keys=self.keys, policy=self.policy, root=self.root, val_ids=self.ids,
            lookup=self.lookup, physical_batch=2, epochs=40, view_epoch=0, chunk=8, amp=False,
            budget=SimpleNamespace(check=Mock()), checkpoint=self.checkpoint,
            pause_requested=pause, debug=True)

    def test_partial_pause_resume_uses_exact_graph_without_repeat_or_state_mutation(self):
        protected = {key: copy.deepcopy(self.state[key]) for key in ('updates', 'best', 'history', 'seen_comparisons')}
        model_before = digest(self.net.state_dict())
        optimizer_before = digest(self.optimizer.state_dict())
        scheduler_before = digest(self.scheduler.state_dict())
        with self.boundaries():
            rng_before = digest(capture_rng())
            self.assertIsNone(self.evaluate(pause=lambda: self.state.get('stage_validation', {}).get('position') == 2))
            self.assertEqual(digest(capture_rng()), rng_before)
            self.assertEqual(self.state['stage_validation']['position'], 2)
            self.assertEqual(self.scored, [10, 11])
            self.assertEqual([s['stage_validation']['position'] for s in self.saved], [0, 2])
            self.assertFalse((self.root / 'validation_stage_epoch_008.json').exists())
            # Restore the last complete checkpoint, just as an interrupted invocation does.
            self.state = copy.deepcopy(self.saved[-1])
            report = self.evaluate()
            self.assertEqual(digest(capture_rng()), rng_before)
            self.assertEqual(self.scored, self.ids)
            self.assertEqual(self.loads, [[10, 11], [12]])
            self.assertEqual(self.plans, [self.keys, self.keys])
            self.assertEqual([s['stage_validation']['position'] for s in self.saved], [0, 2, 3])
            before_repeat = list(self.scored)
            again = self.evaluate()
            self.assertEqual(self.scored, before_repeat)
            self.assertEqual(again['rows'], report['rows'])
        self.assertEqual({k: self.state[k] for k in protected}, protected)
        self.assertEqual(digest(self.net.state_dict()), model_before)
        self.assertEqual(digest(self.optimizer.state_dict()), optimizer_before)
        self.assertEqual(digest(self.scheduler.state_dict()), scheduler_before)
        self.assertEqual(report['candidate_keys'], self.keys)
        self.assertEqual([row['sample_index'] for row in report['rows']], self.ids)
        self.assertEqual([row['first_P_rank'] for row in report['rows']], [1, 2, 3])
        self.assertAlmostEqual(report['metrics']['mrr'], ((1. + .5) / 2 + 1. / 3) / 2)
        self.assertAlmostEqual(report['metrics']['top1'], .25)
        self.assertTrue(report['actual_subset_graph_forward'])
        self.assertTrue(report['full_held_out_source_evaluation'])
        self.assertFalse(report['full_candidate_evaluation'])
        self.assertFalse(report['used_for_global_best'])
        stored = json.loads((self.root / 'validation_stage_epoch_008.json').read_text())
        self.assertEqual(stored['rows'], report['rows'])
        timing = [json.loads(line) for line in (self.root / 'stage_validation_timing.jsonl').read_text().splitlines()]
        self.assertEqual([r['source_indices'] for r in timing], [[10, 11], [12]])
        self.assertTrue(all(r['candidate_keys'] == self.keys for r in timing))

    def test_forward_error_restores_all_rng_without_advancing_saved_cursor(self):
        with self.boundaries():
            before = digest(capture_rng())
            def fail(*args, **kwargs):
                self.consume_rng()
                raise RuntimeError('UNIT forward failure')
            with patch.object(self.net, 'score_inference_chunked', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'forward failure'):
                    self.evaluate()
            self.assertEqual(digest(capture_rng()), before)
        self.assertEqual(self.state['stage_validation']['position'], 0)
        self.assertEqual(self.state['stage_validation']['rows'], [])
        self.assertEqual([s['stage_validation']['position'] for s in self.saved], [0])

    def test_initial_validation_uses_epoch_zero_and_does_not_select_best(self):
        self.state['phase'] = 'initial_validation'
        best = copy.deepcopy(self.state['best'])
        with self.boundaries():
            report = self.evaluate()
        self.assertEqual(report['epoch'], 0)
        self.assertEqual(self.state['best'], best)
        self.assertTrue((self.root / 'validation_stage_epoch_000.json').is_file())

    def test_cursor_and_policy_tampering_fail_before_any_additional_forward(self):
        with self.boundaries():
            self.evaluate(pause=lambda: self.state.get('stage_validation', {}).get('position') == 2)
            clean = copy.deepcopy(self.state)
            changes = [('position', 1), ('keys', ['P', 'U:2', 'U:91']),
                       ('policy_sha256', 'OTHER'), ('view_epoch', 1), ('update', 1209)]
            for field, value in changes:
                self.state = copy.deepcopy(clean)
                self.state['stage_validation'][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.evaluate()
            self.state = copy.deepcopy(clean)
            self.state['stage_validation']['rows'][0]['sample_index'] = 999
            with self.assertRaisesRegex(ValueError, 'repeat or skip'):
                self.evaluate()
        self.assertEqual(self.scored, [10, 11])

    def test_incomplete_graph_missing_view_or_wrong_source_never_becomes_a_metric(self):
        for field, value in (('counts', [2, 3]), ('local_batch_view2', None), ('bridge_indices', (11, 10))):
            self.state.pop('stage_validation', None)
            def malformed(*args, **kwargs):
                for batch, timing in self.prefetch(*args, **kwargs):
                    setattr(batch, field, value)
                    yield batch, timing
            with self.subTest(field=field), self.boundaries(prefetch=malformed):
                rng_before = digest(capture_rng())
                with self.assertRaises(ValueError):
                    self.evaluate()
                self.assertEqual(digest(capture_rng()), rng_before)
                self.assertEqual(self.state['stage_validation']['position'], 0)
        self.assertFalse(self.scored)


if __name__ == '__main__':
    unittest.main()
