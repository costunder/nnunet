"""CPU UNIT integration of the actual runtime checkpoint closure, not training.

The tiny CPU model exercises real AdamW/scheduler/RNG checkpoint restore only.
It is not a replacement model, CUDA test, or full research training result.
"""
from __future__ import annotations

from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import CodeType, FunctionType
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hiercp_v1x import comparison_execution as adapter
from hiercp_v1x import comparison_runtime as runtime
from hiercp_v1x import comparison_training as comparison
from hiercp_v1x import u_bridge_training as engine
from hiercp_v1x.comparison_checkpoint import ComparisonCheckpointWriter


def _cell(value):
    return (lambda: value).__closure__[0]


def _actual_checkpoint(run, values):
    """Bind the shipped nested checkpoint code to real CPU UNIT state objects."""
    codes = [code for code in run.__code__.co_consts
             if isinstance(code, CodeType) and code.co_name == 'checkpoint']
    if len(codes) != 1:
        raise AssertionError('Expected the actual single run_arm checkpoint closure')
    code = codes[0]
    return FunctionType(code, run.__globals__, 'checkpoint', ('RUNNING', False),
                        tuple(_cell(values[name]) for name in code.co_freevars))


def _update(net, optimizer):
    optimizer.zero_grad(set_to_none=True)
    values = torch.tensor([[0.25, -0.5, 1., 2.], [1., 0.5, -0.25, 0.]])
    net(values).square().mean().backward()
    optimizer.step()


class ActualCheckpointIntegrationUnit(unittest.TestCase):
    def setUp(self):
        self.cpu_scope = patch('torch.cuda.is_available', return_value=False)
        self.cpu_scope.start()
        self.addCleanup(self.cpu_scope.stop)
        self.original_rng = engine.capture_rng()
        self.addCleanup(engine.restore_rng, self.original_rng)
        self.directory = TemporaryDirectory(prefix='UNIT_checkpoint_integration_',
                                            dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.directory.cleanup)
        self.frozen_run = engine.run_arm
        self.execution = adapter.comparison_execution()
        self.execution.__enter__()
        self.addCleanup(self.execution.__exit__, None, None, None)

    def build(self, arm=None, suffix='actual'):
        if arm is None:
            run = engine.run_arm
            config = {}
        else:
            policy = comparison.arm_policy(arm)
            run = comparison._engine_functions(policy)['run_arm']
            config = comparison._bind_config({}, policy)
        self.assertIs(run.__code__, runtime.run_arm.__code__)
        self.assertIs(run.__globals__['ComparisonCheckpointWriter'], ComparisonCheckpointWriter)
        net = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        _update(net, optimizer)
        state = dict(epoch=1, phase='validation', position=1, order=[8], updates=1,
                     attempts=1, overflows=0, history=[], train_rows=[{'sample_index': 8}],
                     validation_position=0, validation_rows=[], invocation_status='RUNNING',
                     initial_validation={'UNIT': True}, best=None, connected=['weight', 'bias'],
                     seen_comparisons={'8': []}, validation_active_seconds=0., epoch_active_seconds=1.)
        root = Path(self.directory.name) / ((arm or 'v18') + '_' + suffix)
        root.mkdir()
        values = dict(binding_hash='UNIT_exact_binding', checkpoint_stats={},
                      checkpoint_writer=ComparisonCheckpointWriter(), config=config,
                      generator=torch.Generator().manual_seed(2045), net=net,
                      optimizer=optimizer, root=root, scaler=scaler,
                      scheduler=scheduler, state=state)
        return run, values, _actual_checkpoint(run, values)

    def load(self, root, name='checkpoint_latest.pt'):
        saved = torch.load(root / name, map_location='cpu', weights_only=False)
        checksum = saved.pop('content_sha256')
        self.assertEqual(checksum, engine.digest(saved))
        self.assertEqual(saved['identity_sha256'], 'UNIT_exact_binding')
        return saved

    def test_actual_v18_and_all_v19_checkpoint_values_match_frozen_path(self):
        for arm in (None, *comparison.ARMS):
            with self.subTest(arm=arm):
                run, values, checkpoint = self.build(arm)
                original_globals = run.__globals__
                original = FunctionType(self.frozen_run.__code__, original_globals,
                                        'run_arm', self.frozen_run.__defaults__)
                legacy_values = dict(values)
                legacy_values['root'] = values['root'].with_name(values['root'].name + '_legacy')
                legacy_values['root'].mkdir()
                legacy_checkpoint = _actual_checkpoint(original, legacy_values)
                before_rng = engine.digest(engine.capture_rng())
                checkpoint(status='PAUSED', best=True)
                legacy_checkpoint(status='PAUSED', best=True)
                actual = self.load(values['root'])
                legacy = self.load(legacy_values['root'])
                self.assertEqual(engine.digest(actual), engine.digest(legacy))
                self.assertEqual(engine.digest(actual), engine.digest(self.load(values['root'], 'checkpoint_best.pt')))
                self.assertEqual(before_rng, engine.digest(engine.capture_rng()))
                self.assertEqual(actual['format'], run.__globals__['FORMAT'])
                if arm is None:
                    self.assertNotIn('comparison_policy', actual)
                else:
                    self.assertEqual(actual['comparison_policy'], comparison.arm_policy(arm))

    def test_actual_eval_reuse_and_epoch_transition_preserve_cursor_and_scheduler(self):
        _, values, checkpoint = self.build('native_listwise')
        checkpoint()
        self.assertFalse(values['checkpoint_stats']['static_snapshot_reused'])
        state = values['state']
        state['validation_position'] = 1
        state['validation_rows'].append({'sample_index': 10, 'UNIT_metric': 0.75})
        torch.rand(4)
        torch.randperm(5, generator=values['generator'])
        checkpoint()
        self.assertTrue(values['checkpoint_stats']['static_snapshot_reused'])
        saved = self.load(values['root'])
        self.assertEqual(saved['state']['validation_position'], 1)
        self.assertEqual(saved['state']['validation_rows'], state['validation_rows'])
        self.assertEqual(engine.digest(saved['rng']), engine.digest(engine.capture_rng()))
        self.assertTrue(torch.equal(saved['shuffle_generator'], values['generator'].get_state()))
        engine.validate_resume_progress(saved['state'], [8], [10, 11], 1, 40, values['scheduler'])
        values['scheduler'].step()
        state.update(epoch=2, phase='training', position=0, order=None, train_rows=[],
                     validation_position=0, validation_rows=[], history=[{'epoch': 1}])
        checkpoint(best=True)
        self.assertFalse(values['checkpoint_stats']['static_snapshot_reused'])
        saved = self.load(values['root'])
        self.assertEqual(saved['scheduler']['last_epoch'], 1)
        self.assertEqual(saved['optimizer']['param_groups'][0]['lr'], values['optimizer'].param_groups[0]['lr'])
        engine.validate_resume_progress(saved['state'], [8], [10, 11], 1, 40, values['scheduler'])
        _update(values['net'], values['optimizer'])
        state.update(position=1, order=[8], updates=2, attempts=2,
                     train_rows=[{'sample_index': 8}])
        checkpoint()
        self.assertFalse(values['checkpoint_stats']['static_snapshot_reused'])
        saved = self.load(values['root'])
        self.assertEqual(engine.digest(saved['model']), engine.digest(values['net'].state_dict()))
        self.assertEqual(engine.digest(saved['optimizer']), engine.digest(values['optimizer'].state_dict()))

    def test_actual_saved_state_restores_adamw_scheduler_scaler_and_next_rng_values(self):
        for arm in (None, 'native_listwise'):
            with self.subTest(arm=arm):
                _, values, checkpoint = self.build(arm)
                checkpoint()
                saved = self.load(values['root'])
                expected_random = (random.random(), np.random.random(), torch.rand(5),
                                   torch.randperm(7, generator=values['generator']))
                net = torch.nn.Linear(4, 2)
                optimizer = torch.optim.AdamW(net.parameters(), lr=0.3, weight_decay=0.2)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
                scaler = torch.amp.GradScaler('cuda', enabled=False)
                generator = torch.Generator()
                # Same restoration order as the production resume block.
                net.load_state_dict(saved['model'], strict=True)
                optimizer.load_state_dict(saved['optimizer'])
                scheduler.load_state_dict(saved['scheduler'])
                scaler.load_state_dict(saved['scaler'])
                generator.set_state(saved['shuffle_generator'])
                engine.restore_rng(saved['rng'])
                engine.validate_resume_progress(saved['state'], [8], [10, 11], 1, 40, scheduler)
                receipt = engine.restore_optimizer_history(optimizer, saved['state']['updates'])
                self.assertEqual(receipt['verified_genuine_updates'], 1)
                restored_random = (random.random(), np.random.random(), torch.rand(5),
                                   torch.randperm(7, generator=generator))
                self.assertEqual(expected_random[:2], restored_random[:2])
                self.assertTrue(torch.equal(expected_random[2], restored_random[2]))
                self.assertTrue(torch.equal(expected_random[3], restored_random[3]))
                self.assertEqual(scaler.state_dict(), values['scaler'].state_dict())
                self.assertEqual(scheduler.state_dict(), values['scheduler'].state_dict())
                _update(values['net'], values['optimizer'])
                _update(net, optimizer)
                self.assertEqual(engine.digest(net.state_dict()), engine.digest(values['net'].state_dict()))
                self.assertEqual(engine.digest(optimizer.state_dict()), engine.digest(values['optimizer'].state_dict()))
                values['scheduler'].step()
                scheduler.step()
                self.assertEqual(scheduler.state_dict(), values['scheduler'].state_dict())


if __name__ == '__main__':
    unittest.main()
