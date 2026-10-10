"""CPU DEBUG AMP simulations, never native CUDA throughput or training results."""
import copy
from pathlib import Path
import tempfile
from types import MethodType
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hiercp_v1x import v24_matched_native_gradient as runtime
from hiercp_v1x import v24_nnunet_cp as pipeline
from tests import test_v24_native_amp_retry as fixtures


SOURCE = '''# CPU DEBUG ownership fixture; never a production trainer
class _nnUNetTrainer_250epochs_OnlineCP:
    def __init__(self):
        self._online_cp_events = 0
        self._online_cp_samples = 0
        self._online_schedule_hash = 0xCBF29CE484222325
        self._online_train_loader = None
        self._online_train_augmenter = None
        self._online_active_epoch = None
        self._online_process_count = None
    def train_step(self, batch):
        self._online_cp_events += 1
        self._online_cp_samples += 1
        self._online_schedule_hash ^= 1
        self._online_schedule_hash *= 1099511628211
        return super().train_step(batch)
'''


class HistoricalNativeTrainerDebug(fixtures.NativeTrainerDebug):
    """The established CPU AMP simulation with the three historical counters."""
    def train_step(self, batch):
        self.calls += 1
        flags = batch.pop('online_cp_applied'); tokens = batch.pop('online_cp_schedule_token')
        self._online_cp_events += int(flags.sum()); self._online_cp_samples += len(flags)
        self._online_schedule_hash += int(tokens.sum())
        draw = torch.rand(4)
        self.observed.append(dict(data_id=id(batch['data']), target_id=id(batch['target']), draw=draw.clone(),
            flags=flags.copy(), before=[p.detach().clone() for p in self.network.parameters()]))
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.network(batch['data'])
        loss = sum(weight * (output-batch['target']).square().mean()
            for weight, output in zip(self.loss.weight_factors, outputs) if weight > 0)
        loss.backward()
        overflow = self.calls == 1 and self.mode not in ('healthy', 'finite_no_update', 'missing_active')
        if self.mode == 'missing_active': self.network.decoder.seg_layers[1].weight.grad = None
        if overflow:
            next(self.network.parameters()).grad.fill_(float('inf'))
            if self.mode != 'unchanged_scale': self.grad_scaler.value /= 2
            if self.mode == 'changed_on_overflow':
                with torch.no_grad(): next(self.network.parameters()).add_(1)
        elif self.mode != 'finite_no_update': self.optimizer.step()
        result = loss.detach().numpy()
        if self.mode == 'nonfinite_forward': result = np.asarray(float('nan'))
        return dict(loss=result)


class MatchedAMPDebug(fixtures.NativeAMPAdmissionDebug):
    """All nine original AMP invariants plus strict three-counter proof checks."""
    def setUp(self):
        super().setUp()
        folder = Path(__file__).absolute().parents[1] / 'outputs'; folder.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='DEBUG_matched_AMP_', dir=folder)
        self.source = Path(self.temp.name) / 'DEBUG_historical_trainer.py'
        self.source.write_text(SOURCE, encoding='utf8')

    def tearDown(self):
        self.temp.cleanup()

    def clone(self):
        return runtime._clone_step_debug(pipeline._native_clone_step_with_amp_retry,
            historical_trainer_source=self.source, fixture_sha256=runtime._sha(self.source))

    def run_update(self, trainer):
        if hasattr(trainer, '_online_native_transport'): del trainer._online_native_transport
        trainer.train_step = MethodType(HistoricalNativeTrainerDebug.train_step, trainer)
        parameters = dict(trainer.network.named_parameters())
        before = {name:p.detach().clone() for name,p in parameters.items()}
        with patch('torch.cuda.is_available', return_value=False):
            return self.clone()(trainer, self.batch, parameters, before)

    def test_same_real_batch_RNG_and_CP_counters_once_after_native_skip(self):
        trainer = fixtures.NativeTrainerDebug(); keys = set(self.batch)
        result, proof = self.run_update(trainer)
        self.assertEqual(trainer.calls, 2); self.assertTrue(np.isfinite(result['loss']))
        self.assertEqual(set(self.batch), keys)
        self.assertEqual([row['data_id'] for row in trainer.observed], [id(self.batch['data'])]*2)
        self.assertEqual([row['target_id'] for row in trainer.observed], [id(self.batch['target'])]*2)
        torch.testing.assert_close(trainer.observed[0]['draw'], trainer.observed[1]['draw'], rtol=0, atol=0)
        self.assertEqual((trainer._online_cp_events, trainer._online_cp_samples, trainer._online_schedule_hash), (4,4,129))
        self.assertFalse(hasattr(trainer, '_online_native_transport'))
        self.assertEqual(len(proof['AMP_overflow_attempts']), 1)
        self.assertEqual(proof['AMP_overflow_attempts'][0]['scale_before'], 64.)
        self.assertEqual(proof['AMP_overflow_attempts'][0]['scale_after'], 32.)
        self.assertEqual(proof['matched_CP_counter_runtime']['matched_counter_names'], list(runtime.MATCHED_COUNTERS))
        self.assertTrue(proof['matched_CP_counter_runtime']['instruction_bytes_preserved'])
        self.assertTrue(proof['matched_CP_counter_runtime']['original_loss_gradient_clipping_scaler_optimizer_and_AMP_retry_equations_preserved'])
        for before, after in zip(trainer.observed[0]['before'], trainer.observed[1]['before']):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_original_instruction_code_every_other_constant_and_globals_are_preserved(self):
        original = pipeline._native_clone_step_with_amp_retry
        before = original.__code__; old_helper = original.__globals__[runtime.gradient.HELPER]
        clone = self.clone()
        self.assertEqual(clone.__code__.co_code, before.co_code)
        positions = [index for index,(old,new) in enumerate(zip(before.co_consts,clone.__code__.co_consts)) if old != new]
        self.assertEqual(len(positions), 1)
        self.assertEqual(before.co_consts[positions[0]], runtime.ORIGINAL_COUNTERS)
        self.assertEqual(clone.__code__.co_consts[positions[0]], runtime.MATCHED_COUNTERS)
        self.assertEqual(clone.__code__.replace(co_consts=before.co_consts), before)
        self.assertIs(original.__globals__[runtime.gradient.HELPER], old_helper)
        self.assertIs(original.__code__, before)

    def test_historical_source_digest_and_exact_counter_ownership_are_required(self):
        with self.assertRaisesRegex(ValueError, 'source SHA'):
            runtime.clone_step(pipeline._native_clone_step_with_amp_retry, historical_trainer_source=self.source)
        variants = [SOURCE.replace('_online_schedule_hash', '_online_native_transport'),
            SOURCE + '\n# fixture additional class\nclass Other:\n    def transport(self): return self._online_native_transport\n',
            SOURCE.replace('        self._online_cp_samples = 0\n', ''),
            SOURCE.replace('        return super().train_step(batch)',
                '        self._online_active_epoch = 1\n        return super().train_step(batch)'),
            SOURCE.replace('class _nnUNetTrainer_250epochs_OnlineCP:', 'class Other:')]
        for source in variants:
            self.source.write_text(source, encoding='utf8')
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'counter'):
                self.clone()

    def test_runtime_source_tampering_is_rejected_before_actual_forward(self):
        clone = self.clone(); self.source.write_text(SOURCE + '# changed after admission\n', encoding='utf8')
        trainer = fixtures.NativeTrainerDebug('healthy'); del trainer._online_native_transport
        trainer.train_step = MethodType(HistoricalNativeTrainerDebug.train_step, trainer)
        parameters = dict(trainer.network.named_parameters()); before = {k:p.detach().clone() for k,p in parameters.items()}
        with self.assertRaisesRegex(ValueError, 'source changed'): clone(trainer, self.batch, parameters, before)
        self.assertEqual(trainer.calls, 0)

    def test_raw_transport_counter_cannot_be_silently_ignored(self):
        clone = self.clone(); trainer = fixtures.NativeTrainerDebug('healthy')
        parameters = dict(trainer.network.named_parameters()); before = {k:p.detach().clone() for k,p in parameters.items()}
        with self.assertRaisesRegex(ValueError, 'Raw-target transport'): clone(trainer, self.batch, parameters, before)
        self.assertEqual(trainer.calls, 0)

    def test_actual_constructor_identity_is_proved_for_production_clone_entry(self):
        trainer = fixtures.NativeTrainerDebug('healthy'); del trainer._online_native_transport
        clone = runtime._clone_step(pipeline._native_clone_step_with_amp_retry,self.source,
            expected_sha256=runtime._sha(self.source),verify_actual_constructor=True)
        parameters = dict(trainer.network.named_parameters()); before = {k:p.detach().clone() for k,p in parameters.items()}
        with self.assertRaisesRegex(ValueError, 'inherit the admitted original historical constructor'):
            clone(trainer, self.batch, parameters, before)
        self.assertEqual(trainer.calls, 0)

    def test_missing_historical_cp_counter_is_explicit_error(self):
        trainer = fixtures.NativeTrainerDebug('healthy'); del trainer._online_native_transport; del trainer._online_cp_samples
        trainer.train_step = MethodType(HistoricalNativeTrainerDebug.train_step, trainer)
        parameters = dict(trainer.network.named_parameters()); before = {k:p.detach().clone() for k,p in parameters.items()}
        with self.assertRaisesRegex(ValueError, 'CP counter ownership'): self.clone()(trainer, self.batch, parameters, before)
        self.assertEqual(trainer.calls, 0)


if __name__ == '__main__':
    unittest.main()
