"""CPU DEBUG: narrow metadata repair of the actual native saver source.

The method body below was read from the server native base SHA
7096efb2040135eb60df3c8bf39cdbcc373e299671fcee365b3aff2f4da4dfbd.
State fixtures verify serialization, never represent a trained clinical model.
"""
from __future__ import annotations

import copy
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
from hiercp_v1x import basic_cp_completed_epoch_checkpoint as completed

_SAVED = []


class DebugTorch:
    @staticmethod
    def save(payload, filename):
        _SAVED.append((payload, filename))


torch = DebugTorch


class OptimizedModule:
    def __init__(self, original): self._orig_mod = original


class NativeCheckpointFixture:
    # Exact actual native save_checkpoint body; fixtures adapt external APIs.
    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')


class MissingOffsetFixture:
    def save_checkpoint(self, filename):
        checkpoint = {'current_epoch': self.current_epoch}
        torch.save(checkpoint, filename)


class DuplicateOffsetFixture:
    def save_checkpoint(self, filename):
        checkpoint = {'current_epoch': self.current_epoch + 1, 'current_epoch': self.current_epoch + 1}
        torch.save(checkpoint, filename)


class CompletedEpochCheckpointTests(unittest.TestCase):
    def setUp(self):
        _SAVED.clear()

    def trainer(self):
        trainer = NativeCheckpointFixture(); trainer.current_epoch = 3
        trainer.local_rank = 0; trainer.disable_checkpointing = False; trainer.is_ddp = False
        network = {'encoder.weight': np.arange(8, dtype=np.float32).reshape(2, 4),
                   'decoder.bias': np.asarray([.1, .2], dtype=np.float32)}
        optimizer = {'state': {0: {'momentum_buffer': np.asarray([4., 8.])}},
                     'param_groups': [{'params': [0, 1], 'lr': .009, 'momentum': .99}]}
        scaler = {'scale': 65536., '_growth_tracker': 45, 'growth_factor': 2., 'backoff_factor': .5, 'growth_interval': 2000}
        trainer.network = SimpleNamespace(state_dict=lambda: network)
        trainer.optimizer = SimpleNamespace(state_dict=lambda: optimizer)
        trainer.grad_scaler = SimpleNamespace(state_dict=lambda: scaler)
        logging = {name: [1., 2., 3.] for name in completed._LOGGER_FIELDS}
        logging['epoch_start_timestamps'] = [1., 3., 5.]
        logging['epoch_end_timestamps'] = [2., 4., 6.]
        calls = []
        trainer.logger = SimpleNamespace(my_fantastic_logging=logging,
            get_checkpoint=lambda: (calls.append('logger'), copy.deepcopy(logging))[1])
        trainer._logger_calls = calls
        trainer._best_ema = .73; trainer.my_init_kwargs = {'plans': {'DEBUG': True}, 'fold': 0}
        trainer.inference_allowed_mirroring_axes = (0, 1, 2)
        trainer._messages = []; trainer.print_to_log_file = trainer._messages.append
        return trainer

    def test_completed_save_changes_only_epoch_field_preserves_all_native_state(self):
        trainer = self.trainer(); original_method = trainer.save_checkpoint.__func__
        original_code = original_method.__code__
        trainer.save_checkpoint('DEBUG_NATIVE_ORIGINAL')
        original = _SAVED[-1][0]
        completed.save_completed_epoch(trainer, 'DEBUG_COMPLETED_EPOCH')
        corrected, filename = _SAVED[-1]
        self.assertEqual(filename, 'DEBUG_COMPLETED_EPOCH')
        self.assertEqual(original['current_epoch'], 4); self.assertEqual(corrected['current_epoch'], 3)
        self.assertEqual(trainer.current_epoch, 3)
        self.assertIs(trainer.save_checkpoint.__func__, original_method)
        self.assertIs(original_method.__code__, original_code)
        self.assertEqual(set(corrected), set(original))
        for key in corrected:
            if key == 'current_epoch': continue
            if key in ('network_weights', 'optimizer_state', 'grad_scaler_state', 'init_args'):
                self.assertIs(corrected[key], original[key])
            else:
                self.assertEqual(corrected[key], original[key])
        self.assertEqual(trainer._logger_calls, ['logger', 'logger'])
        trainer.save_checkpoint('DEBUG_BEST_UNCHANGED')
        self.assertEqual(_SAVED[-1][0]['current_epoch'], 4)

    def test_private_clone_cached_and_original_globals_untouched(self):
        function = NativeCheckpointFixture.save_checkpoint
        first = completed._private_completed_saver(function)
        second = completed._private_completed_saver(function)
        self.assertIs(first, second)
        self.assertIsNot(first.__globals__, function.__globals__)
        for key, value in function.__globals__.items():
            if key == function.__name__: continue
            self.assertIs(first.__globals__[key], value)
        self.assertEqual(first.__defaults__, function.__defaults__)
        self.assertEqual(first.__annotations__, function.__annotations__)

    def test_all_native_rank_disable_ddp_and_optimized_module_guards_are_retained(self):
        trainer = self.trainer(); trainer.local_rank = 1
        completed.save_completed_epoch(trainer, 'DEBUG_NONZERO_RANK')
        self.assertEqual(_SAVED, []); self.assertEqual(trainer._logger_calls, [])
        trainer.local_rank = 0; trainer.disable_checkpointing = True
        completed.save_completed_epoch(trainer, 'DEBUG_DISABLED')
        self.assertEqual(_SAVED, []); self.assertEqual(trainer._messages, ['No checkpoint written, checkpointing is disabled'])
        trainer.disable_checkpointing = False; trainer.is_ddp = True
        trainer.network = SimpleNamespace(module=OptimizedModule(trainer.network))
        completed.save_completed_epoch(trainer, 'DEBUG_DDP_COMPILED')
        self.assertEqual(_SAVED[-1][0]['current_epoch'], 3)
        self.assertIn('encoder.weight', _SAVED[-1][0]['network_weights'])

    def test_native_none_scaler_behavior_unchanged(self):
        trainer = self.trainer(); trainer.grad_scaler = None
        completed.save_completed_epoch(trainer, 'DEBUG_NO_SCALER')
        self.assertIsNone(_SAVED[-1][0]['grad_scaler_state'])

    def test_incomplete_or_inconsistent_logger_rejected_before_any_save(self):
        trainer = self.trainer()
        for field in completed._LOGGER_FIELDS:
            values = trainer.logger.my_fantastic_logging[field]
            trainer.logger.my_fantastic_logging[field] = values[:-1]
            with self.assertRaisesRegex(ValueError, 'Every actual logger array'):
                completed.save_completed_epoch(trainer, 'DEBUG_PARTIAL')
            trainer.logger.my_fantastic_logging[field] = values
        trainer.current_epoch = 0
        with self.assertRaisesRegex(ValueError, 'positive completed'):
            completed.save_completed_epoch(trainer, 'DEBUG_ZERO_EPOCH')
        self.assertEqual(_SAVED, [])

    def test_missing_fields_and_unfinished_timestamps_rejected(self):
        trainer = self.trainer(); del trainer.logger.my_fantastic_logging['val_losses']
        with self.assertRaisesRegex(ValueError, 'all eight'):
            completed.save_completed_epoch(trainer, 'DEBUG_MISSING')
        for bad in (0., float('nan'), float('inf')):
            trainer = self.trainer(); trainer.logger.my_fantastic_logging['epoch_end_timestamps'][-1] = bad
            with self.assertRaisesRegex(ValueError, 'finite ordered'):
                completed.save_completed_epoch(trainer, 'DEBUG_BAD_END')
        self.assertEqual(_SAVED, [])

    def test_unexpected_or_duplicate_serialization_fields_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'exactly self.current_epoch'):
            completed._private_completed_saver(MissingOffsetFixture.save_checkpoint)
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            completed._private_completed_saver(DuplicateOffsetFixture.save_checkpoint)

    def test_closure_wrapped_native_guard_cannot_be_bypassed(self):
        trainer = self.trainer(); original = trainer.save_checkpoint
        def guard_wrapper(self, filename):
            return original(filename)
        trainer.save_checkpoint = MethodType(guard_wrapper, trainer)
        with self.assertRaisesRegex(ValueError, 'wrapped guards cannot be bypassed'):
            completed.save_completed_epoch(trainer, 'DEBUG_CLOSURE')
        self.assertEqual(_SAVED, [])


if __name__ == '__main__': unittest.main()
