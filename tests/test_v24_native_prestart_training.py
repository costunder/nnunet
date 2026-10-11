"""CPU DEBUG real pre-start hooks for GPU4 and private early-stop GPU5.

These fixtures exercise production training=True/on_train_start expressions.
The model, loader and process observations are explicit CPU DEBUG dependencies;
no optimization, CUDA, production file or remote process is touched.
"""
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

import psutil

from hiercp_v1x import v24_early_stop_downstream as early
from hiercp_v1x import v24_matched_basic_cp_runtime as runtime
from hiercp_v1x import v24_nnunet_cp as resource
from tests import test_v24_matched_basic_cp_runtime as fixtures


REAL_BUDGET = resource.require_project_budget
GIB = 2**30


class _NativePrestartTrainingCases:
    gpu = None

    def setUp(self):
        # Construct the actual AST-import adapter before the shared fixture
        # mocks importlib. Reuse its cache if another CPU test already made it.
        adapted = early.private_modules() if self.gpu == 5 else None
        originals = (runtime, early, resource)
        self.source_hashes = {Path(module.__file__): hashlib.sha256(
            Path(module.__file__).read_bytes()).hexdigest() for module in originals}
        self.addCleanup(self._assert_original_sources_preserved)
        self.fixture = fixtures.MatchedNativeTrainingStartCpuDebugTests(
            methodName='test_full_training_start_executes_project_bounded_observation_and_preserves_contract')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        native = json.loads(f.native_path.read_text(encoding='utf8'))
        native['physical_GPU'] = self.gpu
        f.binding['overlay']['pin']['physical_GPU'] = self.gpu
        f.native_path.write_text(json.dumps(native), encoding='utf8')
        self.runtime = runtime
        self.stage = f.stage
        if adapted is not None:
            self.runtime = adapted['EARLY_RUNTIME']
            self.stage = f.stack.enter_context(patch.object(
                early, 'early_stage_guard', return_value=f.access))
            for name, value in (
                ('validate_overlay', Mock(return_value=f.binding)),
                ('install_matched_overlay', f.overlay_adapter),
                ('_guard_binding', f.binding_guard),
                ('runtime_observation', runtime.runtime_observation),
            ):
                f.stack.enter_context(patch.object(self.runtime, name, value))
            self.assertIsNot(self.runtime, runtime)
            self.assertIsNot(self.runtime._BINDINGS, runtime._BINDINGS)
            self.assertFalse(adapted['proofs']['runtime'][
                'model_geometry_loss_optimizer_and_loader_expressions_changed'])
            self.assertFalse(adapted['proofs']['runtime']['original_module_globals_changed'])

    def _assert_original_sources_preserved(self):
        for path, expected in self.source_hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, str(path))

    def _observed_budget(self, *, parent_rss=2*GIB, child_rss=GIB, affinity=None):
        """Mock process measurements only; execute the real resource function."""
        parent = Mock()
        child = Mock()
        parent.cpu_affinity.return_value = [32, 33, 34, 35] if affinity is None else affinity
        parent.memory_info.return_value = types.SimpleNamespace(rss=parent_rss)
        child.memory_info.return_value = types.SimpleNamespace(rss=child_rss)
        child.is_running.return_value = True
        parent.children.return_value = [child]
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(psutil, 'Process', return_value=parent))
        budget = stack.enter_context(patch.object(resource, 'require_project_budget', wraps=REAL_BUDGET))
        return parent, child, budget

    def test_real_budget_and_training_start_publish_full_contract(self):
        f = self.fixture
        parent, child, budget = self._observed_budget()
        proofs = self.runtime.install_native_runtime(f.native_path, training=True)
        self.assertEqual(f.trainer.on_train_start(), 'DEBUG native started')
        self.stage.assert_called_once_with(stage_action='train')
        f.access.assert_called_once_with()
        f.initialize.assert_called_once_with()
        f.original_start.assert_called_once_with(f.trainer)
        f.best_adapter.assert_called_once_with()
        f.direct_adapter.assert_called_once_with(f.cls, f.access)
        f.checkpoint_adapter.assert_called_once()
        f.binding_guard.assert_called_once_with(f.binding)
        self.assertEqual(budget.call_count, 2)
        path, report = f.publish.call_args.args
        self.assertEqual(path, f.root/'actual_nnunet_configuration.json')
        expected = dict(train_cases=105, validation_cases=26, epochs=250,
            physical_batch=2, gradient_accumulation=1, effective_batch=2,
            patch_size=[128]*3, CPU_workers=4, cp_probability=.5,
            source_policy='same_patient_uniform_original_sources_per_visit',
            candidate_count=128, source_entries=642, eligible_cases=81,
            validation_checkpoint='checkpoint_best.pth', debug=False,
            full_training_complete=False, model_weights_fresh=True,
            train_steps_per_epoch=250, validation_steps_per_epoch=50)
        for name, value in expected.items():
            with self.subTest(field=name):
                self.assertEqual(report[name], value)
        self.assertEqual(report['runtime_proofs'], proofs)
        self.assertEqual(report['resources']['actual_affinity'], [32, 33, 34, 35])
        self.assertEqual(report['resources']['actual_process_and_children_RSS_bytes'], 3*GIB)
        self.assertEqual(report['resources']['process_and_children_RSS_bytes'], 48*GIB)
        self.assertTrue(report['resources']['host_available_resources_are_not_project_allocation'])
        self.assertEqual(parent.children.call_args.kwargs, {'recursive': True})
        self.assertTrue(child.is_running.called)
        f.trainer.on_train_epoch_start()
        f.original_epoch.assert_called_once_with(f.trainer)
        self.assertEqual(budget.call_count, 3)

    def test_actual_resource_limits_block_start_before_loader_or_report(self):
        f = self.fixture
        parent, child, budget = self._observed_budget(parent_rss=48*GIB, child_rss=GIB)
        self.runtime.install_native_runtime(f.native_path, training=True)
        with self.assertRaisesRegex(MemoryError, 'process\\+DA children exceed48GiB'):
            f.trainer.on_train_start()
        f.original_start.assert_not_called()
        f.publish.assert_not_called()
        parent.cpu_affinity.return_value = [32, 33, 34]
        with self.assertRaisesRegex(ValueError, 'exactly four logical CPU cores'):
            f.trainer.on_train_start()
        f.original_start.assert_not_called()
        f.publish.assert_not_called()
        self.assertEqual(budget.call_count, 2)

    def test_post_loader_resource_failure_is_visible_without_configuration(self):
        f = self.fixture
        parent, child, budget = self._observed_budget()
        self.runtime.install_native_runtime(f.native_path, training=True)
        def started(trainer):
            parent.memory_info.return_value = types.SimpleNamespace(rss=48*GIB)
            return 'DEBUG loader started'
        f.original_start.side_effect = started
        with self.assertRaisesRegex(MemoryError, 'process\\+DA children exceed48GiB'):
            f.trainer.on_train_start()
        f.original_start.assert_called_once_with(f.trainer)
        self.assertEqual(budget.call_count, 2)
        f.publish.assert_not_called()
        self.assertFalse((f.root/'actual_nnunet_configuration.json').exists())

    def test_reduced_training_or_worker_contract_is_not_published(self):
        f = self.fixture
        self._observed_budget()
        self.runtime.install_native_runtime(f.native_path, training=True)
        for name, changed, original in (
            ('num_epochs', 249, 250), ('batch_size', 1, 2),
            ('configuration_manager', types.SimpleNamespace(patch_size=[127]*3),
                f.trainer.configuration_manager),
            ('do_split', Mock(return_value=(f.split['outer_train'][:-1], f.split['outer_val'])),
                f.trainer.do_split),
        ):
            with self.subTest(field=name):
                setattr(f.trainer, name, changed)
                with self.assertRaisesRegex(ValueError, 'full105/26/B2/250/128cube'):
                    f.trainer.on_train_start()
                setattr(f.trainer, name, original)
                f.original_start.assert_not_called()
                f.publish.assert_not_called()
        with patch.dict(os.environ, nnUNet_n_proc_DA='3'):
            with self.assertRaisesRegex(ValueError, 'four native data workers'):
                f.trainer.on_train_start()
        f.original_start.assert_called_once_with(f.trainer)
        f.publish.assert_not_called()


class GPU4OrdinaryNativePrestartCpuDebug(_NativePrestartTrainingCases, unittest.TestCase):
    gpu = 4


class GPU5EarlyPrivateNativePrestartCpuDebug(_NativePrestartTrainingCases, unittest.TestCase):
    gpu = 5

    def test_early_import_redirect_uses_actual_resource_function_without_snapshot(self):
        f = self.fixture
        self.assertFalse(hasattr(resource, 'snapshot'))
        _, _, budget = self._observed_budget()
        self.assertIs(early.__getattr__('require_project_budget'), resource.require_project_budget)
        self.runtime.install_native_runtime(f.native_path, training=True)
        self.assertEqual(f.trainer.on_train_start(), 'DEBUG native started')
        self.assertEqual(budget.call_count, 2)
        self.assertIs(budget._mock_wraps, REAL_BUDGET)
        f.publish.assert_called_once()


if __name__ == '__main__':
    unittest.main()
