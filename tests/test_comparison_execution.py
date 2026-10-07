"""CPU UNIT checks for source-preserving runtime bindings; no CT/CUDA run."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from hiercp_v1x import comparison_execution as adapter
from hiercp_v1x import comparison_experiment
from hiercp_v1x import comparison_training as comparison
from hiercp_v1x import u_bridge_training as engine


ROOT = Path(__file__).resolve().parents[1]


def sealed_sources():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in comparison_experiment.FILES}


class ComparisonExecutionTests(unittest.TestCase):
    def test_source_identity_policy_and_original_helpers_remain_exact(self):
        from hiercp_v1x import comparison_runtime as runtime
        before_sources = sealed_sources()
        policies = {arm: comparison.arm_policy(arm) for arm in comparison.ARMS}
        before = dict(vars(engine))
        with adapter.comparison_execution() as evidence:
            self.assertIs(engine.run_arm.__code__, runtime.run_arm.__code__)
            self.assertIs(engine.run_arm.__globals__, vars(engine))
            self.assertEqual(inspect.signature(engine.run_arm), inspect.signature(before['run_arm']))
            self.assertEqual(engine.run_arm.__kwdefaults__, before['run_arm'].__kwdefaults__)
            for name in ('pair_objective', 'calibrate_batches', 'digest', 'capture_rng',
                         'restore_rng', 'atomic_save', 'validate_resume_progress', 'FORMAT', 'ARMS'):
                self.assertIs(getattr(engine, name), before[name])
            self.assertEqual(evidence['frozen_engine_sha256'], comparison.ENGINE_SHA256)
            self.assertTrue(evidence['sealed_source_files_and_identity_unchanged'])
            self.assertTrue(evidence['training_loop_implementation_changed'])
            self.assertEqual({arm: comparison.arm_policy(arm) for arm in comparison.ARMS}, policies)
            self.assertEqual(sealed_sources(), before_sources)
        self.assertEqual(set(vars(engine)), set(before))
        for name, value in before.items():
            self.assertIs(vars(engine)[name], value, name)
        self.assertEqual(sealed_sources(), before_sources)

    def test_v19_private_namespace_keeps_all_four_objective_and_checkpoint_bindings(self):
        from hiercp_v1x import comparison_runtime as runtime
        original_calibrate = engine.calibrate_batches
        with adapter.comparison_execution():
            namespaces = []
            for arm in comparison.ARMS:
                policy = comparison.arm_policy(arm)
                functions = comparison._engine_functions(policy)
                run = functions['run_arm']
                namespace = run.__globals__
                namespaces.append(namespace)
                self.assertIs(run.__code__, runtime.run_arm.__code__)
                self.assertIsNot(namespace, vars(engine))
                self.assertEqual(namespace['FORMAT'], comparison.FORMAT)
                self.assertEqual(namespace['ARMS'], comparison.ARMS)
                self.assertIs(functions['calibrate_batches'].__code__, original_calibrate.__code__)
                self.assertIsNot(namespace['atomic_save'], engine.atomic_save)
                self.assertIsNot(namespace['_write_new'], engine._write_new)
                self.assertIsNot(namespace['_append'], engine._append)
                for name, helper in runtime.EXECUTION_HELPERS.items():
                    self.assertIs(namespace[name], helper)
                scores = [torch.linspace(-1., 1., 8), torch.linspace(.5, 2., 8)]
                expected, terms = comparison.objective(scores, torch.tensor(.25), arm=arm)
                actual, bound_terms = namespace['pair_objective'](scores, torch.tensor(.25))
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(bound_terms['ranking'], terms['ranking']))
                config = {'u_bridge_runtime': {'batch_calibration': dict(
                    format=comparison.FORMAT, comparison_policy=policy)}}
                with patch('torch.cuda.is_available', return_value=False):
                    with self.assertRaisesRegex(RuntimeError, 'no CPU training fallback'):
                        comparison.run_arm(None, None, config, arm=arm, output='UNIT_unused',
                            physical_batch=2, workers=2, epochs=40, identity={}, budget={})
            self.assertEqual(len({id(namespace) for namespace in namespaces}), 4)

    def test_failure_restores_existing_and_absent_names_exactly(self):
        before = dict(vars(engine))
        with self.assertRaisesRegex(RuntimeError, 'UNIT injected failure'):
            with adapter.comparison_execution():
                raise RuntimeError('UNIT injected failure')
        self.assertEqual(set(vars(engine)), set(before))
        for name, value in before.items():
            self.assertIs(vars(engine)[name], value, name)

    def test_nested_context_restores_outer_then_frozen_function(self):
        original = engine.run_arm
        with adapter.comparison_execution():
            outer = engine.run_arm
            with adapter.comparison_execution():
                self.assertIsNot(engine.run_arm, outer)
            self.assertIs(engine.run_arm, outer)
        self.assertIs(engine.run_arm, original)

    def test_changed_source_rejected_without_namespace_mutation(self):
        before = dict(vars(engine))
        with patch.object(adapter, '_sha', return_value='UNIT changed source'):
            with self.assertRaisesRegex(ValueError, 'cannot migrate'):
                with adapter.comparison_execution():
                    self.fail('Changed source must be rejected before activation')
        self.assertEqual(set(vars(engine)), set(before))
        for name, value in before.items():
            self.assertIs(vars(engine)[name], value, name)

    def test_unreviewed_helper_and_signature_changes_are_rejected(self):
        from hiercp_v1x import comparison_runtime as runtime
        original = engine.run_arm
        with patch.object(runtime, 'EXECUTION_HELPERS', {'pair_objective': engine.pair_objective}):
            with self.assertRaisesRegex(ValueError, 'helper inventory'):
                with adapter.comparison_execution():
                    self.fail('Objective substitution cannot enter helper inventory')
        with patch.object(runtime, 'run_arm', lambda net: None):
            with self.assertRaisesRegex(ValueError, 'signature'):
                with adapter.comparison_execution():
                    self.fail('Signature mutation must be rejected before activation')
        self.assertIs(engine.run_arm, original)

    def test_closure_dependencies_are_rejected(self):
        dependency = 'UNIT closure'
        def replacement(*args, **kwargs):
            return dependency
        with self.assertRaisesRegex(ValueError, 'closure dependencies'):
            adapter._bound_run_arm(engine.run_arm, replacement, vars(engine))


if __name__ == '__main__':
    unittest.main()
