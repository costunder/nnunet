"""CPU UNIT: explicit checkpoint policy adoption and scoped installation."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import comparison_geometry_execution as execution


class GeometryExecutionTests(unittest.TestCase):
    def test_legacy_resume_explicitly_adopts_without_mutating_policy(self):
        policy = {'policy_sha256': 'unit', 'format': 'unit'}
        receipt = execution.validate_resume_policy(None, policy)
        self.assertEqual(receipt['transition'], 'adopt_from_strict_nonempty_checkpoint')
        self.assertTrue(receipt['model_optimizer_scheduler_rng_and_cursor_preserved'])
        receipt['policy']['format'] = 'changed'
        self.assertEqual(policy['format'], 'unit')

    def test_same_policy_resumes_but_missing_or_changed_active_policy_rejects(self):
        policy = {'policy_sha256': 'unit'}
        self.assertEqual(execution.validate_resume_policy(policy, policy)['transition'], 'same_policy_resume')
        for active in (None, {'policy_sha256': 'other'}):
            with self.assertRaisesRegex(ValueError, 'policy differs'):
                execution.validate_resume_policy(policy, active)
        self.assertIsNone(execution.validate_resume_policy(None, None))

    def test_controller_scope_installs_before_adapter_and_is_restored_on_error(self):
        from hiercp_v1x import bounded_scope, comparison_empty_context
        scope = {'contract_sha256': 'scope', 'margin_mm': 10.}
        calls = []
        def original_install(margin_mm, expected_snapshot_root):
            calls.append('scope')
            self.assertEqual(margin_mm, 10)
            return scope
        @contextmanager
        def activated(runtime):
            calls.append('adapter')
            self.assertIs(runtime['scope'], scope)
            self.assertEqual(runtime['snapshot'], Path.cwd())
            try:
                yield
            finally:
                calls.append('restore')
        with patch.object(bounded_scope, 'install', original_install), \
                patch.object(bounded_scope, '_ACTIVE', None), \
                patch.object(comparison_empty_context, 'activated', activated), \
                patch.object(comparison_empty_context, 'identity', return_value={'policy_sha256': 'unit'}), \
                patch.object(execution.importlib, 'import_module', side_effect=lambda name: SimpleNamespace(name=name)):
            with self.assertRaisesRegex(RuntimeError, 'UNIT failure'):
                with execution.comparison_geometry_execution():
                    self.assertIsNone(execution.current_policy())
                    self.assertIs(bounded_scope.install(10, Path.cwd()), scope)
                    self.assertEqual(execution.current_policy(), {'policy_sha256': 'unit'})
                    raise RuntimeError('UNIT failure')
            self.assertIs(bounded_scope.install, original_install)
        self.assertEqual(calls, ['scope', 'adapter', 'restore'])
        self.assertIsNone(execution.current_policy())


if __name__ == '__main__':
    unittest.main()
