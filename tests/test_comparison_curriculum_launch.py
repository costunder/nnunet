"""UNIT launcher/policy boundaries; no real process, CUDA, or training."""
from contextlib import redirect_stderr
import copy
import importlib
from io import StringIO
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import resume_comparison_cached as cached
from tools import run_allocated_comparison_arm as allocated
from tools import run_comparison_arm as frozen
from hiercp_v1x import arm_process, comparison_training, u_bridge_training
from hiercp_v1x.comparison_curriculum import policy


POLICY = cached.CURRICULUM_POLICY


class CurriculumLauncherTests(unittest.TestCase):
    def test_cli_opt_in_is_explicit_and_default_is_legacy(self):
        common = ['--gpu', '4', '--arm', 'native']
        cache_args = [*common, '--experiment', 'UNIT', '--cache-sources', 'UNIT_CACHE',
                      '--inventory', 'UNIT_INVENTORY']
        for parser, args in ((cached.parse, cache_args), (allocated.parse, common)):
            self.assertIsNone(parser(args).curriculum_policy)
            self.assertEqual(parser([*args, '--curriculum-policy', POLICY]).curriculum_policy, POLICY)
            with self.subTest(parser=parser), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parser([*args, '--curriculum-policy', 'unknown'])
        with patch.dict('os.environ', {'CP_CURRICULUM_POLICY': POLICY}):
            self.assertIsNone(cached.requested_curriculum_policy(SimpleNamespace()))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            frozen.parse([*common, '--curriculum-policy', POLICY])

    def test_owned_entrypoint_must_match_exact_frozen_child(self):
        args = ['--gpu', '4', '--arm', 'native', '--curriculum-policy', POLICY]
        expected = str(frozen.ROOT / 'tools/run_comparison_arm.py')
        accepted = allocated.parse([*args, '--owned-child', '--preserved-entrypoint', expected])
        self.assertTrue(accepted.owned_child)
        for tail in (['--owned-child'], ['--preserved-entrypoint', expected],
                     ['--owned-child', '--preserved-entrypoint', 'UNIT_wrong.py']):
            with self.subTest(tail=tail), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                allocated.parse([*args, *tail])

    def test_both_families_receive_policy_without_mutating_sealed_config(self):
        for family, module in (('u_bridge', u_bridge_training), ('comparison', comparison_training)):
            config = {'u_bridge_runtime': {'resume_checkpoint': 'UNIT.pt'}, 'model': {'hidden': 128}}
            before = copy.deepcopy(config)
            original = Mock(return_value='UNIT_result')
            with patch.object(module, 'run_arm', original):
                with cached._curriculum_execution(family, POLICY):
                    self.assertEqual(module.run_arm('net', 'provider', config, arm='native'), 'UNIT_result')
                    received = original.call_args.args[2]
                    self.assertEqual(received['u_bridge_runtime']['curriculum_policy'], POLICY)
                    self.assertEqual(received['model'], config['model'])
                    self.assertIsNot(received, config)
                self.assertIs(module.run_arm, original)
            self.assertEqual(config, before)

    def test_v19_private_engine_keeps_closure_free_binding(self):
        original = u_bridge_training.run_arm
        with cached._curriculum_execution('comparison', POLICY):
            self.assertIs(u_bridge_training.run_arm, original)
            self.assertIsNone(u_bridge_training.run_arm.__closure__)
            functions = comparison_training._engine_functions(comparison_training.arm_policy('native_listwise'))
            self.assertIsNotNone(functions)

    def test_conflicting_config_fails_and_all_bindings_restore_on_error(self):
        original = comparison_training.run_arm
        normalize = comparison_training._normalize_report
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            with cached._curriculum_execution('comparison', POLICY):
                comparison_training.run_arm(None, None, {'u_bridge_runtime': {'curriculum_policy': 'other'}})
        self.assertIs(comparison_training.run_arm, original)
        self.assertIs(comparison_training._normalize_report, normalize)

    def test_v19_report_preserves_baseline_identity_and_discloses_effective_schedule(self):
        for arm in ('native_fixed', 'native_listwise'):
            base = comparison_training.arm_policy(arm)
            active = policy(arm)
            report = dict(curriculum_policy=active, trained_comparisons=dict(counts_by_source={'0': 14}))
            before = copy.deepcopy(report)
            with cached._curriculum_execution('comparison', POLICY):
                result = comparison_training._normalize_report(report, base)
            self.assertEqual(report, before)
            self.assertEqual(result['comparison_policy'], base)
            self.assertEqual(result['baseline_comparison_policy'], base)
            self.assertIn('original experiment identity', result['comparison_policy_role'])
            schedule = active['sampling'] if active['adaptive'] else base['candidate_schedule']
            self.assertEqual(result['trained_comparisons']['candidate_schedule'], schedule)
        legacy = {'trained_comparisons': {'counts_by_source': {'0': 7}}}
        expected = comparison_training._normalize_report(legacy, base)
        with cached._curriculum_execution('comparison', POLICY):
            self.assertEqual(comparison_training._normalize_report(legacy, base), expected)

    def test_parent_dispatch_preserves_all_owned_arguments_and_checkpoint(self):
        entry = str(frozen.ROOT / 'tools/run_comparison_arm.py')
        command = ['UNIT_python', '-B', '-u', entry, '--owned-child', '--gpu', '4',
                   '--arm', 'native', '--experiment', 'UNIT_EXPERIMENT']
        kwargs = dict(checkpoint=Path('UNIT.pt'), grace_seconds=10, cwd=frozen.ROOT)
        original = Mock(return_value='UNIT_owned')
        with patch.object(arm_process, 'run_owned', original):
            with allocated._curriculum_dispatch(SimpleNamespace(owned_child=False, curriculum_policy=POLICY)):
                self.assertEqual(arm_process.run_owned(command, **kwargs), 'UNIT_owned')
            self.assertIs(arm_process.run_owned, original)
        actual = original.call_args.args[0]
        self.assertEqual(command[3], entry)
        self.assertEqual(actual[:3], command[:3])
        self.assertEqual(actual[4:-4], command[4:])
        self.assertEqual(actual[-4:], ['--curriculum-policy', POLICY, '--preserved-entrypoint', entry])
        self.assertEqual(original.call_args.kwargs, kwargs)
        parsed = allocated.parse(actual[4:])
        self.assertEqual((parsed.arm, parsed.gpu, parsed.curriculum_policy), ('native', 4, POLICY))

    def test_child_dispatch_passes_copy_and_restores_after_failure(self):
        supplied = SimpleNamespace(arm='native', gpu=4)
        original = Mock(side_effect=RuntimeError('UNIT downstream failure'))
        with patch.object(cached, 'run', original):
            with self.assertRaisesRegex(RuntimeError, 'downstream'):
                with allocated._curriculum_dispatch(SimpleNamespace(owned_child=True, curriculum_policy=POLICY)):
                    cached.run(supplied)
            self.assertIs(cached.run, original)
        self.assertFalse(hasattr(supplied, 'curriculum_policy'))
        actual = original.call_args.args[0]
        self.assertIsNot(actual, supplied)
        self.assertEqual(actual.curriculum_policy, POLICY)

    def test_default_dispatch_leaves_originals_unchanged(self):
        original, child = arm_process.run_owned, cached.run
        with allocated._curriculum_dispatch(SimpleNamespace(owned_child=False)):
            self.assertIs(arm_process.run_owned, original)
            self.assertIs(cached.run, child)

    def test_curriculum_child_is_recognized_by_unchanged_owner_verifier(self):
        # This tests only metadata. No Process method signals or launches jobs.
        with TemporaryDirectory(prefix='UNIT_curriculum_owner_') as name:
            root = Path(name)
            (root / '.pipeline.lock').write_text('{}')
            command = ['python', '-B', '-u', str(Path(allocated.__file__).resolve()),
                       '--owned-child', '--arm', 'native', '--gpu', '4', '--experiment', str(root),
                       '--curriculum-policy', POLICY, '--preserved-entrypoint',
                       str(frozen.ROOT / 'tools/run_comparison_arm.py')]
            process = Mock()
            process.cmdline.return_value = command
            process.exe.return_value = 'python.exe'
            process.create_time.return_value = 123.0
            process.username.return_value = 'UNIT_user'
            import os
            if hasattr(os, 'getuid'):
                process.uids.return_value = SimpleNamespace(real=os.getuid())
            owner = dict(pid=999991, host=socket.gethostname())
            request = dict(experiment=root, arm='native', checkpoint=root/'native/checkpoint_latest.pt')
            with patch.object(frozen, '_read', return_value=owner), patch('psutil.Process', return_value=process):
                result = frozen._active_owner(request)
            self.assertEqual(result['status'], 'ALREADY_RUNNING')
            self.assertEqual(result['command'], command)
            process.terminate.assert_not_called()
            process.kill.assert_not_called()

    def test_cached_route_records_policy_for_all_four_arms_without_resealing(self):
        from tests.test_comparison_cached_resume import unit_fixture
        extra_helpers = ('hiercp_v1x/comparison_curriculum.py',
                         'hiercp_v1x/comparison_curriculum_data.py',
                         'hiercp_v1x/comparison_stage_validation.py')
        for arm in ('selected', 'native', 'native_fixed', 'native_listwise'):
            family = 'u_bridge' if arm in ('selected', 'native') else 'comparison'
            with self.subTest(arm=arm), TemporaryDirectory(prefix='UNIT_curriculum_route_') as name:
                root, repository, _, _, supplied = unit_fixture(Path(name), family=family)
                supplied.arm = arm
                supplied.curriculum_policy = POLICY
                supplied.cache_sources[0].mkdir()
                for helper in extra_helpers:
                    (repository / helper).write_text('UNIT provenance fixture: ' + helper, encoding='utf8')
                before = (root / 'experiment.json').read_bytes()
                controller = importlib.import_module('tools.run_v18_u_bridge' if family == 'u_bridge'
                                                    else 'tools.run_v19_comparison')
                training = importlib.import_module('hiercp_v1x.' + family + '_training')
                original = training.run_arm
                def unit_controller(args):
                    self.assertEqual(args.arm, arm)
                    self.assertIsNot(training.run_arm, original)
                    # The policy wrapper encloses the already-bound execution
                    # runtime; preserved controllers need no new CLI/config file.
                    self.assertTrue(hasattr(training.run_arm, '__wrapped__'))
                    return 'UNIT_no_neural_execution'
                with patch.object(cached, 'ROOT', repository), patch('tools.local_cnn_device.select'), \
                        patch.object(controller, 'run', side_effect=unit_controller):
                    self.assertEqual(cached.run(supplied), 'UNIT_no_neural_execution')
                self.assertIs(training.run_arm, original)
                self.assertEqual((root / 'experiment.json').read_bytes(), before)
                paths = list((root / 'execution_overrides').glob('cache_*.json'))
                self.assertEqual(len(paths), 1)
                receipt = json.loads(paths[0].read_text())
                self.assertEqual(receipt['curriculum_policy'], POLICY)
                self.assertTrue(receipt['evaluation_policy_changed'])
                self.assertEqual(receipt['training_candidate_schedule_changed'], arm in ('native', 'native_listwise'))
                self.assertTrue(set(extra_helpers) <= set(receipt['helpers']))


if __name__ == '__main__':
    unittest.main()
