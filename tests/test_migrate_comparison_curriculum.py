"""CPU UNIT migration fixtures; no server, GPU, or real job is launched."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import copy
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import arm_process
from tools import migrate_comparison_curriculum as migration
from tools import run_comparison_arm as preserved


class MigrationUnitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix='UNIT_curriculum_migration_')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(migration, '_matching_processes', return_value=[]))
        self.stack.enter_context(redirect_stdout(StringIO()))

    def fixture(self, arm='native'):
        root = self.base / ('UNIT_' + arm)
        root.mkdir()
        (root / 'data').mkdir()
        (root / arm).mkdir()
        (root / 'execution_overrides').mkdir()
        (root / 'invocations').mkdir()
        self.write(root/'experiment.json', {'UNIT_only': True})
        self.write(root/'calibration.json', {'UNIT_only': True, 'physical_batch': 4})
        self.write(root/'calibration_native.json', {'UNIT_only': True})
        self.write(root/'continuation.json', {'UNIT_only': True})
        self.write(root/arm/'training_identity.json', {'UNIT_only': True})
        self.write(root/arm/'validation_epoch_008.json', {'UNIT_only': True, 'rows': []})
        self.write(root/'execution_overrides/cache_UNIT.json', {'UNIT_only': True})
        self.write(root/'invocations/UNIT.json', {'UNIT_only': True})
        # Synthetic byte payloads only test copying, never torch.load/training.
        for filename in ('initial.pt', f'{arm}/checkpoint_latest.pt', f'{arm}/checkpoint_best.pt'):
            (root / filename).write_bytes(b'UNIT_NOT_A_MODEL_CHECKPOINT:' + filename.encode() * 100)
        request = dict(arm=arm, gpu=4, experiment=root, source_experiment=self.base/'UNIT_original',
            inventory=self.base/'UNIT_inventory.json', data_root=root/'data',
            checkpoint=root/arm/'checkpoint_latest.pt', cache_sources=[root/'data'], requires_clone=False)
        arguments = SimpleNamespace(arm=arm, gpu=4, experiments_dir=self.base, experiment=root,
            action='inspect', curriculum_policy=None, owned_child=False, backup=None,
            debug_fixture=None, debug_config=None, debug_source=None, debug_bank=None)
        return arguments, request

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf8')

    def complete(self, arm='native'):
        arguments, request = self.fixture(arm)
        result = migration.backup(arguments, request)
        arguments.action = 'resume'
        arguments.curriculum_policy = migration.CURRICULUM_POLICY
        arguments.backup = Path(result['backup'])
        return arguments, request, result

    def test_default_is_inspect_and_resume_requires_explicit_policy_and_backup(self):
        basic = ['--arm', 'native', '--gpu', '4']
        arguments = migration.parse(basic)
        self.assertEqual(arguments.action, 'inspect')
        self.assertIsNone(arguments.curriculum_policy)
        for extra in (['--action', 'resume'], ['--action', 'resume', '--backup', 'UNIT'],
                      ['--action', 'inspect', '--backup', 'UNIT'], ['--request-pause']):
            with self.subTest(extra=extra), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                migration.parse([*basic, *extra])

    def test_default_inspection_never_modifies_files_or_launches_a_process(self):
        arguments, request = self.fixture()
        before = {p: p.read_bytes() for p in request['experiment'].rglob('*') if p.is_file()}
        with patch.object(preserved, 'resolve_request', return_value=request), \
                patch.object(preserved, 'run') as launch, patch.object(migration, 'backup') as backup:
            result = migration.run(arguments)
        self.assertTrue(result['quiescent'])
        launch.assert_not_called()
        backup.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in request['experiment'].rglob('*') if p.is_file()})

    def test_full_independent_backup_for_each_arm_and_exact_resume_command(self):
        for arm in preserved.ARMS:
            with self.subTest(arm=arm):
                arguments, request, result = self.complete(arm)
                receipt = migration.verify_backup(arguments, request)
                self.assertTrue(receipt['full_independent_byte_copies'])
                for name, record in receipt['files'].items():
                    source, saved = request['experiment']/name, arguments.backup/'files'/name
                    self.assertEqual(source.read_bytes(), saved.read_bytes())
                    self.assertFalse(os.path.samefile(source, saved))
                    self.assertEqual(migration._record(source), record)
                for required in ('initial.pt', 'calibration.json', f'{arm}/checkpoint_latest.pt',
                                 f'{arm}/checkpoint_best.pt', f'{arm}/training_identity.json'):
                    self.assertIn(required, receipt['files'])
                argv = result['resume_argv']
                parsed = migration.parse(argv[4:])
                self.assertEqual(parsed.action, 'resume')
                self.assertEqual(parsed.curriculum_policy, migration.CURRICULUM_POLICY)
                self.assertEqual(parsed.experiment, request['experiment'])
                self.assertEqual(parsed.cache_sources, request['cache_sources'])
                self.assertEqual(parsed.backup, arguments.backup)
                second = migration.backup(arguments, request)
                self.assertNotEqual(second['backup'], result['backup'])

    def test_active_owner_process_or_any_lock_prevents_copy(self):
        arguments, request = self.fixture()
        with patch.object(preserved, '_active_owner', return_value={'pid': 9999}):
            with self.assertRaisesRegex(RuntimeError, 'already be stopped'):
                migration.backup(arguments, request)
        with patch.object(migration, '_matching_processes', return_value=[{'pid': 9999}]):
            with self.assertRaises(RuntimeError):
                migration.backup(arguments, request)
        for name in ('.setup.lock', '.native.lock', 'data/.data.lock'):
            path = request['experiment']/name
            path.write_text('UNIT stale lock is intentionally preserved')
            with self.subTest(lock=name), self.assertRaises(RuntimeError):
                migration.backup(arguments, request)
            self.assertTrue(path.is_file())
            path.unlink()  # This is the test's own temporary fixture only.
        self.assertFalse((request['experiment']/migration.BACKUPS).exists())

    def test_missing_best_and_uncloned_arm_are_not_partially_backed_up(self):
        arguments, request = self.fixture()
        request['requires_clone'] = True
        with self.assertRaisesRegex(ValueError, 'no clone'):
            migration.backup(arguments, request)
        request['requires_clone'] = False
        (request['experiment']/'native/checkpoint_best.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'regular preserved'):
            migration.backup(arguments, request)
        self.assertFalse((request['experiment']/migration.BACKUPS).exists())

    def test_mutation_during_copy_leaves_incomplete_backup_and_no_complete_receipt(self):
        arguments, request = self.fixture()
        original = migration._copy_new
        changed = False
        def copying(source, target, expected):
            nonlocal changed
            original(source, target, expected)
            if source == request['checkpoint'] and not changed:
                changed = True
                source.write_bytes(b'UNIT newer checkpoint published during copy')
        with patch.object(migration, '_copy_new', side_effect=copying):
            with self.assertRaisesRegex(RuntimeError, 'changed during backup'):
                migration.backup(arguments, request)
        folders = list((request['experiment']/migration.BACKUPS).iterdir())
        self.assertEqual(len(folders), 1)
        self.assertTrue((folders[0]/'backup_in_progress.json').is_file())
        self.assertFalse((folders[0]/'backup_complete.json').exists())

    def test_changed_original_or_backup_bytes_reject_resume(self):
        arguments, request, _ = self.complete()
        for path in (request['checkpoint'], arguments.backup/'files/native/checkpoint_best.pt'):
            before = path.read_bytes()
            path.write_bytes(before + b'UNIT changed')
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'byte identity differs'):
                migration.verify_backup(arguments, request)
            path.write_bytes(before)
        receipt_path = arguments.backup/'backup_complete.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['arm'] = 'selected'
        self.write(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, 'digest changed'):
            migration.verify_backup(arguments, request)

    def test_pause_flag_is_preserved_and_resume_is_rejected(self):
        arguments, request, _ = self.complete()
        pause = request['experiment']/'native/STOP_AFTER_BATCH'
        pause.write_text('UNIT explicit pause')
        with self.assertRaisesRegex(RuntimeError, 'clear it explicitly'):
            migration.verify_backup(arguments, request)
        self.assertEqual(pause.read_text(), 'UNIT explicit pause')

    def test_resume_keeps_existing_supervisor_and_owned_child_policy_without_pbs(self):
        arguments, request, _ = self.complete()
        command = [sys.executable, '-B', '-u', str(preserved.ROOT/'tools/run_comparison_arm.py'),
                   '--owned-child', '--gpu', '4', '--arm', 'native', '--experiment', str(request['experiment'])]
        def original_parent(args):
            return arm_process.run_owned(command, checkpoint=request['checkpoint'], grace_seconds=10, cwd=preserved.ROOT)
        with patch.object(preserved, 'resolve_request', return_value=request), \
                patch.object(preserved, 'run', side_effect=original_parent), \
                patch.object(arm_process, 'run_owned', return_value={'UNIT_no_process_started': True}) as owned, \
                patch('tools.run_allocated_comparison_arm.check_current_allocation') as allocation:
            migration.run(arguments)
        allocation.assert_not_called()
        spawned = owned.call_args.args[0]
        self.assertEqual(spawned[3], str(Path(migration.__file__).resolve()))
        child = migration.parse(spawned[4:])
        self.assertTrue(child.owned_child)
        self.assertEqual(child.action, 'resume')
        self.assertEqual(child.backup, arguments.backup)
        self.assertEqual(child.curriculum_policy, migration.CURRICULUM_POLICY)
        self.assertEqual(owned.call_args.kwargs['checkpoint'], request['checkpoint'])

    def test_inspection_import_does_not_import_torch(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            "import sys; import tools.migrate_comparison_curriculum; assert 'torch' not in sys.modules"],
            cwd=migration.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
