"""UNIT filesystem fault injection only; no real quota, CT, model or CUDA run."""
from contextlib import redirect_stderr, redirect_stdout
import errno
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import comparison_storage as storage
from hiercp_v1x.comparison_progress import PhaseProgress
from tools import run_comparison_arm as launch


REPOSITORY = Path(__file__).resolve().parents[1]
EDQUOT = getattr(errno, 'EDQUOT', 122)


class StorageCase(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(prefix='UNIT_comparison_storage_', dir=REPOSITORY)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.saved = self.root / 'checkpoint_latest.pt'
        self.saved.write_bytes(b'UNIT opaque saved checkpoint; not neural data')
        self.before = self.saved.read_bytes()

    def assert_saved(self):
        self.assertEqual(self.saved.read_bytes(), self.before)


class StorageUNIT(StorageCase):
    def test_real_small_write_probe_cleans_only_own_files_and_limits_claim(self):
        existing = self.root / '.storage_probe_existing'
        existing.mkdir()
        (existing / 'write.tmp').write_bytes(b'UNIT existing unrelated probe')
        receipt = storage.probe_output_storage(self.root)
        self.assertEqual(receipt['bytes_written'], 4096)
        self.assertIn('not future', receipt['scope'])
        self.assertEqual(set(self.root.iterdir()), {self.saved, existing})
        self.assertEqual((existing / 'write.tmp').read_bytes(), b'UNIT existing unrelated probe')
        self.assert_saved()

    def test_fsync_quota_and_space_failures_keep_errno_path_and_saved_checkpoint(self):
        for code, label in ((EDQUOT, 'EDQUOT'), (errno.ENOSPC, 'ENOSPC')):
            with self.subTest(code=code):
                original = OSError(code, 'UNIT injected storage allocation failure')
                with patch.object(storage.os, 'fsync', side_effect=original):
                    with self.assertRaises(OSError) as caught:
                        storage.probe_output_storage(self.root)
                self.assertEqual(caught.exception.errno, code)
                self.assertIs(caught.exception.__cause__, original)
                self.assertIn(label, str(caught.exception))
                self.assertIn('fsync storage probe', str(caught.exception))
                self.assertIn('write.tmp', caught.exception.filename)
                self.assertEqual(list(self.root.iterdir()), [self.saved])
                self.assert_saved()

    def test_existing_probe_name_collision_is_never_removed_or_overwritten(self):
        existing = self.root / '.storage_probe_UNIT_collision'
        existing.mkdir()
        saved = existing / 'write.tmp'
        saved.write_bytes(b'UNIT do not replace')
        with patch.object(storage.uuid, 'uuid4', return_value=SimpleNamespace(hex='UNIT_collision')):
            with self.assertRaises(OSError) as caught:
                storage.probe_output_storage(self.root)
        self.assertEqual(caught.exception.errno, errno.EEXIST)
        self.assertEqual(saved.read_bytes(), b'UNIT do not replace')
        self.assert_saved()

    def test_probe_create_quota_reports_filename_without_touching_existing_files(self):
        quota = OSError(EDQUOT, 'UNIT exclusive create quota')
        with patch.object(Path, 'open', side_effect=quota):
            with self.assertRaises(OSError) as caught:
                storage.probe_output_storage(self.root)
        self.assertIn('exclusively create', str(caught.exception))
        self.assertIn('write.tmp', caught.exception.filename)
        self.assertEqual(list(self.root.iterdir()), [self.saved])
        self.assert_saved()

    def test_cleanup_failure_is_reported_without_masking_primary_quota(self):
        original = OSError(EDQUOT, 'UNIT quota')
        secondary = OSError(errno.EACCES, 'UNIT cleanup denied')
        stream = io.StringIO()
        with patch.object(storage.os, 'fsync', side_effect=original), \
                patch.object(Path, 'unlink', side_effect=secondary), redirect_stderr(stream):
            with self.assertRaises(OSError) as caught:
                storage.probe_output_storage(self.root)
        self.assertEqual(caught.exception.errno, EDQUOT)
        self.assertIs(caught.exception.__cause__, original)
        self.assertIs(caught.exception.comparison_secondary_errors[0][1], secondary)
        self.assertIn('cleanup denied', stream.getvalue())
        self.assert_saved()

    def test_parent_quota_failure_precedes_spawn_and_child_quota_precedes_heavy_imports(self):
        from hiercp_v1x import arm_process
        from tools import current_gpu
        request = dict(experiment=self.root, checkpoint=self.root/'selected/checkpoint_latest.pt',
                       requires_clone=False)
        args = SimpleNamespace(owned_child=False)
        quota = OSError(EDQUOT, 'UNIT quota', str(self.root))
        with patch.object(launch, 'resolve_request', return_value=request), \
                patch.object(launch, '_active_owner', return_value=None), \
                patch.object(storage, 'probe_output_storage', side_effect=quota), \
                patch.object(arm_process, 'run_owned') as spawn:
            with self.assertRaises(OSError) as caught:
                launch.run(args)
            self.assertIs(caught.exception, quota)
            spawn.assert_not_called()
        with patch.object(storage, 'probe_output_storage', side_effect=quota), \
                patch.object(current_gpu, 'current_device_selection') as gpu:
            with self.assertRaises(OSError):
                launch._child(args, request)
            gpu.assert_not_called()
        self.assert_saved()

    def test_arm_output_is_checked_separately_and_active_parent_does_not_probe(self):
        arm = self.root / 'selected'
        arm.mkdir()
        request = dict(experiment=self.root, checkpoint=arm/'checkpoint_latest.pt', requires_clone=False)
        args = SimpleNamespace(owned_child=False)
        quota = OSError(EDQUOT, 'UNIT arm output quota', str(arm))
        with patch.object(storage, 'probe_output_storage', side_effect=[{}, quota]) as probe, \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                launch._probe_storage(request)
        self.assertEqual([call.args[0] for call in probe.call_args_list], [self.root, arm])
        active = dict(status='ALREADY_RUNNING')
        with patch.object(launch, 'resolve_request', return_value=request), \
                patch.object(launch, '_active_owner', return_value=active), \
                patch.object(storage, 'probe_output_storage') as probe, redirect_stdout(io.StringIO()):
            self.assertIs(launch.run(args), active)
            probe.assert_not_called()
        self.assert_saved()


class ProgressStorageUNIT(StorageCase):
    def progress(self, stream):
        return PhaseProgress(root=self.root, arm='selected', phase='train7', epoch=1, epochs=40,
                             total=151, initial=0, physical_batch=2, stream=stream, interval=60.)

    def test_secondary_progress_quota_preserves_primary_object_cause_and_checkpoint(self):
        stream = io.StringIO()
        original_cause = ValueError('UNIT original cause')
        primary = OSError(EDQUOT, 'UNIT primary audit quota', str(self.root/'gpu_execution.jsonl'))
        secondary = OSError(EDQUOT, 'UNIT progress quota')
        progress = self.progress(stream)
        original_open = Path.open
        fail = False
        def open_path(path, *args, **kwargs):
            if fail and args and args[0] == 'x':
                raise secondary
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', autospec=True, side_effect=open_path):
            with self.assertRaises(OSError) as caught:
                with progress:
                    before = (self.root/'progress.json').read_bytes()
                    bar = Mock()
                    progress.bar = bar
                    fail = True
                    raise primary from original_cause
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, original_cause)
        self.assertIn('Secondary progress close', stream.getvalue())
        self.assertIn('EDQUOT', stream.getvalue())
        self.assertIn('.progress.json.', stream.getvalue())
        self.assertTrue(progress.closed)
        self.assertTrue(progress.stop.is_set())
        self.assertFalse(progress.thread.is_alive())
        bar.close.assert_called_once()
        self.assertEqual((self.root/'progress.json').read_bytes(), before)
        self.assert_saved()

    def test_failed_replace_removes_only_new_temporary_and_preserves_previous_progress(self):
        previous = self.root/'progress.json'
        previous.write_bytes(b'UNIT prior progress receipt')
        old_temporary = self.root/'.progress.json.previous.tmp'
        old_temporary.write_bytes(b'UNIT old temporary stays')
        progress = self.progress(io.StringIO())
        quota = OSError(EDQUOT, 'UNIT replace quota')
        with patch('hiercp_v1x.comparison_progress.os.replace', side_effect=quota):
            with self.assertRaises(OSError):
                progress.__enter__()
        self.assertEqual(previous.read_bytes(), b'UNIT prior progress receipt')
        self.assertEqual(old_temporary.read_bytes(), b'UNIT old temporary stays')
        self.assertEqual(set(self.root.iterdir()), {self.saved, previous, old_temporary})
        self.assert_saved()

    def test_initial_publish_quota_closes_tty_bar_and_starts_no_heartbeat(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        progress = self.progress(Terminal())
        bar = Mock()
        with patch('tqdm.tqdm', return_value=bar), \
                patch.object(progress, '_publish', side_effect=OSError(EDQUOT, 'UNIT initial quota')):
            with self.assertRaises(OSError):
                progress.__enter__()
        self.assertTrue(progress.closed)
        self.assertTrue(progress.stop.is_set())
        self.assertIsNone(progress.thread)
        bar.close.assert_called_once()
        self.assert_saved()


if __name__ == '__main__':
    unittest.main()
