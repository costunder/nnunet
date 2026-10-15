"""CPU DEBUG tests: coordination cost must not serialize unchanged CUDA steps.

Synthetic actor identities only exercise guards; these are not model results.
Filesystem ownership uses the actual test UID in a scoped DEBUG adapter.
"""
import copy
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import basic_cp_512_claims as claims
from hiercp_v1x import basic_cp_512_resources as resources
from hiercp_v1x import basic_cp_ablation_resources as common


def actor(pid, command, uid=41833):
    return dict(pid=pid, create_time=1000.0 + pid, UID=uid, command=list(command))


class AtomicReadTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='DEBUG_atomic_ledger_',
            dir=Path(__file__).resolve().parent)
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.uid = self.root.stat().st_uid
        # A test-only real filesystem ownership adapter, never production.
        owner = patch.object(claims, 'UID', self.uid)
        owner.start(); self.addCleanup(owner.stop)
        self.control = actor(100, ['DEBUG_controller'], self.uid)
        self.binding = dict(registry_path=str(self.root/'registry.json'), registry_sha256='a'*64,
            request_sha256={str(gpu):'b'*64 for gpu in range(7)}, host_lifetime_sha256='c'*64,
            expires_at_unix=claims.time.time()+3600, owned_tmpfs_root=str(self.root),
            source_initially_pending=False)
        native = dict(claims.FLAGS, gpu=6, RAM512_claim_ledger=str(self.root/'ledger.json'),
            owned_tmpfs_root=str(self.root), controller_identity=self.control,
            code_sha256={claims.FILE:'d'*64})
        self.ledger = claims.Ledger(native, self.binding, Mock())
        self.state = dict(format=claims.FORMAT, binding=self.binding, claims={},
            storage_reservations={}, source_pending=False, source_released=True,
            priority_complete=False, revision=1)
        self.ledger._write(self.state)

    def test_coherent_snapshot_matches_locked_read_and_does_not_create_lock(self):
        self.assertFalse(self.ledger.lock_path.exists())
        self.assertEqual(self.ledger.snapshot_atomic_readonly(), self.state)
        self.assertFalse(self.ledger.lock_path.exists())
        self.assertEqual(self.ledger.snapshot(), self.state)

    def test_read_completes_while_writer_mutex_is_held(self):
        acquired = threading.Event(); release = threading.Event()
        def hold():
            with self.ledger._locked():
                acquired.set(); release.wait(5)
        worker = threading.Thread(target=hold)
        worker.start(); self.assertTrue(acquired.wait(5))
        try:
            with patch.object(self.ledger, '_locked', side_effect=AssertionError('read took writer mutex')):
                self.assertEqual(self.ledger.snapshot_atomic_readonly(), self.state)
        finally:
            release.set(); worker.join(5)
        self.assertFalse(worker.is_alive())

    def test_atomic_replacement_during_open_is_retried(self):
        real_open = claims.os.open
        next_state = dict(self.state, revision=2)
        replaced = False
        def opening(path, flags, *args, **kwargs):
            nonlocal replaced
            if Path(path) == self.ledger.path and not replaced:
                replaced = True; self.ledger._write(next_state)
            return real_open(path, flags, *args, **kwargs)
        with patch.object(claims.os, 'open', side_effect=opening):
            self.assertEqual(self.ledger.snapshot_atomic_readonly(), next_state)

    def test_atomic_replacement_after_read_is_retried(self):
        real_regular = claims._regular; calls = 0
        def regular(path):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.ledger._write(dict(self.state, revision=2))
            return real_regular(path)
        with patch.object(claims, '_regular', side_effect=regular):
            self.assertEqual(self.ledger.snapshot_atomic_readonly()['revision'], 2)

    def test_revision_regression_and_same_revision_mutation_are_rejected(self):
        for replacement in (dict(self.state, revision=0),
                            dict(self.state, priority_complete=True)):
            with self.subTest(replacement=replacement):
                self.ledger._atomic_read_version = None
                self.ledger._write(self.state); self.ledger.snapshot_atomic_readonly()
                self.ledger._write(replacement)
                with self.assertRaisesRegex(RuntimeError, 'revision'):
                    self.ledger.snapshot_atomic_readonly()

    def test_changed_binding_and_invalid_revision_are_rejected(self):
        for replacement in (dict(self.state, binding=dict(self.binding, registry_sha256='e'*64)),
                            dict(self.state, revision=True)):
            with self.subTest(replacement=replacement):
                self.ledger._write(replacement)
                with self.assertRaisesRegex(RuntimeError, 'binding or phase'):
                    self.ledger.snapshot_atomic_readonly()

    def test_foreign_owned_open_fd_is_rejected(self):
        real_fstat = claims.os.fstat
        def foreign(fd):
            value = real_fstat(fd)
            fields = ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_uid','st_gid','st_mode')
            row = {name:getattr(value, name) for name in fields}; row['st_uid'] = self.uid + 1
            return SimpleNamespace(**row)
        with (patch.object(claims.os, 'getuid', create=True, return_value=self.uid),
              patch.object(claims.os, 'fstat', side_effect=foreign),
              self.assertRaisesRegex(ValueError, 'owned regular')):
            self.ledger.snapshot_atomic_readonly()

    def test_in_place_write_during_read_is_rejected(self):
        real_fstat = claims.os.fstat; calls = 0
        def observe(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.ledger.path.write_text(json.dumps(dict(self.state, revision=2)), encoding='utf8')
            return real_fstat(fd)
        with patch.object(claims.os, 'fstat', side_effect=observe):
            with self.assertRaisesRegex(RuntimeError, 'inode changed'):
                self.ledger.snapshot_atomic_readonly()

    def test_expired_lifetime_and_missing_file_do_not_produce_empty_snapshot(self):
        with patch.object(claims.time, 'time', return_value=self.binding['expires_at_unix']):
            with self.assertRaisesRegex(RuntimeError, 'lifetime'):
                self.ledger.snapshot_atomic_readonly()
        self.ledger.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.ledger.snapshot_atomic_readonly()


class BoundNativeTests(unittest.TestCase):
    def setUp(self):
        self.control = actor(100, ['DEBUG_controller'])
        self.child = actor(200, ['DEBUG_python', '--stage', 'train'])
        self.claim = dict(token='exact-token', kind='native', gpu=5, stage='train',
            command=self.child['command'], request_sha256='a'*64, child_identity=self.child,
            controller_identity=self.control, claimed_at_unix=1000.0, captured=[self.child])
        self.state = dict(revision=5, claims={'exact-token':self.claim})
        self.ledger = Mock(unsafe=True)
        self.ledger.snapshot.side_effect = lambda:copy.deepcopy(self.state)
        self.ledger.snapshot_atomic_readonly.side_effect = lambda:copy.deepcopy(self.state)
        self.lease = object.__new__(resources._RAM512)
        self.lease.self_identity = copy.deepcopy(self.child)
        self.lease.request = dict(gpu=5, controller_identity=copy.deepcopy(self.control))
        self.lease.claim_ledger = lambda:self.ledger
        self.parent = Mock()
        self.own = Mock(); self.own.parent.return_value = self.parent
        exact = patch.object(common, '_exact', return_value=self.own)
        exact.start(); self.addCleanup(exact.stop)
        ident = patch.object(resources, '_identity', return_value=copy.deepcopy(self.control))
        ident.start(); self.addCleanup(ident.stop)

    def test_initial_bound_claim_uses_lock_then_every_step_reads_fresh_without_lock(self):
        self.lease._bind_self()
        self.ledger.snapshot.assert_called_once()
        for i in range(10):
            self.state['revision'] += 1
            self.lease._bind_self()
        self.ledger.snapshot.assert_called_once()
        self.assertEqual(self.ledger.snapshot_atomic_readonly.call_count, 10)

    def test_first_selfbind_retains_exact_parent_mutex_path(self):
        self.claim['child_identity'] = None
        def binding(stage, command, child, parent):
            self.assertEqual((stage, command, child, parent),
                ('train', self.child['command'], self.child, self.control))
            self.claim['child_identity'] = copy.deepcopy(child)
            return copy.deepcopy(self.claim)
        self.ledger.assert_current_child.side_effect = binding
        self.lease._bind_self()
        self.ledger.assert_current_child.assert_called_once()
        self.assertEqual(self.lease._bound_native_claim['token'], 'exact-token')

    def test_removed_or_replaced_token_cannot_continue(self):
        self.lease._bind_self(); self.state['claims'].clear()
        with self.assertRaisesRegex(RuntimeError, 'removed'):
            self.lease._bind_self()

    def test_any_bound_identity_change_and_revision_regression_fail_immediately(self):
        fields = dict(gpu=6, kind='cpu_source', stage='debug_B4', command=['DEBUG_different'],
            child_identity=dict(self.child, create_time=9999),
            controller_identity=dict(self.control, UID=0), request_sha256='f'*64)
        for field, value in fields.items():
            with self.subTest(field=field):
                self.lease._bound_native_claim = None
                original = copy.deepcopy(self.claim)
                self.lease._bind_self(); self.claim[field] = value
                with self.assertRaisesRegex(RuntimeError, 'changed identity'):
                    self.lease._bind_self()
                self.claim.clear(); self.claim.update(original)
        self.state['revision'] = 4
        with self.assertRaisesRegex(RuntimeError, 'regressed'):
            self.lease._bind_self()

    def test_missing_or_duplicate_initial_claim_is_fatal(self):
        self.state['claims']['second'] = copy.deepcopy(self.claim)
        with self.assertRaisesRegex(RuntimeError, 'Exactly one'):
            self.lease._bind_self()

    def test_unadmitted_parent_or_replaced_child_cannot_populate_fast_path(self):
        self.own.parent.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'exact admitted parent'):
            self.lease._bind_self()
        self.assertIsNone(getattr(self.lease, '_bound_native_claim', None))
        self.own.parent.return_value = self.parent
        self.claim['child_identity'] = dict(self.child, create_time=5555)
        with self.assertRaisesRegex(RuntimeError, 'exact live identity'):
            self.lease._bind_self()

    def test_cpu_source_never_uses_native_readonly_fast_path(self):
        self.claim['kind'] = 'cpu_source'
        self.lease._bind_self(); self.lease._bind_self()
        self.assertEqual(self.ledger.snapshot.call_count, 2)
        self.ledger.snapshot_atomic_readonly.assert_not_called()


class BatchedMetadataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='DEBUG_batch_metadata_',
            dir=Path(__file__).resolve().parent)
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.code = self.root/'code'; self.code.mkdir()
        self.paths = [self.code/('DEBUG_'+str(i)+'.py') for i in range(20)]
        for path in self.paths:
            path.write_text('DEBUG immutable metadata\n', encoding='utf8')
        owner = patch.object(common, 'UID', self.root.stat().st_uid)
        owner.start(); self.addCleanup(owner.stop)
        self.lease = object.__new__(common.BasicLease)
        self.lease.request = dict(DEBUG=True)
        self.lease.request_bytes = json.dumps(self.lease.request, sort_keys=True, separators=(',', ':'))
        self.lease._files = {str(path):common._snapshot(path) for path in self.paths}

    def test_shared_ancestors_checked_once_then_revalidated_each_invocation(self):
        real = Path.lstat; counts = {}
        def observed(path, *args, **kwargs):
            counts[path] = counts.get(path, 0) + 1
            return real(path, *args, **kwargs)
        with patch.object(Path, 'lstat', observed):
            self.lease._inputs()
        self.assertEqual(counts[self.code], 2)
        with patch.object(Path, 'lstat', observed):
            self.lease._inputs()
        self.assertEqual(counts[self.code], 4)

    def test_modified_missing_and_replaced_file_fail(self):
        changed = self.paths[0]; changed.write_text('DEBUG changed metadata', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'metadata changed'):
            self.lease._inputs()
        changed.unlink()
        with self.assertRaises(FileNotFoundError):
            self.lease._inputs()

    def test_foreign_file_uid_is_still_rejected(self):
        with patch.object(common, 'UID', self.root.stat().st_uid+1):
            with self.assertRaisesRegex(ValueError, 'Own UID'):
                self.lease._inputs()

    def test_ancestor_replacement_during_batch_is_rejected(self):
        real_snapshot = common._snapshot; moved = False
        def snapshot(path):
            nonlocal moved
            result = real_snapshot(path)
            if Path(path) == self.paths[-1] and not moved:
                moved = True
                self.code.rename(self.root/'DEBUG_old_code')
                self.code.mkdir()
            return result
        with patch.object(common, '_snapshot', side_effect=snapshot):
            with self.assertRaisesRegex(ValueError, 'ancestor changed'):
                self.lease._inputs()

    def test_in_memory_request_mutation_remains_fatal(self):
        self.lease.request['DEBUG'] = False
        with self.assertRaisesRegex(ValueError, 'request changed'):
            self.lease._inputs()


if __name__ == '__main__':
    unittest.main()
