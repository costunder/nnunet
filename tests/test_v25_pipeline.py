"""DEBUG CPU contract regressions; these fixtures do not execute training."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import psutil

from tools import run_v25_pipeline as pipeline


GIB = 2**30


def canonical_sha(value):
    contents = {key: item for key, item in value.items() if key != 'request_sha256'}
    return hashlib.sha256(json.dumps(contents, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf8')


class PreparationFixture:
    def __init__(self, root):
        self.group = root / 'group'
        self.cache = root / 'cache'
        self.request_path = self.group / 'A/request.json'
        self.prepared_path = self.group / 'A/preparation.json'
        self.index_path = self.cache / 'local/index.json'
        self.request = dict(format='v25_complete_execution_request_v1', arm='A',
                            config=dict(epochs=40, updates=680, debug=False),
                            source_sha256={'v25_factory.py': 'a' * 64},
                            input_cache=str(self.cache), output=str(self.group / 'A'))
        self.request['request_sha256'] = canonical_sha(self.request)
        self.index = dict(complete=True, debug=False,
                          records=[{'record_id': index} for index in range(14102)])
        write_json(self.request_path, self.request)
        self.update_index()

    def update_index(self):
        write_json(self.index_path, self.index)
        self.prepared = dict(request_sha256=self.request['request_sha256'],
                             local_index_sha256=hashlib.sha256(self.index_path.read_bytes()).hexdigest(),
                             full_upper={'complete': True})
        write_json(self.prepared_path, self.prepared)


class PreparationContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='DEBUG_v25_pipeline_')
        self.addCleanup(self.temporary.cleanup)
        self.fixture = PreparationFixture(Path(self.temporary.name))

    def validate(self):
        return pipeline.validate_preparation(self.fixture.group, self.fixture.cache)

    def test_complete_all_14102_source_bound_records_are_admitted(self):
        self.assertEqual(self.validate(), self.fixture.prepared)
        self.assertEqual(len(json.loads(self.fixture.index_path.read_text())['records']), 14102)

    def test_fewer_or_extra_records_are_not_silently_subsetted(self):
        for size in (14101, 14103):
            with self.subTest(records=size):
                self.fixture.index['records'] = [{'record_id': index} for index in range(size)]
                self.fixture.update_index()
                with self.assertRaisesRegex(ValueError, 'full14102'):
                    self.validate()
                self.assertEqual(len(json.loads(self.fixture.index_path.read_text())['records']), size)

    def test_only_explicit_production_debug_false_and_complete_true_are_admitted(self):
        for field, values in (('debug', (True, 0, None)), ('complete', (False, 1, None))):
            for value in values:
                with self.subTest(field=field, value=value):
                    self.fixture.index.update(debug=False, complete=True)
                    self.fixture.index[field] = value
                    self.fixture.update_index()
                    with self.assertRaisesRegex(ValueError, 'full14102'):
                        self.validate()
        for field in ('debug', 'complete'):
            with self.subTest(missing=field):
                self.fixture.index.update(debug=False, complete=True)
                self.fixture.index.pop(field)
                self.fixture.update_index()
                with self.assertRaises(ValueError):
                    self.validate()

    def test_changed_scientific_request_is_rejected_even_if_receipts_share_old_sha(self):
        self.fixture.request['config']['epochs'] = 1
        write_json(self.fixture.request_path, self.fixture.request)
        with self.assertRaisesRegex(ValueError, 'source-bound'):
            self.validate()

    def test_equal_forged_request_and_receipt_sha_are_rejected(self):
        self.fixture.request['request_sha256'] = 'b' * 64
        self.fixture.prepared['request_sha256'] = 'b' * 64
        write_json(self.fixture.request_path, self.fixture.request)
        write_json(self.fixture.prepared_path, self.fixture.prepared)
        with self.assertRaises(ValueError):
            self.validate()

    def test_preparation_must_belong_to_the_exact_request(self):
        self.fixture.prepared['request_sha256'] = 'c' * 64
        write_json(self.fixture.prepared_path, self.fixture.prepared)
        with self.assertRaises(ValueError):
            self.validate()

    def test_index_byte_mutation_is_rejected_even_when_semantic_json_is_equal(self):
        self.fixture.index_path.write_text(json.dumps(self.fixture.index, indent=2), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'full14102'):
            self.validate()

    def test_partial_json_is_an_explicit_error_without_completion_fallback(self):
        self.fixture.index_path.write_text('{"records":', encoding='utf8')
        with self.assertRaises(json.JSONDecodeError):
            self.validate()
        self.assertFalse((self.fixture.group / 'completion.json').exists())


def metadata_file(*, size=1, blocks=None, symlink=False):
    values = {'st_size': size}
    if blocks is not None:
        values['st_blocks'] = blocks
    return SimpleNamespace(is_symlink=lambda: symlink, is_file=lambda: True,
                           stat=lambda: SimpleNamespace(**values))


class CacheMemoryContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='DEBUG_v25_memory_')
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name) / 'cache'
        self.cache.mkdir()

    def admit_metadata(self, *files):
        # Mock filesystem metadata only; execute the real admission validator.
        with mock.patch.object(Path, 'rglob', return_value=iter(files)):
            return pipeline.cache_memory_admission(self.cache)

    def test_real_files_are_all_charged_and_preserved_in_128_gib_budget(self):
        files = (self.cache / 'one.bin', self.cache / 'two.bin')
        files[0].write_bytes(b'x' * 513)
        files[1].write_bytes(b'y' * 7)
        before = {path: path.read_bytes() for path in files}
        expected = sum(getattr(path.stat(), 'st_blocks', (path.stat().st_size + 511) // 512) * 512
                       for path in files)
        value = pipeline.cache_memory_admission(self.cache)
        self.assertEqual(value['allocated_bytes'], expected)
        self.assertEqual(value['shared_cache_memory_reserve_bytes'], 32 * GIB)
        self.assertEqual(value['native_process_and_children_RSS_bytes'], 96 * GIB)
        self.assertEqual(value['PBS_memory_bytes'], 128 * GIB)
        self.assertTrue(value['full_cache_preserved'])
        self.assertIs(value['debug'], False)
        self.assertEqual({path: path.read_bytes() for path in files}, before)

    def test_sparse_file_uses_actual_allocated_blocks_instead_of_logical_size(self):
        value = self.admit_metadata(metadata_file(size=128 * GIB, blocks=16))
        self.assertEqual(value['allocated_bytes'], 8192)

    def test_exact_32_gib_fits_and_one_more_block_is_refused_without_cap(self):
        value = self.admit_metadata(metadata_file(size=32 * GIB, blocks=32 * GIB // 512))
        self.assertEqual(value['allocated_bytes'], 32 * GIB)
        with self.assertRaisesRegex(MemoryError, '32GiB.*no cap/fallback'):
            self.admit_metadata(metadata_file(blocks=32 * GIB // 512), metadata_file(blocks=1))

    def test_platform_without_blocks_uses_all_file_bytes_rounded_to_blocks(self):
        value = self.admit_metadata(metadata_file(size=513), metadata_file(size=1))
        self.assertEqual(value['allocated_bytes'], 1536)

    def test_external_symlink_file_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'external symlinks'):
            self.admit_metadata(metadata_file(symlink=True))

    def test_cache_root_or_ancestor_symlink_is_refused_before_traversal(self):
        for target in (self.cache, self.cache.parent):
            with self.subTest(target=str(target)):
                with mock.patch.object(Path, 'is_symlink', autospec=True,
                                       side_effect=lambda path: path == target), \
                     mock.patch.object(Path, 'rglob') as traversal:
                    with self.assertRaisesRegex(ValueError, 'no symlink ancestors'):
                        pipeline.cache_memory_admission(self.cache)
                    traversal.assert_not_called()

    def test_missing_cache_directory_is_not_an_empty_cache_success(self):
        with self.assertRaises(ValueError):
            pipeline.cache_memory_admission(self.cache / 'missing')


class ActorAndLeaseContractTests(unittest.TestCase):
    def setUp(self):
        self.identity = dict(child_pid=742, child_create_time=123.5,
                             child_command=['python', '-B', 'run_v25.py', '--mode', 'prepare'], UID=41833)

    def process(self, **changes):
        values = dict(self.identity, **changes)
        return SimpleNamespace(status=lambda: values.get('status', psutil.STATUS_RUNNING),
                               create_time=lambda: values['child_create_time'],
                               cmdline=lambda: values['child_command'],
                               uids=lambda: SimpleNamespace(real=values['UID']))

    def test_exact_owned_birth_command_and_uid_is_the_same_actor(self):
        with mock.patch.object(psutil, 'Process', return_value=self.process()) as lookup:
            self.assertTrue(pipeline.same_actor(self.identity))
        lookup.assert_called_once_with(742)

    def test_pid_reuse_changed_argv_and_foreign_uid_are_not_owned_actor(self):
        for changes in ({'child_create_time': 124.0}, {'UID': 99},
                        {'child_command': ['python', 'unrelated.py']}):
            with self.subTest(changes=changes), mock.patch.object(psutil, 'Process', return_value=self.process(**changes)):
                self.assertFalse(pipeline.same_actor(self.identity))

    def test_gone_or_zombie_actor_is_not_running(self):
        with mock.patch.object(psutil, 'Process', side_effect=psutil.NoSuchProcess(742)):
            self.assertFalse(pipeline.same_actor(self.identity))
        with mock.patch.object(psutil, 'Process', return_value=self.process(status=psutil.STATUS_ZOMBIE)):
            self.assertFalse(pipeline.same_actor(self.identity))

    def test_expired_real_lease_aborts_without_pbs_gpu_calls_or_epoch_reduction(self):
        from hiercp_v1x import v25_server_resources as resources
        with tempfile.TemporaryDirectory(prefix='DEBUG_v25_lease_') as root:
            path = Path(root) / 'allocation.json'
            value = dict(format='v25_owned_GPU_allocation_v1', UID=41833, owner='aicompetition06',
                         node='ece-a6gpu6', gpu=6, job_id='129999.ECE-util1', GPU_UUID='GPU-DEBUG',
                         CPU_affinity=[32, 33, 34, 35], expires_at_unix=1000,
                         scientific_contract={'epochs': 40, 'updates': 680, 'native_epochs': 250})
            write_json(path, value)
            original = path.read_bytes()
            with mock.patch.object(resources.os, 'getuid', create=True, return_value=41833), \
                 mock.patch.object(resources.time, 'time', return_value=900), \
                 mock.patch.object(resources.subprocess, 'run') as external:
                lease = resources.AllocationLease(path)
                with self.assertRaisesRegex(RuntimeError, 'preserve exact checkpoint.*next valid lease'):
                    lease.verify(force=True)
                external.assert_not_called()
            self.assertEqual(path.read_bytes(), original)


class InterfaceContractTests(unittest.TestCase):
    def arguments(self):
        names = ('group', 'allocation-request', 'storage-admission', 'prepare-launch',
                 'native-experiment', 'inventory', 'stunet-checkpoint', 'input-cache',
                 'source-bank', 'baseline-preprocessed', 'project', 'basic-validation', 'evaluator-root')
        return [item for name in names for item in ('--' + name, str(Path('DEBUG_inputs') / name))]

    def test_explicit_evaluator_and_complete_inputs_are_required(self):
        value = pipeline.parse(self.arguments())
        self.assertEqual(value.evaluator_root, Path('DEBUG_inputs/evaluator-root'))
        self.assertEqual(value.input_cache, Path('DEBUG_inputs/input-cache'))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.parse(self.arguments()[:-2])

    def test_no_epoch_subset_cap_or_fast_fallback_cli_is_available(self):
        for extra in (['--epochs', '1'], ['--max-records', '10'], ['--subset-size', '1'],
                      ['--debug'], ['--fast-mode']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pipeline.parse(self.arguments() + extra)

    def test_status_replacement_preserves_latest_state_and_leaves_no_temporary(self):
        with tempfile.TemporaryDirectory(prefix='DEBUG_v25_status_') as root:
            path = Path(root) / 'status.json'
            pipeline.atomic_status(path, {'status': 'RUNNING', 'epoch': 27})
            pipeline.atomic_status(path, {'status': 'FAILED', 'error': 'lease exhausted; receipt preserved'})
            self.assertEqual(json.loads(path.read_text())['status'], 'FAILED')
            self.assertEqual(list(Path(root).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
