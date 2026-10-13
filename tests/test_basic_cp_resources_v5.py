"""CPU-only DEBUG: natural procfs ESRCH and final group/shared observations."""
import copy
import errno
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_ablation_resources as common
from hiercp_v1x import basic_cp_group_resources_v5 as group
from hiercp_v1x import basic_cp_shared_bank_resources_v5 as shared
from hiercp_v1x import basic_cp_shared_bank_resources_v3 as shared_prior
from tests import test_basic_cp_resources_v3 as fixture


class ProcStatTests(unittest.TestCase):
    def test_kernel_ESRCH_is_exact_no_such_process_not_fake_flags(self):
        process = SimpleNamespace(pid=123, is_running=lambda: True)
        with (patch.object(common.Path, 'read_text', side_effect=ProcessLookupError(errno.ESRCH, 'gone')),
              self.assertRaises(common.psutil.NoSuchProcess) as caught):
            common._task_flags(process)
        self.assertEqual(caught.exception.pid, 123)

    def test_non_ESRCH_process_lookup_and_permission_failures_are_preserved(self):
        process = SimpleNamespace(pid=123, is_running=lambda: True)
        for failure in (ProcessLookupError(errno.EPERM, 'wrong error'), PermissionError(errno.EACCES, 'denied')):
            with patch.object(common.Path, 'read_text', side_effect=failure), self.assertRaises(type(failure)) as caught:
                common._task_flags(process)
            self.assertIs(caught.exception, failure)

    def test_ENOENT_still_requires_actual_nonrunning_task(self):
        for running in (False, True):
            process = SimpleNamespace(pid=123, is_running=lambda: running)
            expected = RuntimeError if running else common.psutil.NoSuchProcess
            with patch.object(common.Path, 'read_text', side_effect=FileNotFoundError), self.assertRaises(expected):
                common._task_flags(process)

    def test_actual_flag_parser_and_future_clones_include_ESRCH_fix(self):
        process = SimpleNamespace(pid=123)
        with patch.object(common.Path, 'read_text', return_value='123 (worker name) S 1 2 3 4 5 68 7'):
            self.assertEqual(common._task_flags(process), 68)
            self.assertTrue(common._fork_no_exec(process))
            self.assertTrue(common._kernel_exiting(process))
        for module in (shared_prior, group.prior.resource):
            with (patch.object(module.Path, 'read_text', side_effect=ProcessLookupError(errno.ESRCH, 'gone')),
                  self.assertRaises(module.psutil.NoSuchProcess)):
                module._task_flags(process)


class GroupV5Tests(unittest.TestCase):
    start = fixture.GroupV3Tests.start
    process = fixture.GroupV3Tests.process
    query = fixture.GroupV3Tests.query

    def setUp(self):
        fixture.GroupV3Tests.setUp(self)
        dependency = self.root / group.prior.GUARD_FILE
        dependency.write_text('exact bounded observation dependency\n')
        self.value['code_sha256'][group.prior.GUARD_FILE] = common.sha(dependency)
        source = self.root / group.GUARD_FILE
        source.write_text('exact final group observation guard\n')
        self.value['code_sha256'][group.GUARD_FILE] = common.sha(source)
        self.value['group_observation_revision'] = 5
        path = Path(self.value['root']) / 'request.json'
        path.write_text(json.dumps(self.value))
        self.document['registration']['0']['request_sha256'] = common.sha(path)
        self.registry_path.write_text(json.dumps(self.document))
        self.lease = group.BasicLease(self.value)
        self.start(patch.object(group.prior.resource, '_kernel_exiting', side_effect=lambda p: getattr(p, 'kernel_exiting', False)))

    def test_obs5_keeps_marker3_384_800_and_native96_with_transient_observation(self):
        stage = self.actors[305]
        with (patch.object(stage, 'cpu_affinity', side_effect=[[40], list(group.prior.SHARED_CPU_AFFINITY)]),
              patch.object(group.prior.time, 'sleep')):
            tree, observation = self.lease._group_tree(5)
        self.assertIn(stage.pid, tree)
        self.assertTrue(observation['bounded_CPU_affinity_reobservation']['recovered'])
        self.assertEqual(self.value['shared_resource_guard_version'], 3)
        self.assertEqual(group.CANDIDATE_RSS_LIMIT, 384 * 2**30)
        self.assertEqual(group.AGGREGATE_CAPACITY, 800 * 2**30)
        self.assertEqual(group.NATIVE_RSS_LIMIT, 96 * 2**30)

    def test_wrong_revision_or_missing_new_guard_is_not_an_implicit_upgrade(self):
        for replacement in (4, True, 5.0):
            value = copy.deepcopy(self.value)
            value['group_observation_revision'] = replacement
            with self.assertRaisesRegex(ValueError, 'revision5'):
                group.BasicLease(value)
        value = copy.deepcopy(self.value)
        value['code_sha256'].pop(group.GUARD_FILE)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            group.BasicLease(value)

    def test_inherited_observation_helper_sha_cannot_be_omitted(self):
        value = copy.deepcopy(self.value)
        value['code_sha256'].pop(group.prior.GUARD_FILE)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            group.BasicLease(value)


class SharedV5Tests(unittest.TestCase):
    start = fixture.SharedV3Tests.start
    process = fixture.SharedV3Tests.process
    query = fixture.SharedV3Tests.query

    def setUp(self):
        fixture.SharedV3Tests.setUp(self)
        for filename in (shared.observation.GUARD_FILE, shared.prior._exit_fix.GUARD_FILE):
            dependency = self.root / filename
            dependency.write_text('exact bounded/terminal observation dependency\n')
            self.value['code_sha256'][filename] = common.sha(dependency)
        source = self.root / shared.SHARED_GUARD_FILE
        source.write_text('exact final CPU-bank observation guard\n')
        self.value['code_sha256'][shared.SHARED_GUARD_FILE] = common.sha(source)
        self.value['group_observation_revision'] = 5
        self.request_path.write_text(json.dumps(self.value))
        self.lease = shared.SharedBankLease(self.value)
        self.start(patch.object(shared.resource, '_kernel_exiting', side_effect=lambda p: getattr(p, 'kernel_exiting', False)))

    def test_transient_worker_mask_reobserves_full_actual_tree_then_admits_384(self):
        with (patch.object(self.worker, 'cpu_affinity', side_effect=[[40], list(shared.SHARED_CPU_AFFINITY)]),
              patch.object(shared.time, 'sleep') as sleep):
            tree = self.lease._tree(self.controller, self.stage)
        self.assertIn(self.worker.pid, tree)
        self.assertTrue(self.lease._shared_affinity_reobservations['recovered'])
        sleep.assert_called_once_with(.05)
        result = self.lease.verify(force=True)
        self.assertEqual(result['own_process_tree_RSS_limit_bytes'], 384 * 2**30)
        self.assertEqual(result['aggregate_native_and_cache_capacity_bytes'], 800 * 2**30)

    def test_persistent_mask_has_full_identity_evidence_and_no_reduced_resource_fallback(self):
        self.worker.cpus = [40]
        with patch.object(shared.time, 'sleep'), self.assertRaisesRegex(RuntimeError, 'Persistent exact shared') as caught:
            self.lease._tree(self.controller, self.stage)
        for field in ('"pid": 201', '"UID": 41833', '"actual_CPU_affinity": [40]', '"expected_CPU_affinity"', '"argv"', '"birth"'):
            self.assertIn(field, str(caught.exception))

    def test_exact_kernel_exiting_task_keeps_real_rss_and_unadmitted_mask_evidence(self):
        self.worker.cpus = []
        self.worker.kernel_exiting = True
        self.worker.rss = 12 * 2**30
        result = self.lease.verify(force=True)
        self.assertGreaterEqual(result['own_process_tree_RSS_bytes'], self.worker.rss)
        self.assertTrue(self.lease._terminal_observations[201]['RSS_still_accounted'])
        self.assertFalse(self.lease._terminal_observations[201]['CPU_mask_admitted'])

    def test_foreign_uid_or_changed_controller_identity_never_enters_retry(self):
        self.worker.uid = 999
        with patch.object(shared.time, 'sleep') as sleep, self.assertRaisesRegex(RuntimeError, 'foreign'):
            self.lease._tree(self.controller, self.stage)
        sleep.assert_not_called()

    def test_quick_still_avoids_all_source_metadata_scan_and_full_remains_periodic(self):
        self.lease.verify(force=True)
        with patch.object(self.lease, '_inputs', wraps=self.lease._inputs) as inputs:
            for _ in range(10):
                self.lease.quick_verify()
            self.assertEqual(inputs.call_count, 0)
            self.monotonic += 5
            self.lease.quick_verify()
            self.assertGreater(inputs.call_count, 0)

    def test_wrong_revision_missing_sha_and_native_preservation_stay_strict(self):
        for replacement in (4, True, 5.0):
            value = copy.deepcopy(self.value)
            value['group_observation_revision'] = replacement
            with self.assertRaisesRegex(ValueError, 'revision5'):
                shared.SharedBankLease(value)
        value = copy.deepcopy(self.value)
        value['code_sha256'].pop(shared.SHARED_GUARD_FILE)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            shared.SharedBankLease(value)
        self.assertEqual(self.lease.native_request['RSS_limit_bytes'], 96 * 2**30)
        self.assertEqual(len(self.lease.native_request['cpu_affinity']), 8)

    def test_used_observation_and_kernel_exit_helper_shas_are_both_pinned(self):
        for filename in (shared.observation.GUARD_FILE, shared.prior._exit_fix.GUARD_FILE):
            value = copy.deepcopy(self.value)
            value['code_sha256'].pop(filename)
            with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, 'SHA'):
                shared.SharedBankLease(value)


if __name__ == '__main__':
    unittest.main()
