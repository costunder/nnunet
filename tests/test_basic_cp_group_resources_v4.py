"""CPU-only DEBUG proof of bounded exact affinity snapshots, without training."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_group_resources_v4 as group
from hiercp_v1x import basic_cp_seven_resources as resource
from tests import test_basic_cp_resources_v3 as fixture


class GroupV4Tests(unittest.TestCase):
    start = fixture.GroupV3Tests.start
    process = fixture.GroupV3Tests.process
    query = fixture.GroupV3Tests.query

    def setUp(self):
        fixture.GroupV3Tests.setUp(self)
        source = self.root / group.GUARD_FILE
        source.write_text('exact V4 diagnostic observation guard fixture\n')
        self.value['code_sha256'][group.GUARD_FILE] = resource.sha(source)
        self.value['group_observation_revision'] = 4
        request = Path(self.value['root']) / 'request.json'
        request.write_text(json.dumps(self.value))
        self.document['registration']['0']['request_sha256'] = resource.sha(request)
        self.registry_path.write_text(json.dumps(self.document))
        self.lease = group.BasicLease(self.value)
        self.start(patch.object(group.resource, '_kernel_exiting', side_effect=lambda p: getattr(p, 'kernel_exiting', False)))

    def test_transient_child_mask_refreshes_entire_snapshot_and_actual_resources(self):
        stage = self.actors[305]
        expected = list(group.SHARED_CPU_AFFINITY)
        with patch.object(stage, 'cpu_affinity', side_effect=[[40], expected]), patch.object(group.time, 'sleep') as sleep:
            tree, observation = self.lease._group_tree(5)
        self.assertIn(stage.pid, tree)
        self.assertEqual(observation['actual_CPU_affinity'], expected)
        record = observation['bounded_CPU_affinity_reobservation']
        self.assertTrue(record['recovered'])
        self.assertEqual(record['reobservations'], 1)
        self.assertEqual(record['mismatches'][0]['actual_CPU_affinity'], [40])
        sleep.assert_called_once_with(.05)

    def test_persistent_wrong_mask_still_fails_with_exact_diagnostic_fields(self):
        self.actors[304].cpus = [24]
        with patch.object(group.time, 'sleep') as sleep, self.assertRaisesRegex(RuntimeError, 'Persistent exact group CPU affinity') as caught:
            self.lease._group_tree(4)
        self.assertEqual(sleep.call_count, 3)
        for field in ('"gpu": 4', '"pid": 304', '"UID": 41833', '"birth"', '"argv"',
                      '"actual_CPU_affinity": [24]', '"expected_CPU_affinity"', '"kernel_exiting": false'):
            self.assertIn(field, str(caught.exception))

    def test_changed_shared_controller_stage_is_refreshed_with_its_children(self):
        controller = self.actors[105]
        child = self.actors[305]
        snapshots = [(controller, dict(status='LIVE'), list(resource.CPUS[5]), [child]),
                     (controller, dict(status='LIVE'), list(group.SHARED_CPU_AFFINITY), [child])]
        with (patch.object(self.lease, '_controller_stage_snapshot', side_effect=snapshots) as snapshot,
              patch.object(group.time, 'sleep')):
            tree, observation = self.lease._group_tree(5)
        self.assertEqual(snapshot.call_count, 2)
        self.assertIn(child.pid, tree)
        self.assertTrue(observation['shared_CPU_bank_stage'])

    def test_retry_same_controller_does_not_admit_pid_reuse(self):
        self.actors[304].cpus = [24]
        def reuse(seconds):
            self.actors[104].birth += 1
        with patch.object(group.time, 'sleep', side_effect=reuse), self.assertRaisesRegex(RuntimeError, 'reused'):
            self.lease._group_tree(4)

    def test_retry_live_changed_argv_is_never_an_affinity_recovery(self):
        stage = self.actors[304]
        stage.cpus = [24]
        def replace(seconds):
            stage.command = ['python', 'replacement']
            stage.cpus = list(resource.CPUS[4])
        with patch.object(group.time, 'sleep', side_effect=replace), self.assertRaisesRegex(RuntimeError, 'argv'):
            self.lease._group_tree(4)

    def test_exact_kernel_exiting_wrong_mask_retains_actual_rss_and_records_unadmitted_mask(self):
        stage = self.actors[304]
        stage.cpus = [24]
        stage.kernel_exiting = True
        stage.rss = 12 * 2**30
        result = self.lease.verify(force=True)
        self.assertGreaterEqual(result['group_RSS_by_GPU_bytes']['4'], stage.rss)
        record = self.lease._terminal_observations[304]
        self.assertTrue(record['RSS_still_accounted'])
        self.assertFalse(record['CPU_mask_admitted'])
        self.assertEqual(record['actual_CPU_affinity'], [24])
        self.assertIn(304, result['group_controller_observations']['4']['exact_kernel_exiting_unadmitted_CPU_mask_PIDs'])

    def test_missing_kernel_flag_has_no_terminal_or_zero_rss_fallback(self):
        stage = self.actors[304]
        stage.cpus = []
        stage.rss = 12 * 2**30
        with patch.object(group.time, 'sleep'), self.assertRaisesRegex(RuntimeError, 'Persistent'):
            self.lease._group_tree(4)
        self.assertNotIn(304, self.lease._terminal_observations)

    def test_explicit_revision4_and_new_guard_sha_required_without_mutating_v3_budget(self):
        self.assertEqual(group.AGGREGATE_CAPACITY, 800 * 2**30)
        self.assertEqual(group.CANDIDATE_RSS_LIMIT, 384 * 2**30)
        self.assertEqual(group.NATIVE_RSS_LIMIT, 96 * 2**30)
        for replacement in (3, True, 4.0):
            value = copy.deepcopy(self.value)
            value['group_observation_revision'] = replacement
            with self.assertRaisesRegex(ValueError, 'revision4'):
                group.BasicLease(value)
        value = copy.deepcopy(self.value)
        value['code_sha256'].pop(group.GUARD_FILE)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            group.BasicLease(value)

    def test_unadmitted_foreign_uid_does_not_gain_bounded_retry_permission(self):
        self.actors[304].uid = 999
        self.actors[304].cpus = [24]
        with patch.object(group.time, 'sleep') as sleep, self.assertRaisesRegex(RuntimeError, 'foreign'):
            self.lease._group_tree(4)
        sleep.assert_not_called()

    def test_admission_deadline_caps_retry_wait_without_resource_shrink(self):
        self.actors[304].cpus = [24]
        with (patch.object(group.time, 'monotonic', side_effect=[0, .31]),
              patch.object(group.time, 'sleep') as sleep,
              self.assertRaisesRegex(RuntimeError, 'Persistent')):
            self.lease._group_tree(4)
        sleep.assert_not_called()

    def test_recovered_mask_after_deadline_is_not_a_late_admission(self):
        failure = group._AffinityMismatch(dict(gpu=4, pid=304, UID=41833,
            actual_CPU_affinity=[24], expected_CPU_affinity=list(resource.CPUS[4]), kernel_exiting=False))
        with (patch.object(self.lease, '_group_tree_once', side_effect=[failure, ({}, {})]),
              patch.object(group.time, 'monotonic', side_effect=[0, .1, .2, .31]),
              patch.object(group.time, 'sleep'), self.assertRaisesRegex(RuntimeError, 'Persistent')):
            self.lease._group_tree(4)


if __name__ == '__main__':
    unittest.main()
