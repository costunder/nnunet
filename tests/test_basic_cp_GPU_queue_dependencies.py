"""CPU DEBUG: pending successors never create CUDA on their old GPU."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hiercp_v1x import basic_cp_GPU_queue_dependencies as queue
from tools import run_basic_cp_RAM512 as runner


class StopObservation(Exception):
    pass


class QueueDependencyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root/'queue.json'
        self.path.write_text(json.dumps(queue.receipt_value()))
        self.identity = dict(pid=123, create_time=456., UID=41833, command=['CPU_DEBUG'])
        self.request = dict(root=str(self.root/'GPU0_multi5'), gpu=0, arm='multi5',
            controller_identity=self.identity, code_sha256={queue.FILE: 'a'*64},
            GPU_queue_deferred_transfer=dict(path=str(self.path), sha256=self.sha(self.path)))
        self.upstream = self.root/'GPU5_candidates2048'
        self.upstream.mkdir()
        self.lease = mock.Mock()
        del self.lease._group_view
        self.lease._admit.side_effect = self.admit
        self.lease.self_identity = self.identity
        self.lease.registered_requests = {5: dict(root=str(self.upstream), arm='candidates2048')}

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def admit(self, path, digest):
        if self.sha(path) != digest:
            raise ValueError('CPU DEBUG pinned input changed')

    @staticmethod
    def complete_state():
        return dict(status='COMPLETE', stage='complete', completed_epochs=250, evaluated_cases=26,
            full_native_training=True, full_native_evaluation=True, historical_metrics_complete=True,
            reports={'schema1': 'actual_saved_report'}, archive={'receipt': 'actual_archive_receipt'})

    def test_no_defer_marker_preserves_normal_execution_without_observation(self):
        queue.wait_for_transfer({}, self.lease, mock.Mock(), completed_epochs=None)
        self.lease._admit.assert_not_called()
        queue.reject_native_dispatch({})

    def test_admits_only_exact_same_group_approved_dependencies(self):
        self.assertEqual(queue.deferred_rule(self.request, self.admit), queue.QUEUES[0])
        candidate = copy.deepcopy(self.request)
        candidate.update(gpu=4, arm='candidates1024', root=str(self.root/'GPU4_candidates1024'))
        self.assertEqual(queue.deferred_rule(candidate, self.admit), queue.QUEUES[1])

    def test_pending_gate_remains_waiting_even_after_upstream_complete(self):
        (self.upstream/'status.json').write_text(json.dumps(self.complete_state()))
        statuses = []
        def stop(_):
            raise StopObservation()
        with self.assertRaises(StopObservation):
            queue.wait_for_transfer(self.request, self.lease, lambda *args, **kw: statuses.append((args, kw)),
                completed_epochs=116, sleep=stop)
        self.lease.observe_waiting.assert_called_once_with()
        args, status = statuses[0]
        self.assertEqual(args[1:], ('waiting_GPU_queue_transfer', 'WAITING'))
        self.assertEqual(status['completed_epochs'], 116)
        self.assertTrue(status['upstream_reported_complete'])
        self.assertTrue(status['original_GPU_autostart_disabled'])
        self.assertFalse(status['CUDA_dispatch_started'])

    def test_ordinary_upstream_progress_preserves_actual_checkpoint_epoch(self):
        (self.upstream/'status.json').write_text(json.dumps(dict(status='RUNNING', stage='train', completed_epochs=18)))
        status = mock.Mock()
        with self.assertRaises(StopObservation):
            queue.wait_for_transfer(self.request, self.lease, status, completed_epochs=117,
                sleep=mock.Mock(side_effect=StopObservation()))
        self.assertEqual(status.call_args.kwargs['completed_epochs'], 117)
        self.assertFalse(status.call_args.kwargs['upstream_reported_complete'])

    def test_pinned_plan_mutation_is_rejected_before_waiting(self):
        self.path.write_text(self.path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'pinned input changed'):
            queue.wait_for_transfer(self.request, self.lease, mock.Mock(), completed_epochs=116)
        self.lease.observe_waiting.assert_not_called()

    def test_wrong_dependency_arm_gpu_or_group_is_rejected(self):
        mutations = ({'gpu': 1}, {'arm': 'multi7'}, {'gpu': True},
            {'root': str(self.root.parent/'unrelated'/'GPU0_multi5')})
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                queue.deferred_rule(dict(self.request, **mutation), self.admit)

    def test_unapproved_plan_with_recomputed_sha_is_still_rejected(self):
        value = queue.receipt_value()
        value['queues'][0]['target_physical_GPU'] = 6
        self.path.write_text(json.dumps(value))
        self.request[queue.FIELD]['sha256'] = self.sha(self.path)
        with self.assertRaisesRegex(ValueError, 'explicitly approved'):
            queue.deferred_rule(self.request, self.admit)

    def test_upstream_role_cannot_be_silently_substituted(self):
        self.lease.registered_requests[5]['arm'] = 'blend50'
        with self.assertRaisesRegex(ValueError, 'actual current upstream'):
            queue.wait_for_transfer(self.request, self.lease, mock.Mock(), completed_epochs=116)

    def test_only_incomplete_actual_epoch_and_exact_controller_are_allowed(self):
        for epoch in (None, 0, 250, -1, True, 116.5):
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                queue.wait_for_transfer(self.request, self.lease, mock.Mock(), completed_epochs=epoch)
        self.lease.self_identity = dict(self.identity, pid=124)
        with self.assertRaisesRegex(RuntimeError, 'exact waiting controller'):
            queue.wait_for_transfer(self.request, self.lease, mock.Mock(), completed_epochs=116)

    def test_host_lifetime_or_resource_failure_does_not_release_queue(self):
        self.lease.observe_waiting.side_effect = RuntimeError('actual lifetime expired')
        status = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, 'lifetime expired'):
            queue.wait_for_transfer(self.request, self.lease, status, completed_epochs=116)
        status.assert_not_called()

    def test_in_memory_removal_cannot_release_waiting_original_controller(self):
        def remove(_):
            self.request.pop(queue.FIELD)
        with self.assertRaisesRegex(RuntimeError, 'changed in memory'):
            queue.wait_for_transfer(self.request, self.lease, mock.Mock(), completed_epochs=116, sleep=remove)

    def test_full250_and_best26_and_reports_required_for_reported_ready(self):
        self.assertTrue(queue.upstream_complete(self.complete_state()))
        for key, value in (('status', 'RUNNING'), ('stage', 'train'), ('completed_epochs', 249),
                ('evaluated_cases', 25), ('full_native_training', False), ('full_native_evaluation', False),
                ('historical_metrics_complete', False), ('reports', {}), ('archive', {})):
            with self.subTest(key=key):
                self.assertFalse(queue.upstream_complete(dict(self.complete_state(), **{key: value})))

    def test_pending_native_cli_and_actor_cannot_reach_resource_or_popen(self):
        with mock.patch.object(runner, '_lease_for') as lease, mock.patch.object(runner, '_ACTOR') as actor:
            with self.assertRaisesRegex(RuntimeError, 'fresh cohort'):
                runner.resource_guard(self.request)
            with self.assertRaisesRegex(RuntimeError, 'fresh cohort'):
                runner.actor(self.request, ['CPU_DEBUG'], 'train', {})
        lease.assert_not_called()
        actor.assert_not_called()

    def test_private_run_gates_before_old_gpu_admission_and_bank_preparation(self):
        with mock.patch.dict(runner._SCOPE, _carry_complete_pipeline=lambda _: False,
                _wait_for_queue_transfer=mock.Mock(side_effect=StopObservation()),
                _wait_for_GPU=mock.Mock(), prepare_bank=mock.Mock()):
            with self.assertRaises(StopObservation):
                runner.run(self.request)
            runner._SCOPE['_wait_for_GPU'].assert_not_called()
            runner._SCOPE['prepare_bank'].assert_not_called()


if __name__ == '__main__':
    unittest.main()
