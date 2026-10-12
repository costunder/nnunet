"""CPU UNIT: bounded /proc exit-transition observations, no CUDA claim."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hiercp_v1x import v25_direct_resources as direct
from hiercp_v1x import v25_terminal_descendant_resources as runtime


class EmptyCmdlineTransitionTests(unittest.TestCase):
    def setUp(self):
        self.identity = dict(pid=202, create_time=202., UID=41833,
            command=['python', 'CPU_DEBUG_preparation_worker.py'])
        self.worker = Mock()
        self.worker.pid = 202
        self.worker.status.return_value = direct.psutil.STATUS_RUNNING
        self.worker.create_time.return_value = 202.
        self.worker.uids.return_value = SimpleNamespace(real=41833)
        self.worker.cmdline.return_value = []
        self.lease = SimpleNamespace(managed={})

    def test_empty_running_then_sleeping_then_actual_zombie_is_excluded(self):
        self.worker.status.side_effect = [direct.psutil.STATUS_RUNNING,
            direct.psutil.STATUS_SLEEPING, direct.psutil.STATUS_SLEEPING,
            direct.psutil.STATUS_SLEEPING, direct.psutil.STATUS_ZOMBIE,
            direct.psutil.STATUS_ZOMBIE, direct.psutil.STATUS_ZOMBIE]
        self.lease.managed[202] = self.identity.copy()
        self.assertIsNone(runtime._descendant(self.lease, self.worker))
        self.assertEqual(self.worker.cmdline.call_count, 2)
        self.assertEqual(self.lease._terminal_descendants[202],
            dict(pid=202, create_time=202., UID=41833))
        self.assertEqual(self.lease.managed[202], self.identity)

    def test_empty_then_zombie_after_third_cmdline_observation_is_excluded(self):
        self.worker.status.side_effect = [direct.psutil.STATUS_RUNNING] * 5 + [
            direct.psutil.STATUS_ZOMBIE, direct.psutil.STATUS_ZOMBIE,
            direct.psutil.STATUS_ZOMBIE]
        self.assertIsNone(runtime._descendant(self.lease, self.worker))
        self.assertEqual(self.worker.cmdline.call_count, 3)

    def test_empty_then_actual_disappearance_is_propagated_for_tree_exclusion(self):
        self.worker.status.side_effect = [direct.psutil.STATUS_RUNNING,
            direct.psutil.STATUS_SLEEPING, direct.psutil.NoSuchProcess(202)]
        with self.assertRaises(direct.psutil.NoSuchProcess):
            runtime._descendant(self.lease, self.worker)
        self.assertNotIn(202, self.lease.managed)
        self.assertFalse(hasattr(self.lease, '_terminal_descendants'))

    def test_live_argv_after_empty_reread_requires_complete_same_identity(self):
        self.worker.cmdline.side_effect = [[], self.identity['command']]
        self.lease.managed[202] = self.identity.copy()
        self.assertEqual(runtime._descendant(self.lease, self.worker), self.identity)
        self.assertEqual(self.worker.cmdline.call_count, 2)
        self.assertFalse(hasattr(self.lease, '_terminal_descendants'))

    def test_genuine_live_empty_argv_is_rejected_after_exactly_three_reads(self):
        with patch.object(runtime.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'Exact live descendant'):
                runtime._descendant(self.lease, self.worker)
        self.assertEqual([call.args for call in sleep.call_args_list], [(.01,), (.01,)])
        self.assertEqual(self.worker.cmdline.call_count, 3)
        self.assertEqual(self.worker.status.call_count, 6)
        self.assertFalse(hasattr(self.lease, '_terminal_descendants'))

    def test_complete_live_identity_never_waits_for_terminal_retry(self):
        self.worker.cmdline.return_value = self.identity['command']
        with patch.object(runtime.time, 'sleep') as sleep:
            self.assertEqual(runtime._descendant(self.lease, self.worker), self.identity)
        sleep.assert_not_called()

    def test_empty_retry_yields_before_kernel_terminal_status_becomes_visible(self):
        state = {'status': direct.psutil.STATUS_RUNNING}
        self.worker.status.side_effect = lambda: state['status']
        def yield_to_exit(delay):
            self.assertEqual(delay, .01)
            state['status'] = direct.psutil.STATUS_ZOMBIE
        with patch.object(runtime.time, 'sleep', side_effect=yield_to_exit) as sleep:
            self.assertIsNone(runtime._descendant(self.lease, self.worker))
        sleep.assert_called_once_with(.01)
        self.assertEqual(self.worker.cmdline.call_count, 1)

    def test_pid_birth_reuse_between_empty_observations_is_rejected(self):
        self.worker.create_time.side_effect = [202., 999.]
        with self.assertRaisesRegex(RuntimeError, 'PID/birth/UID changed'):
            runtime._descendant(self.lease, self.worker)
        self.assertEqual(self.worker.cmdline.call_count, 1)

    def test_uid_change_between_empty_observations_is_rejected(self):
        self.worker.uids.side_effect = [SimpleNamespace(real=41833),
            SimpleNamespace(real=1)]
        with self.assertRaisesRegex(RuntimeError, 'Foreign-UID'):
            runtime._descendant(self.lease, self.worker)

    def test_terminal_transition_must_retain_the_first_observed_birth(self):
        self.worker.status.side_effect = [direct.psutil.STATUS_RUNNING,
            direct.psutil.STATUS_ZOMBIE, direct.psutil.STATUS_ZOMBIE]
        self.worker.create_time.side_effect = [202., 999.]
        with self.assertRaisesRegex(RuntimeError, 'PID/birth/UID changed'):
            runtime._descendant(self.lease, self.worker)
        self.assertFalse(hasattr(self.lease, '_terminal_descendants'))

    def test_managed_birth_or_argv_change_after_empty_is_rejected(self):
        self.lease.managed[202] = self.identity.copy()
        self.worker.create_time.return_value = 999.
        with self.assertRaisesRegex(RuntimeError, 'Managed direct child PID/birth/argv'):
            runtime._descendant(self.lease, self.worker)
        self.worker.create_time.return_value = 202.
        self.worker.cmdline.side_effect = [[], ['python', 'unrelated.py']]
        with self.assertRaisesRegex(RuntimeError, 'Managed direct child PID/birth/argv'):
            runtime._descendant(self.lease, self.worker)

    def test_previously_terminal_pid_cannot_regain_live_admission(self):
        self.lease._terminal_descendants = {
            202: dict(pid=202, create_time=202., UID=41833)}
        self.worker.cmdline.side_effect = [[], self.identity['command']]
        with self.assertRaisesRegex(RuntimeError, 'Previously terminal'):
            runtime._descendant(self.lease, self.worker)

    def test_access_denied_is_not_hidden_by_empty_retry(self):
        self.worker.cmdline.side_effect = [[], direct.psutil.AccessDenied(202)]
        with self.assertRaises(direct.psutil.AccessDenied):
            runtime._descendant(self.lease, self.worker)

    def test_invalid_nonempty_argv_and_fake_zombie_exception_are_rejected(self):
        self.worker.cmdline.side_effect = [[], ['']]
        with self.assertRaisesRegex(RuntimeError, 'Exact live descendant'):
            runtime._descendant(self.lease, self.worker)
        self.worker.cmdline.side_effect = [[], direct.psutil.ZombieProcess(202)]
        with self.assertRaisesRegex(RuntimeError, 'must have actual zombie status'):
            runtime._descendant(self.lease, self.worker)


if __name__ == '__main__':
    unittest.main()
