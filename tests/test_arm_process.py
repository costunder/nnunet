"""UNIT owned-PID supervision; these tests do not run training or user jobs."""
from pathlib import Path
import os
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from hiercp_v1x.arm_process import OwnedArm


class Child:
    pid = 314159
    def __init__(self):
        self.returncode = None
        self.signals = []
        self.terminations = 0
    def poll(self):
        return self.returncode
    def send_signal(self, value):
        self.signals.append(value)
    def terminate(self):
        self.terminations += 1
        self.returncode = -signal.SIGTERM


class OwnedArmTests(unittest.TestCase):
    def fixture(self):
        child, time, events = Child(), [100.], []
        command = ['python', '-B', 'tools/run_comparison_arm.py', '--owned-child', '--arm', 'native_fixed']
        owner = dict(pid=child.pid, parent_pid=os.getpid(), create_time=42., argv=command,
                     uid=os.getuid() if hasattr(os, 'getuid') else 123)
        identities = [owner]
        calls = []
        def factory(argv, **kwargs):
            calls.append((argv, kwargs)); return child
        # Production is Linux-only. The identity hook uses the synthetic UNIT
        # UID on Windows, without spawning or signaling any actual process.
        from unittest.mock import patch
        with patch('hiercp_v1x.arm_process.os.getuid', create=True, return_value=owner['uid']):
            arm = OwnedArm(command, checkpoint=Path('UNIT_missing_checkpoint.pt'), grace_seconds=10,
                popen_factory=factory, identity_reader=lambda pid: identities[0],
                clock=lambda:time[0], emit=events.append)
        return arm, child, time, identities, events, calls

    def test_shell_false_and_fresh_child_session(self):
        arm, child, time, identities, events, calls = self.fixture()
        self.assertFalse(calls[0][1]['shell'])
        self.assertTrue(calls[0][1]['start_new_session'])
        self.assertEqual(calls[0][0], arm.command)

    def test_first_interrupt_exact_child_sigint_once(self):
        arm, child, time, identities, events, calls = self.fixture()
        arm.request_pause(); arm.request_pause(); arm.request_pause()
        self.assertEqual(child.signals, [signal.SIGINT])
        self.assertEqual(arm.pause_started, 100.)
        self.assertEqual(sum(row['stage']=='cooperative_pause_requested' for row in events), 1)

    def test_repeated_interrupt_does_not_reset_grace_deadline(self):
        arm, child, time, identities, events, calls = self.fixture()
        arm.request_pause(); time[0]=109.; arm.request_pause(); arm.check()
        self.assertEqual(child.terminations, 0)
        time[0]=110.; arm.check()
        self.assertEqual(child.terminations, 1)
        arm.check(); self.assertEqual(child.terminations, 1)
        self.assertFalse(arm.report(child.returncode)['parent_shell_or_SSH_signaled'])

    def test_cooperative_child_return_does_not_terminate(self):
        arm, child, time, identities, events, calls = self.fixture()
        arm.request_pause(); child.returncode=0; time[0]=999.; self.assertEqual(arm.check(), 0)
        self.assertEqual(child.terminations, 0)

    def test_pid_reuse_create_time_change_refuses_signal(self):
        arm, child, time, identities, events, calls = self.fixture()
        identities[0]=dict(identities[0], create_time=43.)
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            arm.request_pause()
        self.assertEqual(child.signals, [])

    def test_command_change_refuses_termination(self):
        arm, child, time, identities, events, calls = self.fixture()
        arm.request_pause(); identities[0]=dict(identities[0], argv=['UNIT_other_process'])
        time[0]=111.
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            arm.check()
        self.assertEqual(child.terminations, 0)

    def test_user_change_refuses_termination(self):
        arm, child, time, identities, events, calls = self.fixture()
        arm.request_pause(); identities[0]=dict(identities[0], uid=-999)
        time[0]=111.
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            arm.check()
        self.assertEqual(child.terminations, 0)

    def test_gone_child_is_not_signaled(self):
        arm, child, time, identities, events, calls = self.fixture()
        identities[0]=None; arm.request_pause(); time[0]=111.; arm.check()
        self.assertEqual(child.signals, [])
        self.assertEqual(child.terminations, 0)

    def test_no_checkpoint_does_not_prevent_owned_stop(self):
        arm, child, time, identities, events, calls = self.fixture()
        arm.request_pause(); time[0]=111.; arm.check()
        report=arm.report(child.returncode)
        self.assertEqual(report['status'], 'PAUSED_NO_OWN_CHECKPOINT')
        self.assertFalse(report['checkpoint_exists'])
        self.assertEqual(child.terminations, 1)

    def test_normal_return_is_not_full_training_claim(self):
        arm, child, time, identities, events, calls = self.fixture()
        child.returncode=0; report=arm.report(arm.check())
        self.assertEqual(report['status'], 'CHILD_RETURNED')
        self.assertFalse(report['paused'])
        self.assertFalse(report['full_training_claimed'])

    def test_argument_string_or_null_rejected_before_spawn(self):
        for command in ('python script.py', ['python', '\0bad']):
            with self.assertRaises(ValueError):
                OwnedArm(command, checkpoint=Path('UNIT_missing.pt'))

    def test_nonfinite_or_nonpositive_grace_rejected_before_spawn(self):
        for grace in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                OwnedArm(['python','UNIT_not_spawned.py'], checkpoint=Path('UNIT_missing.pt'),
                         grace_seconds=grace)

    def test_signal_callback_defers_inspection_and_io_to_poll_loop(self):
        from hiercp_v1x.arm_process import run_owned
        arm, child, clock, identities, events, calls = self.fixture()
        previous = object(); handlers = []; during_callback = [False]
        def register(sig, handler):
            self.assertEqual(sig, signal.SIGINT)
            handlers.append(handler)
        def trigger():
            during_callback[0] = True
            try:
                handlers[0](signal.SIGINT, None)
                handlers[0](signal.SIGINT, None)
            finally:
                during_callback[0] = False
        def construct(*args, **kwargs):
            trigger()  # Ctrl+C before the new child identity has been captured.
            self.assertIsNone(arm.pause_started)
            self.assertEqual(child.signals, [])
            return arm
        emit = arm.emit
        def checked_emit(row):
            self.assertFalse(during_callback[0], 'Signal handler performed output')
            emit(row)
        arm.emit = checked_emit
        identity = arm.identity_reader
        def checked_identity(pid):
            self.assertFalse(during_callback[0], 'Signal handler inspected a process')
            return identity(pid)
        arm.identity_reader = checked_identity
        def sleep(seconds):
            trigger()  # Repeated Ctrl+C does not reset the first pause deadline.
            clock[0] = 111.
        with patch('hiercp_v1x.arm_process.sys.platform', 'linux'), \
             patch('hiercp_v1x.arm_process.signal.getsignal', return_value=previous), \
             patch('hiercp_v1x.arm_process.signal.signal', side_effect=register), \
             patch('hiercp_v1x.arm_process.OwnedArm', side_effect=construct), \
             patch('hiercp_v1x.arm_process.time.sleep', side_effect=sleep):
            report = run_owned(arm.command, checkpoint=arm.checkpoint)
        self.assertEqual(child.signals, [signal.SIGINT])
        self.assertEqual(child.terminations, 1)
        self.assertTrue(report['paused'])
        self.assertIs(handlers[-1], previous)


class RealOwnedChildTests(unittest.TestCase):
    """Two tiny owned CPU children; no model, GPU, or existing process is used.

Windows lacks this runner's Linux SIGINT semantics. On Windows only the
cooperative signal callback is simulated; actual single-PID termination and
identity checks still run against the newly spawned local child.
    """
    def run_child(self, body, pause):
        import psutil
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory(prefix='UNIT_owned_arm_', dir=root) as directory:
            directory = Path(directory)
            self.assertTrue(directory.resolve().is_relative_to(root.resolve()))
            script = directory/'owned_cpu.py'; script.write_text(body, encoding='utf8')
            command = [sys.executable, '-B', str(script)]
            uid = psutil.Process().username() if os.name == 'nt' else os.getuid()
            def identity(pid):
                try:
                    process=psutil.Process(pid)
                    return dict(pid=pid, parent_pid=process.ppid(), create_time=process.create_time(),
                        argv=process.cmdline(), uid=process.username() if os.name=='nt' else process.uids().real)
                except psutil.NoSuchProcess:
                    return None
            def factory(argv, **kwargs):
                return subprocess.Popen(argv, **kwargs, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            with patch('hiercp_v1x.arm_process.os.getuid', create=True, return_value=uid):
                arm=OwnedArm(command, checkpoint=directory/'UNIT_uncreated_checkpoint.pt',
                    grace_seconds=.15, cwd=directory, popen_factory=factory,
                    identity_reader=identity, emit=lambda row:None)
            child=arm.child
            try:
                limit=time.monotonic()+5
                while not (directory/'ready').exists() and child.poll() is None and time.monotonic()<limit:
                    time.sleep(.01)
                if pause:
                    self.assertTrue((directory/'ready').exists(), 'Owned UNIT child never reached its gate')
                    if os.name=='nt':
                        with patch.object(child, 'send_signal', return_value=None):
                            arm.request_pause()
                    else:
                        arm.request_pause()
                result=None
                while result is None and time.monotonic()<limit:
                    result=arm.check(); time.sleep(.01)
                self.assertIsNotNone(result, 'Only the owned UNIT child failed to return within the test deadline')
                return arm.report(result)
            finally:
                if child.poll() is None and arm._verified_alive():
                    child.terminate(); child.wait(timeout=5)
                if child.stderr is not None:
                    child.stderr.close()

    def test_real_owned_normal_child_returns_without_stop(self):
        report=self.run_child("from pathlib import Path\nPath('ready').write_text('UNIT normal child')\n", False)
        self.assertEqual(report['child_returncode'], 0)
        self.assertFalse(report['paused'])
        self.assertFalse(report['forced_stop'])

    def test_real_owned_nonresponding_cpu_child_is_stopped(self):
        body=("import signal,time\nfrom pathlib import Path\n"
              "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
              "Path('ready').write_text('UNIT owned nonresponding child')\n"
              "while True:\n    time.sleep(.02)\n")
        report=self.run_child(body, True)
        self.assertTrue(report['paused'])
        self.assertTrue(report['forced_stop'])
        self.assertFalse(report['checkpoint_exists'])
        self.assertFalse(report['parent_shell_or_SSH_signaled'])


if __name__ == '__main__':
    unittest.main()
