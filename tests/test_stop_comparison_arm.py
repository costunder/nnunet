"""UNIT exact-process mocks only; no process is started, signaled or stopped."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import psutil

from tools import run_comparison_arm as launcher
from tools import stop_comparison_arm as stop


class StopComparisonArmTests(unittest.TestCase):
    def directory(self):
        return TemporaryDirectory(prefix="UNIT_stop_comparison_", dir=Path(__file__).resolve().parents[1])

    def fixture(self, path, *, arm="native_fixed", pid=713):
        root = Path(path) / "UNIT_existing_experiment"; root.mkdir()
        output = root / arm; output.mkdir()
        checkpoint = output / "checkpoint_latest.pt"
        checkpoint.write_bytes(b"UNIT opaque existing saved progress; not neural evidence")
        lock = root / ".pipeline.lock"; lock.write_text('{"UNIT":"unchanged existing lock"}')
        command = ["python", "/UNIT/tools/run_v19_comparison.py", "--arm", arm, "--experiment", str(root)]
        owner = dict(status="ALREADY_RUNNING", arm=arm, pid=pid, create_time=123.5,
            command=command, experiment=str(root), checkpoint=str(checkpoint), duplicate_started=False)
        process = SimpleNamespace(pid=pid, create_time=Mock(return_value=123.5),
            cmdline=Mock(return_value=command), terminate=Mock(), wait=Mock(return_value=0),
            kill=Mock(side_effect=AssertionError("No forced kill")),
            send_signal=Mock(side_effect=AssertionError("No extra signals")))
        parent = SimpleNamespace(pid=71, terminate=Mock(side_effect=AssertionError("No parent signal")))
        current = SimpleNamespace(parents=Mock(return_value=[parent]))
        def process_factory(target=None):
            return current if target is None else process
        arguments = SimpleNamespace(arm=arm, experiment=root)
        before = (checkpoint.read_bytes(), lock.read_bytes())
        return arguments, owner, process, parent, process_factory, checkpoint, lock, before

    def assert_preserved(self, checkpoint, lock, before):
        self.assertEqual((checkpoint.read_bytes(), lock.read_bytes()), before)

    def test_exact_requested_job_is_reported_before_one_terminate_and_wait(self):
        with self.directory() as directory:
            args, owner, process, parent, factory, checkpoint, lock, before = self.fixture(directory)
            output = StringIO()
            def terminate():
                printed = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(printed[-1]["status"], "STOP_REQUESTED")
                self.assertEqual(printed[-1]["pid"], process.pid)
                self.assertEqual(printed[-1]["command"], owner["command"])
                self.assertFalse(printed[-1]["parent_shell_or_SSH_signaled"])
            process.terminate.side_effect = terminate
            with patch.object(launcher, "_active_owner", side_effect=[owner, owner]) as active, \
                    patch.object(psutil, "Process", side_effect=factory), redirect_stdout(output):
                result = stop.run(args)
            self.assertEqual(result["status"], "STOPPED_LAST_SAVED")
            process.terminate.assert_called_once_with()
            process.wait.assert_called_once_with(timeout=10)
            self.assertEqual(active.call_count, 2)
            process.kill.assert_not_called(); process.send_signal.assert_not_called(); parent.terminate.assert_not_called()
            self.assert_preserved(checkpoint, lock, before)

    def test_current_process_or_ancestor_is_never_signaled(self):
        for pid in (os.getpid(), 71):
            with self.subTest(pid=pid), self.directory() as directory:
                args, owner, process, parent, factory, checkpoint, lock, before = self.fixture(directory, pid=pid)
                with patch.object(launcher, "_active_owner", return_value=owner), \
                        patch.object(psutil, "Process", side_effect=factory):
                    with self.assertRaisesRegex(RuntimeError, "terminal or an ancestor"):
                        stop.run(args)
                process.terminate.assert_not_called(); parent.terminate.assert_not_called()
                self.assert_preserved(checkpoint, lock, before)

    def test_changed_process_creation_time_or_argv_is_never_signaled(self):
        for difference in ("create_time", "argv"):
            with self.subTest(difference=difference), self.directory() as directory:
                args, owner, process, parent, factory, checkpoint, lock, before = self.fixture(directory)
                if difference == "create_time": process.create_time.return_value = 124.5
                else: process.cmdline.return_value = ["python", "/UNIT/unrelated.py"]
                with patch.object(launcher, "_active_owner", return_value=owner), \
                        patch.object(psutil, "Process", side_effect=factory):
                    with self.assertRaisesRegex(RuntimeError, "job changed"):
                        stop.run(args)
                process.terminate.assert_not_called(); process.wait.assert_not_called(); parent.terminate.assert_not_called()
                self.assert_preserved(checkpoint, lock, before)

    def test_changed_or_missing_lock_owner_on_final_recheck_is_never_signaled(self):
        for next_owner in (None, dict(pid=714, create_time=124.5, command=["UNIT_other_process"])):
            with self.subTest(next_owner=next_owner), self.directory() as directory:
                args, owner, process, parent, factory, checkpoint, lock, before = self.fixture(directory)
                with patch.object(launcher, "_active_owner", side_effect=[owner, next_owner]), \
                        patch.object(psutil, "Process", side_effect=factory), redirect_stdout(StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "ownership changed"):
                        stop.run(args)
                process.terminate.assert_not_called(); process.wait.assert_not_called(); parent.terminate.assert_not_called()
                self.assert_preserved(checkpoint, lock, before)

    def test_inactive_job_returns_without_signal_lock_cleanup_or_checkpoint_changes(self):
        with self.directory() as directory:
            args, _, process, _, _, checkpoint, lock, before = self.fixture(directory)
            with patch.object(launcher, "_active_owner", return_value=None), \
                    patch.object(psutil, "Process", side_effect=AssertionError("No inactive process access")), \
                    redirect_stdout(StringIO()):
                result = stop.run(args)
            self.assertEqual(result["status"], "NOT_RUNNING")
            self.assertTrue(result["lock_and_outputs_preserved"])
            process.terminate.assert_not_called()
            self.assert_preserved(checkpoint, lock, before)

    def test_prior_host_user_arm_root_or_executable_rejection_never_signals(self):
        with self.directory() as directory:
            args, _, process, _, _, checkpoint, lock, before = self.fixture(directory)
            with patch.object(launcher, "_active_owner", side_effect=RuntimeError("UNIT unverified owner")), \
                    patch.object(psutil, "Process", side_effect=AssertionError("No unverified process access")):
                with self.assertRaisesRegex(RuntimeError, "unverified owner"):
                    stop.run(args)
            process.terminate.assert_not_called()
            self.assert_preserved(checkpoint, lock, before)

    def test_timeout_sends_no_kill_group_signal_retry_or_output_changes(self):
        with self.directory() as directory:
            args, owner, process, parent, factory, checkpoint, lock, before = self.fixture(directory)
            process.wait.side_effect = psutil.TimeoutExpired(seconds=10, pid=process.pid)
            output = StringIO()
            with patch.object(launcher, "_active_owner", side_effect=[owner, owner]), \
                    patch.object(psutil, "Process", side_effect=factory), \
                    patch.object(stop.os, "kill", side_effect=AssertionError("No broad or direct OS signal")) as os_signal, \
                    redirect_stdout(output):
                result = stop.run(args)
            self.assertEqual(result["status"], "TERMINATION_PENDING")
            process.terminate.assert_called_once_with(); process.wait.assert_called_once_with(timeout=10)
            process.kill.assert_not_called(); process.send_signal.assert_not_called(); parent.terminate.assert_not_called(); os_signal.assert_not_called()
            self.assertIn("no additional force or group signal", output.getvalue())
            self.assert_preserved(checkpoint, lock, before)

    def test_process_disappearing_after_verification_is_not_replaced_or_signaled(self):
        with self.directory() as directory:
            args, owner, process, _, _, checkpoint, lock, before = self.fixture(directory)
            with patch.object(launcher, "_active_owner", return_value=owner), \
                    patch.object(psutil, "Process", side_effect=psutil.NoSuchProcess(owner["pid"])):
                with self.assertRaises(psutil.NoSuchProcess):
                    stop.run(args)
            process.terminate.assert_not_called()
            self.assert_preserved(checkpoint, lock, before)

    def test_stop_requires_explicit_single_arm_and_experiment(self):
        for argv in ([], ["--arm", "native_fixed"], ["--experiment", "/UNIT/path"],
                     ["--arm", "all", "--experiment", "/UNIT/path"],
                     ["--arm", "both", "--experiment", "/UNIT/path"]):
            with self.subTest(argv=argv), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                stop.parse(argv)


if __name__ == "__main__":
    unittest.main()
