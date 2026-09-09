"""DEBUG telemetry lifecycle tests; no bank, medical data, or GPU workload."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from tools import online_bank_progress as progress


class BankProgressDebugTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="DEBUG_bank_progress_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.snapshot = mock.patch.object(progress, "snapshot", return_value={"DEBUG": True, "rss_bytes": 100}).start()
        self.addCleanup(mock.patch.stopall)

    def test_unique_reports_and_heartbeat_preserve_other_artifacts(self):
        protected = self.root / "existing-result.json"
        protected.write_bytes(b"DEBUG existing bytes")
        heard = threading.Event()
        def output(line, **kwargs):
            if '"event": "heartbeat"' in line:
                heard.set()
        with mock.patch("builtins.print", side_effect=output):
            with progress.BankProgress(self.root, heartbeat_seconds=0.01) as monitor:
                monitor.update("candidate_validation", case_id="DEBUG", required=128)
                monitor.counters(selected=3)
                self.assertTrue(heard.wait(2), "DEBUG heartbeat did not run")
            with progress.BankProgress(self.root) as second:
                second.update("local_graphs", case_id="DEBUG", candidates=128)
        self.assertNotEqual(monitor.path, second.path)
        self.assertFalse(monitor.thread.is_alive())
        self.assertTrue(monitor.handle.closed)
        rows = [json.loads(line) for line in monitor.path.read_text().splitlines()]
        self.assertEqual(rows[-1]["event"], "preparation_finished")
        self.assertTrue(any(row["event"] == "heartbeat" and row.get("selected") == 3 for row in rows))
        self.assertEqual(protected.read_bytes(), b"DEBUG existing bytes")

    def test_domain_error_is_not_swallowed_and_monitor_closes(self):
        error = ValueError("DEBUG domain failure")
        with mock.patch("builtins.print"), self.assertRaises(ValueError) as caught:
            with progress.BankProgress(self.root) as monitor:
                raise error
        self.assertIs(caught.exception, error)
        self.assertFalse(monitor.thread.is_alive())
        self.assertTrue(monitor.handle.closed)
        self.assertEqual(json.loads(monitor.path.read_text().splitlines()[-1])["event"], "failed")

    def test_monitor_error_propagates_instead_of_silent_success(self):
        attempted = threading.Event()
        def state():
            if threading.current_thread().name == "bank-progress":
                attempted.set()
                raise OSError("DEBUG unavailable resource snapshot")
            return {"DEBUG": True, "rss_bytes": 100}
        self.snapshot.side_effect = state
        with mock.patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "measurement failed"):
            with progress.BankProgress(self.root, heartbeat_seconds=0.01) as monitor:
                self.assertTrue(attempted.wait(2))
        self.assertFalse(monitor.thread.is_alive())
        self.assertTrue(monitor.handle.closed)

    def test_start_failure_closes_new_report_without_touching_old_files(self):
        self.snapshot.side_effect = OSError("DEBUG initial snapshot failure")
        monitor = progress.BankProgress(self.root)
        with self.assertRaises(OSError):
            monitor.__enter__()
        self.assertTrue(monitor.handle.closed)
        self.assertIsNone(monitor.thread)


if __name__ == "__main__":
    unittest.main()
