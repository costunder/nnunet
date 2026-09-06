"""DEBUG resource-accounting/failure-report tests; no medical data or training."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from hiercp import preparation_runtime as runtime


def state(*, available=100000, rss=10, cpu=0.):
    return {"rss_bytes": rss, "available_memory_bytes": available,
            "host_total_memory_bytes": 1000000, "cpu_capacity": 4,
            "cpu_affinity_cores": 4, "cpu_allocation_cores": 4.,
            "process_cpu_seconds": cpu, "cgroup": {"debug": True}}


def write_values(folder, values):
    folder.mkdir(parents=True)
    for name, value in values.items():
        (folder / name).write_text(str(value), encoding="ascii")


class FakeMeasurement:
    """Deterministic timing for scheduler-policy tests, explicitly not a benchmark."""
    def __enter__(self):
        self.before, self.peak_rss = state(), 15
        return self

    def __exit__(self, kind, error, traceback):
        self.report = {"elapsed_seconds": 1., "status": "failed" if error else "complete",
                       "sampled_peak_rss_bytes": self.peak_rss,
                       "error": str(error) if error else None}
        return False


class PreparationRuntimeDebugTests(unittest.TestCase):
    def test_debug_v1_memory_and_fractional_cpu_limits(self):
        with tempfile.TemporaryDirectory(prefix="hiercp_cgroup_debug_") as directory:
            memory, cpu = Path(directory) / "memory", Path(directory) / "cpu"
            write_values(memory, {"memory.limit_in_bytes": 1000, "memory.usage_in_bytes": 200})
            write_values(cpu, {"cpu.cfs_quota_us": 150000, "cpu.cfs_period_us": 100000})
            with mock.patch.object(runtime, "_cgroup_locations", return_value=(("memory", str(memory)), ("cpu", str(cpu)))):
                available, capacity, evidence = runtime._allocation_limits(5000, 8.)
            self.assertEqual(available, 800)
            self.assertEqual(capacity, 1.5)
            self.assertTrue(evidence["locations_resolved"])

    def test_debug_v2_ancestor_limits_bound_unlimited_leaf(self):
        with tempfile.TemporaryDirectory(prefix="hiercp_cgroup_debug_") as directory:
            parent, leaf = Path(directory) / "parent", Path(directory) / "parent" / "leaf"
            write_values(parent, {"memory.max": 1000, "memory.current": 600, "cpu.max": "150000 100000"})
            write_values(leaf, {"memory.max": "max", "memory.current": 100, "cpu.max": "200000 100000"})
            with mock.patch.object(runtime, "_cgroup_locations", return_value=(("v2", str(leaf)), ("v2", str(parent)))):
                available, capacity, _ = runtime._allocation_limits(5000, 8.)
            self.assertEqual((available, capacity), (400, 1.5))

    def test_debug_v1_unlimited_sentinel_does_not_invent_headroom(self):
        with tempfile.TemporaryDirectory(prefix="hiercp_cgroup_debug_") as directory:
            folder = Path(directory) / "memory"
            write_values(folder, {"memory.limit_in_bytes": 9223372036854771712, "memory.usage_in_bytes": 200})
            with mock.patch.object(runtime, "_cgroup_locations", return_value=(("memory", str(folder)),)):
                self.assertEqual(runtime._allocation_limits(500, 8.)[:2], (500, 8.))

    def test_debug_finite_cgroup_limit_without_usage_fails_explicitly(self):
        with tempfile.TemporaryDirectory(prefix="hiercp_cgroup_debug_") as directory:
            folder = Path(directory) / "memory"
            write_values(folder, {"memory.limit_in_bytes": 1000})
            with mock.patch.object(runtime, "_cgroup_locations", return_value=(("memory", str(folder)),)):
                with self.assertRaisesRegex(RuntimeError, "Incomplete cgroup memory"):
                    runtime._allocation_limits(5000, 8.)

    def test_debug_membership_and_mount_subtree_resolve_actual_leaf_and_ancestors(self):
        contents = {"/proc/self/cgroup": "0::/group/task\n",
                    "/proc/self/mountinfo": "31 21 0:27 /group /cg rw - cgroup2 cgroup rw\n"}
        runtime._cgroup_locations.cache_clear()
        try:
            with mock.patch.object(Path, "is_file", lambda path: path.as_posix() in contents), \
                 mock.patch.object(Path, "read_text", lambda path, **kwargs: contents[path.as_posix()]):
                values = runtime._cgroup_locations()
            self.assertEqual([(kind, Path(path).as_posix()) for kind, path in values],
                             [("v2", "/cg/task"), ("v2", "/cg")])
        finally:
            runtime._cgroup_locations.cache_clear()

    def test_debug_unresolvable_parent_namespace_membership_is_not_guessed(self):
        contents = {"/proc/self/cgroup": "0::/../../foreign\n", "/proc/self/mountinfo": ""}
        runtime._cgroup_locations.cache_clear()
        try:
            with mock.patch.object(Path, "is_file", lambda path: path.as_posix() in contents), \
                 mock.patch.object(Path, "read_text", lambda path, **kwargs: contents[path.as_posix()]):
                with self.assertRaisesRegex(RuntimeError, "cgroup namespace"):
                    runtime._cgroup_locations()
        finally:
            runtime._cgroup_locations.cache_clear()

    def test_debug_snapshot_preserves_fractional_allocation(self):
        process = SimpleNamespace(cpu_affinity=lambda: list(range(8)),
            cpu_times=lambda: SimpleNamespace(user=1., system=2.),
            memory_info=lambda: SimpleNamespace(rss=20))
        with mock.patch.object(runtime, "_process_for_pid", return_value=process), \
             mock.patch.object(runtime, "_allocation_limits", return_value=(80, .5, {})), \
             mock.patch("psutil.virtual_memory", return_value=SimpleNamespace(available=500, total=1000)), \
             mock.patch.dict(runtime.os.environ, {}, clear=True):
            result = runtime.snapshot()
        self.assertEqual(result["cpu_capacity"], 1)
        self.assertEqual(result["cpu_allocation_cores"], .5)

    def test_debug_initial_measurement_failure_still_has_failed_report(self):
        measurement = runtime.Measurement()
        with mock.patch.object(runtime, "snapshot", side_effect=OSError("initial snapshot")):
            with self.assertRaisesRegex(OSError, "initial snapshot"):
                with measurement:
                    self.fail("work must not start")
        self.assertEqual(measurement.report["status"], "failed")

    def test_debug_task_error_is_not_masked_by_final_snapshot_failure(self):
        measurement = runtime.Measurement()
        thread = SimpleNamespace(start=lambda: None, join=lambda: None)
        with mock.patch.object(runtime, "snapshot", side_effect=[state(), OSError("final snapshot")]), \
             mock.patch.object(runtime.threading, "Thread", return_value=thread):
            with self.assertRaisesRegex(ValueError, "domain error"):
                with measurement:
                    raise ValueError("domain error")
        self.assertEqual(measurement.report["status"], "failed")
        self.assertIn("domain error", measurement.report["error"])
        self.assertIn("final snapshot", measurement.report["final_snapshot_error"])
        self.assertIsNone(measurement.report["average_cpu_cores"])

    def test_debug_real_monitor_records_peak_and_its_own_cost(self):
        observed = threading.Event()
        def fake_snapshot():
            if threading.current_thread().name == "prepare-resource-monitor":
                observed.set()
                return state(available=30, rss=50)
            return state(available=100, rss=10)
        with mock.patch.object(runtime, "snapshot", side_effect=fake_snapshot):
            with runtime.Measurement() as measurement:
                self.assertTrue(observed.wait(2.))
        self.assertEqual(measurement.report["sampled_peak_rss_bytes"], 50)
        self.assertEqual(measurement.report["minimum_available_memory_bytes"], 30)
        self.assertGreaterEqual(measurement.report["monitor_samples"], 1)
        self.assertGreaterEqual(measurement.report["monitor_cpu_seconds"], 0.)
        self.assertGreaterEqual(measurement.report["monitor_wall_seconds"], 0.)

    def run_jobs(self, directory, *, tasks, function, workers="auto", available=100000):
        committed = []
        with mock.patch.object(runtime, "snapshot", return_value=state(available=available)), \
             mock.patch.object(runtime, "Measurement", FakeMeasurement):
            report = runtime.run_case_jobs(tasks=tasks, function=function, commit=committed.append,
                workers=workers, report_path=Path(directory) / "resource.json")
        return report, committed

    def test_debug_all_tasks_committed_once_with_explicit_wave_measurements(self):
        with tempfile.TemporaryDirectory(prefix="hiercp_runtime_debug_") as directory, \
             mock.patch.dict(runtime.os.environ, {}, clear=True):
            report, committed = self.run_jobs(directory, tasks=list(range(7)), function=lambda item: item)
        self.assertEqual(sorted(committed), list(range(7)))
        self.assertEqual(report["applied_worker_counts"], [1, 2, 4])
        self.assertEqual(report["completed_tasks"], 7)
        self.assertEqual(report["pending_tasks"], 0)
        self.assertIn("domain/cache validity", report["completion_semantics"])

    def test_debug_worker_failure_retains_other_commits_and_failed_wave(self):
        committed = []
        def work(item):
            if item == 0:
                raise ValueError("case failed")
            return item
        with tempfile.TemporaryDirectory(prefix="hiercp_runtime_debug_") as directory, \
             mock.patch.dict(runtime.os.environ, {}, clear=True), \
             mock.patch.object(runtime, "snapshot", return_value=state()), \
             mock.patch.object(runtime, "Measurement", FakeMeasurement):
            with self.assertRaisesRegex(ValueError, "case failed"):
                runtime.run_case_jobs(tasks=list(range(4)), function=work, commit=committed.append,
                    workers=2, report_path=Path(directory) / "resource.json")
            reports = list(Path(directory).glob("*.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
        self.assertEqual(committed, [1])
        self.assertEqual((report["attempted_tasks"], report["failed_tasks"], report["pending_tasks"]), (2, 1, 2))
        self.assertEqual(report["waves"][0]["status"], "failed")

    def test_debug_initial_scheduler_snapshot_failure_is_persisted(self):
        with tempfile.TemporaryDirectory(prefix="hiercp_runtime_debug_") as directory, \
             mock.patch.dict(runtime.os.environ, {}, clear=True), \
             mock.patch.object(runtime, "snapshot", side_effect=[OSError("before"), state()]):
            with self.assertRaisesRegex(OSError, "before"):
                runtime.run_case_jobs(tasks=[1], function=lambda item: item, commit=lambda item: None,
                    workers="auto", report_path=Path(directory) / "resource.json")
            report = json.loads(next(Path(directory).glob("*.json")).read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["attempted_tasks"], 0)

    def test_debug_final_scheduler_snapshot_failure_keeps_failed_report(self):
        with tempfile.TemporaryDirectory(prefix="hiercp_runtime_debug_") as directory, \
             mock.patch.dict(runtime.os.environ, {}, clear=True), \
             mock.patch.object(runtime, "snapshot", side_effect=[state(), OSError("after")]):
            with self.assertRaisesRegex(RuntimeError, "final resource snapshot"):
                runtime.run_case_jobs(tasks=[], function=lambda item: item, commit=lambda item: None,
                    workers="auto", report_path=Path(directory) / "resource.json")
            report = json.loads(next(Path(directory).glob("*.json")).read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("after", report["final_snapshot_error"])

    def test_debug_measured_pilot_peak_clamps_concurrency_without_skipping_work(self):
        with tempfile.TemporaryDirectory(prefix="hiercp_runtime_debug_") as directory, \
             mock.patch.dict(runtime.os.environ, {"HIERCP_PREPARE_MEASURED_CASE_RSS_BYTES": "60"}, clear=True):
            report, committed = self.run_jobs(directory, tasks=list(range(3)), function=lambda item: item, available=100)
        self.assertEqual(sorted(committed), [0, 1, 2])
        self.assertEqual(report["provided_measured_case_peak_rss_bytes"], 60)
        self.assertEqual(report["applied_worker_counts"], [1, 1, 1])

    def test_debug_pilot_peak_larger_than_first_wave_headroom_stops_before_work(self):
        work = mock.Mock()
        with tempfile.TemporaryDirectory(prefix="hiercp_runtime_debug_") as directory, \
             mock.patch.dict(runtime.os.environ, {"HIERCP_PREPARE_MEASURED_CASE_RSS_BYTES": "81"}, clear=True):
            with self.assertRaisesRegex(MemoryError, "headroom"):
                self.run_jobs(directory, tasks=[1], function=work, available=100)
        work.assert_not_called()

    def test_debug_explicit_workers_are_not_silently_reduced(self):
        work = mock.Mock()
        with tempfile.TemporaryDirectory(prefix="hiercp_runtime_debug_") as directory, \
             mock.patch.dict(runtime.os.environ, {"HIERCP_PREPARE_MEASURED_CASE_RSS_BYTES": "60"}, clear=True):
            with self.assertRaisesRegex(MemoryError, "Explicit preparation worker count"):
                self.run_jobs(directory, tasks=[1, 2], function=work, workers=2, available=100)
        work.assert_not_called()

    def test_debug_invalid_pilot_peak_is_an_explicit_error(self):
        for value in ("0", "-1", "nan", "1.5", ""):
            with self.subTest(value=value), tempfile.TemporaryDirectory(prefix="hiercp_runtime_debug_") as directory, \
                 mock.patch.dict(runtime.os.environ, {"HIERCP_PREPARE_MEASURED_CASE_RSS_BYTES": value}, clear=True):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.run_jobs(directory, tasks=[1], function=lambda item: item)


if __name__ == "__main__":
    unittest.main()
