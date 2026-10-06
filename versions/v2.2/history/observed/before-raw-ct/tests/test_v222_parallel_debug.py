"""DEBUG scheduler tests; synthetic jobs are never medical training results."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from hiercp_v222.parallel import run_jobs


class ParallelDebug(unittest.TestCase):
    def test_real_overlap_exactly_once_and_continuous_refill(self):
        barrier = threading.Barrier(2)
        seen = []; lock = threading.Lock(); active = 0; peak = 0
        def work(i):
            nonlocal active, peak
            with lock: active += 1; peak = max(peak, active)
            if i < 2: barrier.wait(timeout=5)
            time.sleep(.03)
            with lock: active -= 1
            return i
        with tempfile.TemporaryDirectory() as d:
            result = run_jobs(range(7), work, seen.append, 2, Path(d)/'resources.json', memory_per_job=1024**2)
            self.assertEqual(sorted(seen), list(range(7)))
            self.assertEqual(peak, 2)
            self.assertEqual(result['completed'], 7)
            self.assertEqual(result['waves'][0]['workers'], 2)
            self.assertTrue((Path(d)/'resources.wave1.json').is_file())

    def test_auto_admission_and_complete_cohort(self):
        seen = []
        def work(i): time.sleep(.015); return i
        with tempfile.TemporaryDirectory() as d:
            result = run_jobs(range(9), work, seen.append, 'auto', Path(d)/'resources.json', memory_per_job=1024**2)
            self.assertEqual(sorted(seen), list(range(9)))
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(sum(w['tasks'] for w in result['waves']), 9)

    def test_failure_report_and_successful_outputs_preserved(self):
        seen = []
        def work(i):
            if i == 1: raise ValueError('DEBUG explicit failure')
            return i
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'resources.json'
            with self.assertRaisesRegex(ValueError, 'explicit failure'):
                run_jobs(range(6), work, seen.append, 2, path, memory_per_job=1024**2)
            result = json.loads(path.read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertNotIn(1, seen)
            self.assertEqual(result['completed'], len(seen))

    def test_unsafe_first_job_not_started(self):
        seen = []
        with tempfile.TemporaryDirectory() as d, patch('hiercp_v222.parallel.snapshot', return_value=dict(cpu_capacity=16, available_memory_bytes=100)):
            with self.assertRaises(MemoryError):
                run_jobs([1], seen.append, seen.append, 'auto', Path(d)/'resources.json', memory_per_job=100)
        self.assertEqual(seen, [])

    def test_explicit_unsafe_concurrency_not_silently_changed(self):
        with tempfile.TemporaryDirectory() as d, patch('hiercp_v222.parallel.snapshot', return_value=dict(cpu_capacity=1, available_memory_bytes=10000)):
            with self.assertRaises(MemoryError):
                run_jobs([1,2], lambda x:x, lambda x:None, 2, Path(d)/'resources.json', memory_per_job=100)


if __name__ == '__main__': unittest.main()
