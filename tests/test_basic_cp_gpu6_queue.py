"""CPU-only completion gate tests; no claim of GPU evaluation completion."""
import unittest
from tools.queue_basic_cp_best_after_gpu6 import ready


class CompletionGateTests(unittest.TestCase):
    def setUp(self):
        self.complete = dict(status='COMPLETE', stage='complete', completed_epochs=250,
            evaluated_cases=26, full_native_evaluation=True, historical_metrics_complete=True)

    def test_complete_and_released(self):
        self.assertTrue(ready(self.complete, True, True))

    def test_each_completion_field_required(self):
        for key in self.complete:
            with self.subTest(key=key):
                incomplete = dict(self.complete)
                incomplete.pop(key)
                self.assertFalse(ready(incomplete, True, True))

    def test_live_controller_or_occupied_gpu_blocks_start(self):
        self.assertFalse(ready(self.complete, False, True))
        self.assertFalse(ready(self.complete, True, False))

    def test_failure_and_training_only_do_not_trigger_evaluation(self):
        self.assertFalse(ready(dict(self.complete, status='FAILED'), True, True))
        self.assertFalse(ready(dict(self.complete, evaluated_cases=0), True, True))
