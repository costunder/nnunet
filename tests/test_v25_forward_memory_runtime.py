"""CPU UNIT: reclaim then real admission, without a training success claim."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import v25_forward_memory_runtime as runtime


class ForwardMemoryRuntimeUnit(unittest.TestCase):
    def fixture(self, rss, *, after=None, CUDA_overflow=False):
        readings = [rss]; events = []; active = object()
        budget = SimpleNamespace(rss_bytes=64*2**30, cuda_bytes=40*2**30)
        def original():
            events.append('original')
            if CUDA_overflow:
                raise MemoryError('original CUDA overflow')
            if readings[0] > budget.rss_bytes:
                raise MemoryError('original actual RSS overflow')
        budget.check = original
        def reclaim():
            events.append('reclaim')
            if after is not None:
                readings[0] = after
        inputs = SimpleNamespace(rss_bytes=budget.rss_bytes, _rss=reclaim, active=active)
        process = lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=readings[0]))
        return budget, inputs, events, process

    def test_overflow_reclaimed_before_original_actual_check(self):
        budget, inputs, events, process = self.fixture(65*2**30, after=49*2**30)
        active = inputs.active
        with patch.object(runtime.psutil, 'Process', side_effect=process):
            runtime.bind(budget, inputs); budget.check()
        self.assertEqual(events, ['reclaim', 'original'])
        self.assertIs(inputs.active, active)
        report = runtime.receipt(budget)
        self.assertEqual(report['profile']['last_reclamation']['after_RSS_bytes'], 49*2**30)
        self.assertEqual(report['RSS_limit_bytes'], 64*2**30)
        self.assertFalse(report['limit_relaxed'])

    def test_active_overflow_still_fails(self):
        budget, inputs, events, process = self.fixture(65*2**30)
        with patch.object(runtime.psutil, 'Process', side_effect=process):
            runtime.bind(budget, inputs)
            with self.assertRaisesRegex(MemoryError, 'original actual RSS overflow'):
                budget.check()
        self.assertEqual(events, ['reclaim', 'original'])
        self.assertEqual(runtime.receipt(budget)['profile']['strict_failures'], 1)

    def test_cuda_strict_check_is_retained_after_reclamation(self):
        budget, inputs, events, process = self.fixture(65*2**30, after=30*2**30, CUDA_overflow=True)
        with patch.object(runtime.psutil, 'Process', side_effect=process):
            runtime.bind(budget, inputs)
            with self.assertRaisesRegex(MemoryError, 'original CUDA overflow'):
                budget.check()
        self.assertEqual(events, ['reclaim', 'original'])

    def test_below_pressure_no_extra_gc_or_eviction(self):
        budget, inputs, events, process = self.fixture(60*2**30)
        with patch.object(runtime.psutil, 'Process', side_effect=process):
            runtime.bind(budget, inputs); budget.check()
        self.assertEqual(events, ['original'])

    def test_original_boundary_before_64GiB(self):
        budget, inputs, events, process = self.fixture(63*2**30+1, after=62*2**30)
        with patch.object(runtime.psutil, 'Process', side_effect=process):
            runtime.bind(budget, inputs); budget.check()
        self.assertEqual(events, ['reclaim', 'original'])

    def test_idempotent_binding_cannot_switch_inputs(self):
        budget, inputs, _, _ = self.fixture(20*2**30)
        one = runtime.bind(budget, inputs); two = runtime.bind(budget, inputs)
        self.assertEqual(one, two)
        with self.assertRaisesRegex(ValueError, 'different inputs'):
            runtime.bind(budget, SimpleNamespace(rss_bytes=budget.rss_bytes, _rss=lambda: None))

    def test_different_limit_or_missing_reclaimer_rejected(self):
        budget, inputs, _, _ = self.fixture(20*2**30)
        with self.assertRaisesRegex(ValueError, 'budgets'):
            runtime.bind(budget, SimpleNamespace(rss_bytes=65*2**30, _rss=lambda: None))
        with self.assertRaisesRegex(ValueError, 'reclamation'):
            runtime.bind(budget, SimpleNamespace(rss_bytes=64*2**30))

    def test_reclamation_failure_not_hidden(self):
        budget, inputs, events, process = self.fixture(65*2**30)
        def fail(): raise MemoryError('complete active input still too large')
        inputs._rss = fail
        with patch.object(runtime.psutil, 'Process', side_effect=process):
            runtime.bind(budget, inputs)
            with self.assertRaisesRegex(MemoryError, 'complete active input'):
                budget.check()
        self.assertEqual(events, [])

    def test_original_resource_budget_call_reclaims_and_remeasures(self):
        import torch
        from hiercp_v1x.historical_evaluation import ResourceBudget
        actual = ResourceBudget(cuda_bytes=40*2**30, rss_bytes=64*2**30)
        readings = [65*2**30]
        def reclaim(): readings[0] = 50*2**30
        inputs = SimpleNamespace(rss_bytes=64*2**30, _rss=reclaim)
        process = lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=readings[0]))
        with patch.object(runtime.psutil, 'Process', side_effect=process), \
                patch.object(torch.cuda, 'memory_allocated', return_value=0):
            runtime.bind(actual, inputs)
            actual()
        self.assertEqual(runtime.receipt(actual)['profile']['checks'], 1)
        self.assertEqual(readings[0], 50*2**30)

    def test_original_resource_budget_rejects_actual_cuda_overflow(self):
        import torch
        from hiercp_v1x.historical_evaluation import ResourceBudget
        actual = ResourceBudget(cuda_bytes=40*2**30, rss_bytes=64*2**30)
        inputs = SimpleNamespace(rss_bytes=64*2**30, _rss=lambda: None)
        process = lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=40*2**30))
        with patch.object(runtime.psutil, 'Process', side_effect=process), \
                patch.object(torch.cuda, 'memory_allocated', return_value=40*2**30+1):
            runtime.bind(actual, inputs)
            with self.assertRaisesRegex(MemoryError, 'Historical evaluation CUDA budget exceeded'):
                actual.check()


if __name__ == '__main__': unittest.main()
