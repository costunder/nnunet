"""CPU DEBUG native CLI affinity admission; no server, CUDA or training run."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_v25_downstream as execution


class DownstreamExecutionTests(unittest.TestCase):
    @staticmethod
    def argv(cpus):
        return ['--arm','D','--gpu','4','--output','DEBUG_owned_D',
            '--score-overlay','DEBUG_own_BEST_overlay.json','--storage-admission','DEBUG_storage.json',
            '--project','DEBUG_project','--basic-validation','DEBUG_Basic26',
            '--cpu-affinity='+cpus,'--allocation-request','DEBUG_direct_D.json',
            '--evaluator-root','DEBUG_evaluator']

    def test_native_stage_applies_its_six_CPUs_before_request_or_live_lease(self):
        import psutil
        from hiercp_v1x import v25_downstream as downstream
        affinity=list(range(24));events=[]
        def cpu_affinity(value=None):
            nonlocal affinity
            if value is not None:
                affinity=list(value);events.append('apply_six_CPUs')
            return list(affinity)
        def run(**kwargs):
            self.assertEqual(affinity,list(range(6)))
            self.assertEqual(kwargs['cpu_affinity'],list(range(6)))
            events.append('native_request_and_lease');return {'DEBUG_complete':True}
        with patch.object(psutil,'Process',return_value=SimpleNamespace(cpu_affinity=cpu_affinity)), \
             patch.object(downstream,'run_downstream',side_effect=run), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(execution.main(self.argv('0,1,2,3,4,5')),0)
        self.assertEqual(events,['apply_six_CPUs','native_request_and_lease'])

    def test_foreign_or_duplicate_CPU_request_cannot_reach_native(self):
        import psutil
        from hiercp_v1x import v25_downstream as downstream
        for cpus in ('0,1,2,3,4,100','0,1,2,3,4,4','0,1,2','-1,0,1,2,3,4'):
            with self.subTest(cpus=cpus):
                applied=[]
                def cpu_affinity(value=None):
                    if value is not None:applied.append(value)
                    return list(range(24))
                with patch.object(psutil,'Process',return_value=SimpleNamespace(cpu_affinity=cpu_affinity)), \
                     patch.object(downstream,'run_downstream') as run:
                    with self.assertRaises(ValueError):execution.main(self.argv(cpus))
                self.assertFalse(applied);run.assert_not_called()

    def test_failed_actual_affinity_application_prevents_native_lease(self):
        import psutil
        from hiercp_v1x import v25_downstream as downstream
        with patch.object(psutil,'Process',return_value=SimpleNamespace(cpu_affinity=lambda value=None:list(range(24)))), \
             patch.object(downstream,'run_downstream') as run:
            with self.assertRaisesRegex(RuntimeError,'was not applied exactly'):
                execution.main(self.argv('0,1,2,3,4,5'))
        run.assert_not_called()


if __name__=='__main__':unittest.main()
