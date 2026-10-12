"""DEBUG CPU launcher identity regression, without real server admission."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_v25_execution as execution


class ExecutionRuntimeTests(unittest.TestCase):
    def test_shared_child_sets_six_core_affinity_before_live_lease_constructor(self):
        from tools import run_v25 as frozen
        from hiercp_v1x import v25_direct_resources as resources
        from hiercp_v1x import v25_server_resources as owned
        from hiercp_v1x import v25_upper_memory_runtime as memory
        from hiercp_v1x import v25_independent_preparation as independent
        import psutil
        with tempfile.TemporaryDirectory(prefix='DEBUG_v25_execution_',dir=execution.ROOT/'tmp') as temporary:
            root=Path(temporary);resource=root/'allocation.json'
            resource.write_text(json.dumps({'format':resources.DIRECT_FORMAT}),encoding='utf8')
            args=SimpleNamespace(mode='prepare',arm='A',output=root/'A',
                allocation_request=resource,cpu_affinity=list(range(6)))
            affinity=list(range(24));events=[]
            def cpu_affinity(value=None):
                nonlocal affinity
                if value is not None:
                    affinity=list(value);events.append('six_cpu_affinity')
                return list(affinity)
            def constructor(path):
                self.assertEqual(affinity,args.cpu_affinity)
                events.append('live_lease_constructor');return SimpleNamespace()
            original=owned.AllocationLease
            try:
                with patch.object(frozen,'parse',return_value=args), \
                     patch.object(frozen,'publish'), \
                     patch.object(resources,'DirectAllocationLease',side_effect=constructor), \
                     patch.object(memory,'install_upper_memory_runtime',return_value={'DEBUG':True}), \
                     patch.object(independent,'prepare_shared',return_value='DEBUG complete') as prepare, \
                     patch.object(psutil,'Process',return_value=SimpleNamespace(cpu_affinity=cpu_affinity)):
                    self.assertEqual(execution.main(['--shared-preparation']),'DEBUG complete')
            finally:
                owned.AllocationLease=original
            self.assertEqual(events,['six_cpu_affinity','live_lease_constructor'])
            prepare.assert_called_once()


if __name__=='__main__':unittest.main()
