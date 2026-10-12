"""CPU UNIT direct native lease tests; no CUDA/native training or data claim."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hiercp_v1x import v25_native_storage as storage
from hiercp_v1x import v25_parallel_native_storage as parallel
from hiercp_v1x import v24_readonly_native_storage as core
from tests import test_v25_native_storage as storage_unit


class V25ParallelNativeStorageCPUUnit(unittest.TestCase):
    setUp = storage_unit.V25StorageDEBUGUnit.setUp
    admission = storage_unit.V25StorageDEBUGUnit.admission

    def direct_allocation(self, arm):
        """CPU UNIT resource metadata, admitted only through a mocked live guard."""
        index = storage.ARMS.index(arm)
        value = dict(format='v25_explicit_direct_GPU_execution_v1', arm=arm,
            CPU_UNIT_SYNTHETIC_METADATA=True, group_root=str(self.group),
            gpu=(1,4,5,6)[index], GPU_UUID='GPU-UNIT-'+arm,
            CPU_affinity=list(range(index*6, index*6+6)), UID=41833,
            PBS_owned_reservation=False, explicit_direct_execution_authorized=True,
            aggregate_native_and_cache_RSS_bytes=416*2**30, canonical_cache_reserve_bytes=32*2**30)
        path = self.group/('UNIT_direct_'+arm+'.json')
        core.publish_json(path, value)
        return path

    def test_distinct_direct_arm_waves_overlap_without_serializing_training(self):
        from contextlib import ExitStack
        from hiercp_v1x import v25_direct_resources as direct
        document = self.admission()
        paths = {arm:self.direct_allocation(arm) for arm in storage.ARMS}
        live_guard = SimpleNamespace(verify=lambda **kwargs:None)
        with patch.object(direct, 'DirectAllocationLease', return_value=live_guard), ExitStack() as stack:
            for arm in storage.ARMS:
                stack.enter_context(parallel.parallel_wave_lock(document, arm,
                    request_sha256='a'*64, allocation_request=paths[arm]))
            leases = list(self.group.glob('parallel_native_arm_*.lease.json'))
            self.assertEqual(len(leases), 4)
            self.assertEqual({core.read(path)['arm'] for path in leases}, set(storage.ARMS))
            self.assertFalse((self.group/'sequential_native_wave.lease.json').exists())
            with self.assertRaisesRegex(FileExistsError, 'explicit same-arm'):
                with parallel.parallel_wave_lock(document, 'A', allocation_request=paths['A']): pass
        self.assertEqual(list(self.group.glob('parallel_native_arm_*.lease.json')), [])
        self.assertEqual(len(list(self.group.glob('parallel_native_arm_*_finished_*.json'))), 4)
        self.assertEqual(document['checkpoint_coordination']['global_peak_checkpoint_slots'], 9)

    def test_direct_wave_rejects_foreign_arm_insufficient_aggregate_and_exclusive_overlap(self):
        from hiercp_v1x import v25_direct_resources as direct
        document = self.admission(); path = self.direct_allocation('A')
        live_guard = SimpleNamespace(verify=lambda **kwargs:None)
        with patch.object(direct, 'DirectAllocationLease', return_value=live_guard):
            with self.assertRaisesRegex(ValueError, 'own-arm'):
                with parallel.parallel_wave_lock(document, 'B', allocation_request=path): pass
            original = core.read(path)
            path.write_text(json.dumps(dict(original,aggregate_native_and_cache_RSS_bytes=128*2**30)))
            with self.assertRaisesRegex(ValueError, 'four-native'):
                with parallel.parallel_wave_lock(document, 'A', allocation_request=path): pass
            path.write_text(json.dumps(original))
            with parallel.parallel_wave_lock(document, 'A', allocation_request=path):
                with self.assertRaisesRegex(RuntimeError, 'cannot overlap'):
                    with parallel.native_wave_lock(document, 'B'): pass
            with parallel.native_wave_lock(document, 'A'):
                with self.assertRaisesRegex(RuntimeError, 'cannot overlap'):
                    with parallel.parallel_wave_lock(document, 'A', allocation_request=path): pass

    def test_direct_peer_wave_rejects_shared_gpu_and_cpu_and_preserves_active_lease(self):
        from hiercp_v1x import v25_direct_resources as direct
        document = self.admission(); a = self.direct_allocation('A'); b = self.direct_allocation('B')
        live_guard = SimpleNamespace(verify=lambda **kwargs:None)
        with patch.object(direct, 'DirectAllocationLease', return_value=live_guard):
            with parallel.parallel_wave_lock(document, 'A', allocation_request=a):
                lease = self.group/'parallel_native_arm_A.lease.json'; before = lease.read_bytes()
                original = core.read(b)
                for changed in (dict(original,GPU_UUID='GPU-UNIT-A'),dict(original,gpu=1),
                        dict(original,CPU_affinity=list(range(6)))):
                    b.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, 'distinct GPUs'):
                        with parallel.parallel_wave_lock(document, 'B', allocation_request=b): pass
                    self.assertEqual(lease.read_bytes(), before)
                self.assertFalse((self.group/'parallel_native_arm_B.lease.json').exists())

    def test_failed_direct_arm_keeps_lease_and_does_not_block_distinct_arm(self):
        from hiercp_v1x import v25_direct_resources as direct
        document = self.admission(); a = self.direct_allocation('A'); b = self.direct_allocation('B')
        live_guard = SimpleNamespace(verify=lambda **kwargs:None)
        with patch.object(direct, 'DirectAllocationLease', return_value=live_guard):
            with self.assertRaisesRegex(RuntimeError, 'UNIT original failure'):
                with parallel.parallel_wave_lock(document, 'A', allocation_request=a):
                    raise RuntimeError('UNIT original failure')
            lease = self.group/'parallel_native_arm_A.lease.json'; before = lease.read_bytes()
            with parallel.parallel_wave_lock(document, 'B', allocation_request=b): pass
            self.assertEqual(lease.read_bytes(), before)
            self.assertFalse((self.group/'parallel_native_arm_B.lease.json').exists())

    def test_direct_same_arm_recovery_requires_released_exact_parent_and_child(self):
        from hiercp_v1x import v25_direct_resources as direct
        document = self.admission(); resource = self.direct_allocation('A')
        value = core.read(resource)
        old = dict(format='v25_distinct_arm_downstream_wave_v1', arm='A',pid=777,create_time=123.,
            command=['DEBUG_CPU_parent'],UID=storage.os.getuid() if hasattr(storage.os,'getuid') else None,
            admission_sha256=storage.canonical_hash(document),arm_root=document['arm_roots']['A'],
            request_sha256='a'*64,direct_resource_proof=core.proof(resource),gpu=value['gpu'],
            GPU_UUID=value['GPU_UUID'],CPU_affinity=value['CPU_affinity'],
            aggregate_native_and_cache_RSS_bytes=416*2**30,PBS_owned_reservation=False)
        path=self.group/'parallel_native_arm_A.lease.json';core.publish_json(path,old);before=path.read_bytes()
        child=dict(pid=778,create_time=124.,command=['DEBUG_CPU_child'],UID=old['UID'],
            parent_actor={key:old[key] for key in ('pid','create_time','command','UID')},
            state='RUNNING',request_sha256='a'*64)
        child_path=self.arms['A']/'UNIT_child.json';core.publish_json(child_path,child)
        live_guard=SimpleNamespace(verify=lambda **kwargs:None)
        with patch.object(direct,'DirectAllocationLease',return_value=live_guard):
            for releases in ((False,), (True,False)):
                with patch.object(storage,'_released',side_effect=releases),self.assertRaisesRegex(RuntimeError,'alive'):
                    with parallel.parallel_wave_lock(document,'A',allocation_request=resource,
                            recover_released_same_arm=True,predecessor_child=child_path):pass
                self.assertEqual(path.read_bytes(),before)
            with patch.object(storage,'_released',side_effect=(True,True)):
                with parallel.parallel_wave_lock(document,'A',allocation_request=resource,request_sha256='b'*64,
                        recover_released_same_arm=True,predecessor_child=child_path):
                    self.assertEqual(core.read(path)['request_sha256'],'b'*64)
        self.assertFalse(path.exists())
        archives=list(self.group.glob('parallel_native_arm_A_released_*.json'))
        self.assertEqual(len(archives),1)
        archived=core.read(archives[0]);self.assertEqual(archived['original_lease'],old)
        self.assertFalse(archived['signals_sent'])

    def test_owned_mode_reuses_original_contextmanager_body_and_one_exclusive_lease(self):
        document=self.admission()
        with parallel.native_wave_lock(document,'A'):
            self.assertTrue((self.group/'sequential_native_wave.lease.json').exists())
            self.assertEqual(list(self.group.glob('parallel_native_arm_*.lease.json')),[])
            self.assertEqual(parallel._OWNED_WAVE.__wrapped__.__code__,storage.sequential_wave_lock.__wrapped__.__code__)
            with self.assertRaises(FileExistsError):
                with parallel.native_wave_lock(document,'B'):pass
        self.assertFalse((self.group/'sequential_native_wave.lease.json').exists())


if __name__ == '__main__':
    unittest.main()
