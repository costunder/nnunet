"""Synthetic NVIDIA inventory selection tests, not an actual MIG execution."""
import os,unittest
from unittest.mock import patch
from tools.local_cnn_device import resolve,select,recorded_allocation

LISTING='''GPU 2: NVIDIA RTX A6000 (UUID: GPU-test-two)
GPU 6: NVIDIA A100-SXM4-80GB (UUID: GPU-test-six)
  MIG 1g.10gb Device 0: (UUID: MIG-test-current)
GPU 7: NVIDIA A100-SXM4-80GB (UUID: GPU-test-seven)
  MIG 1g.10gb Device 0: (UUID: MIG-test-a)
  MIG 1g.10gb Device 1: (UUID: MIG-test-b)
'''
class DeviceTests(unittest.TestCase):
    def test_full_gpu(self):self.assertEqual(resolve(2,LISTING),('GPU-test-two','GPU'))
    def test_single_mig(self):self.assertEqual(resolve(6,LISTING),('MIG-test-current','MIG'))
    def test_recreated_mig(self):self.assertEqual(resolve(6,LISTING.replace('MIG-test-current','MIG-test-new')),('MIG-test-new','MIG'))
    def test_multiple_uses_existing_allocation(self):self.assertEqual(resolve(7,LISTING,'MIG-test-b'),('MIG-test-b','MIG'))
    def test_multiple_without_unique_allocation_rejected(self):
        for visible in ('','6','MIG-test-other'):
            with self.assertRaisesRegex(ValueError,'No arbitrary'):resolve(7,LISTING,visible)
    def test_multiple_current_allocations_rejected(self):
        with self.assertRaisesRegex(ValueError,'multiple MIG'):resolve(7,LISTING,'MIG-test-a,MIG-test-b')
    def test_recorded_allocation_among_seven(self):
        allocation=dict(physical_index=6,gpu_uuid='GPU-six',mig_uuid='MIG-assigned')
        listing='GPU 6: NVIDIA A100 (UUID: GPU-six)\n'+''.join(
            f'  MIG 1g.10gb Device {i}: (UUID: {"MIG-assigned" if i==4 else f"MIG-other-{i}"})\n' for i in range(7))
        self.assertEqual(resolve(6,listing,'6',allocation),('MIG-assigned','MIG'))
        with self.assertRaisesRegex(ValueError,'no longer exists'):
            resolve(6,listing.replace('MIG-assigned','MIG-recreated'),'6',allocation)
        with self.assertRaisesRegex(ValueError,'physical GPU identity changed'):
            resolve(6,listing.replace('GPU-six','GPU-replaced'),'6',allocation)
    def test_current_allocation_precedes_old_record(self):
        allocation=dict(physical_index=7,gpu_uuid='GPU-test-seven',mig_uuid='MIG-removed')
        self.assertEqual(resolve(7,LISTING,'MIG-test-b',allocation),('MIG-test-b','MIG'))
        with self.assertRaisesRegex(ValueError,'multiple MIG'):
            resolve(7,LISTING,'MIG-test-a,MIG-test-b',allocation)
    def test_record_is_scoped_to_host_user_index(self):
        self.assertIsNotNone(recorded_allocation(6,'ece-agpu16.example','aicompetition06'))
        for index,host,user in ((5,'ece-agpu16','aicompetition06'),(6,'ece-a6gpu8','aicompetition06'),(6,'ece-agpu16','other')):
            self.assertIsNone(recorded_allocation(index,host,user))
    def test_select_with_recorded_server_allocation(self):
        row=recorded_allocation(6,'ece-agpu16','aicompetition06')
        listing=f'GPU 6: NVIDIA A100 (UUID: {row["gpu_uuid"]})\n'+''.join(
            f'  MIG 1g.10gb Device {i}: (UUID: {row["mig_uuid"] if i==0 else f"MIG-other-{i}"})\n' for i in range(7))
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'6'}),patch('subprocess.check_output',return_value=listing),patch('socket.gethostname',return_value='ece-agpu16'),patch('getpass.getuser',return_value='aicompetition06'):
            self.assertEqual(select(6),row['mig_uuid'])
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],row['mig_uuid'])
    def test_missing_physical_gpu_rejected(self):
        with self.assertRaisesRegex(ValueError,'not visible'):resolve(0,LISTING)
    def test_environment_is_set_for_child(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'6'}),patch('subprocess.check_output',return_value=LISTING):
            self.assertEqual(select(6),'MIG-test-current')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'MIG-test-current')

if __name__=='__main__':unittest.main()
