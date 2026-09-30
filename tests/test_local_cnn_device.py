"""Synthetic NVIDIA inventory selection tests, not an actual MIG execution."""
import os,unittest
from unittest.mock import patch
from tools.local_cnn_device import resolve,select

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
        for visible in ('','6','MIG-test-other','MIG-test-a,MIG-test-b'):
            with self.assertRaisesRegex(ValueError,'No arbitrary'):resolve(7,LISTING,visible)
    def test_missing_physical_gpu_rejected(self):
        with self.assertRaisesRegex(ValueError,'not visible'):resolve(0,LISTING)
    def test_environment_is_set_for_child(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'6'}),patch('subprocess.check_output',return_value=LISTING):
            self.assertEqual(select(6),'MIG-test-current')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'MIG-test-current')

if __name__=='__main__':unittest.main()
