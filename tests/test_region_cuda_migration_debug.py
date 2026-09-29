"""Explicit CUDA budget migration preserves all research and RAM contracts."""
import copy
import unittest
from unittest.mock import patch
from tests import test_region_execution_upgrade_debug as upgrade_tests
from l0_regions.execution_upgrade import verify_upgrade

class CudaMigrationChecks(unittest.TestCase):
    def identities(self):
        old,new=upgrade_tests.UpgradeChecks().identities()
        old['resource_limits']['rss_bytes']=192*2**30
        new['resource_limits']['rss_bytes']=192*2**30
        new['resource_limits']['cuda_bytes']=8*2**30
        return old,new

    def test_explicit_change_only_and_input_preserved(self):
        old,new=self.identities();snapshot=copy.deepcopy(old)
        with patch('l0_regions.execution_upgrade.blob_hash',side_effect=lambda rev,path:old['source']['runtime'][path]):
            self.assertTrue(verify_upgrade(old,new,allow_cuda_budget_change=True))
        self.assertEqual(old,snapshot)
        with self.assertRaisesRegex(ValueError,'changes dataset'):verify_upgrade(old,new)

    def test_rejects_other_limits_and_research_changes(self):
        old,new=self.identities()
        for key,value in [('candidates',[16]),('config',{}),('precision','FP16'),('cache_sha256','other')]:
            bad=copy.deepcopy(new);bad[key]=value
            with self.assertRaisesRegex(ValueError,'changes dataset'):
                verify_upgrade(old,bad,allow_cuda_budget_change=True)
        new['resource_limits']['rss_bytes']=64*2**30
        with self.assertRaisesRegex(ValueError,'other resource'):
            verify_upgrade(old,new,allow_cuda_budget_change=True)

    def test_bad_budget_and_fields_rejected(self):
        old,new=self.identities()
        for value in [0,-1,float('nan'),True,8.0]:
            new['resource_limits']['cuda_bytes']=value
            with self.assertRaisesRegex(ValueError,'Positive integer'):
                verify_upgrade(old,new,allow_cuda_budget_change=True)
        del new['resource_limits']['cuda_bytes']
        with self.assertRaisesRegex(ValueError,'matching resource'):
            verify_upgrade(old,new,allow_cuda_budget_change=True)

    def test_explicit_upgrade_still_required_before_read(self):
        from l0_regions.training import load_checkpoint
        with self.assertRaisesRegex(ValueError,'explicit execution upgrade'):
            load_checkpoint('does-not-exist.pt',{},allow_cuda_budget_change=True)

if __name__=='__main__':unittest.main()
