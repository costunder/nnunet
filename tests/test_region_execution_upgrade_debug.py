"""Reject unreviewed changes while permitting explicit execution continuation."""
import copy
import unittest
from unittest.mock import patch
from l0_regions.execution_upgrade import compatible_core,verify_upgrade,MODEL,OLD_MODEL,REVIEWED_MODEL


class UpgradeChecks(unittest.TestCase):
    def identities(self):
        common=dict(cache_sha256='same',config={'epochs':40},base={'hidden':128},
            precision='FP32',candidates=[32,48,64],resource_limits={'cuda_bytes':40*2**30})
        old=dict(common,source=dict(core={MODEL:OLD_MODEL,'config/train.json':'same'},
            runtime={'l0_regions/training.py':'old','l0_sage/encoder.py':'same'}))
        new=copy.deepcopy(old);new['activation_storage']='retained'
        new['source']['core'][MODEL]=REVIEWED_MODEL
        new['source']['runtime']['l0_regions/training.py']='new'
        return old,new

    def test_only_exact_reviewed_core_change(self):
        old,new=self.identities()
        self.assertTrue(compatible_core(old['source']['core'],new['source']['core']))
        for field in (MODEL,'config/train.json'):
            bad=copy.deepcopy(new['source']['core']);bad[field]='unknown'
            self.assertFalse(compatible_core(old['source']['core'],bad))

    def test_pinned_runtime_required(self):
        old,new=self.identities()
        with patch('l0_regions.execution_upgrade.blob_hash',side_effect=lambda rev,path:old['source']['runtime'][path]):
            self.assertTrue(verify_upgrade(old,new))
        with patch('l0_regions.execution_upgrade.blob_hash',return_value='unreviewed'):
            with self.assertRaisesRegex(ValueError,'reviewed release'):verify_upgrade(old,new)

    def test_no_dataset_model_resource_or_batch_change(self):
        old,new=self.identities()
        for key in ('cache_sha256','config','base','precision','candidates','resource_limits'):
            bad=copy.deepcopy(new);bad[key]='changed'
            with self.assertRaisesRegex(ValueError,'changes dataset'):verify_upgrade(old,bad)

    def test_no_unreviewed_kernel_change(self):
        old,new=self.identities();new['source']['runtime']['l0_sage/encoder.py']='changed'
        with self.assertRaisesRegex(ValueError,'Unreviewed model'):verify_upgrade(old,new)

    def test_explicit_storage_target(self):
        old,new=self.identities();new['activation_storage']='checkpointed'
        with self.assertRaisesRegex(ValueError,'retained'):verify_upgrade(old,new)

    def test_preparation_additions_require_current_pinned_bytes(self):
        old,new=self.identities()
        helper='l0_regions/preparation_runtime.py'
        new['source']['runtime'][helper]='reviewed-helper'
        def blob(revision,path):
            return 'reviewed-helper' if path==helper else old['source']['runtime'][path]
        with patch('l0_regions.execution_upgrade.blob_hash',side_effect=blob):
            self.assertTrue(verify_upgrade(old,new))
            new['source']['runtime'][helper]='unreviewed-future-edit'
            with self.assertRaisesRegex(ValueError,'reviewed Git blob'):verify_upgrade(old,new)


if __name__=='__main__':unittest.main()
