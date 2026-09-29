"""Preparation work reuse must preserve receipts, coverage and failure checks."""
import copy
import unittest
from unittest.mock import patch
from l0_regions.preparation import batch_bindings, binding, pair_audits
from tools.v22_artifacts import tree_hash


class PreparationDedup(unittest.TestCase):
    def test_batch_fingerprint_is_fresh_and_once(self):
        records=[dict(id=str(i),sha256='record',shared_source='source',center=[0,0,0],
                      donor_case_id='donor',donor_component=i) for i in range(3)]
        kwargs=dict(cache_sha256='cache',frozen_cnn_sha256='cnn',profile={'region_scales':1},
                    view_epoch=0,view_index=0,feature_evidence='checkpoint_partition_quality_unverified')
        with patch('hiercp_v222.v1_cache.provenance',return_value={}) as provenance, \
             patch('l0_regions.preparation.preparation_fingerprint',return_value='first') as digest:
            batch=batch_bindings(records,[0,1,2],**kwargs)
            self.assertEqual(digest.call_count,1)
            self.assertEqual(provenance.call_count,1)
            singles=[binding(record=r,dataset_index=i,**kwargs) for i,r in enumerate(records)]
            self.assertEqual(batch,singles)
            digest.return_value='changed'
            other=batch_bindings(records,[0,1,2],**kwargs)
            self.assertTrue(all(b['preparation_source_sha256']=='changed' for b in other))

    def test_incomplete_or_duplicate_binding_batch_rejected(self):
        for records,ids in (([],[]),([{}],[0,1]),([{},{}],[0,0])):
            with self.assertRaisesRegex(ValueError,'coverage'):batch_bindings(records,ids)

    def test_invalid_record_view_still_rejected(self):
        with patch('hiercp_v222.v1_cache.provenance',return_value={}), \
             patch('l0_regions.preparation.preparation_fingerprint',return_value='sha'):
            with self.assertRaisesRegex(ValueError,'view'):
                batch_bindings([{}],[0],cache_sha256='cache',frozen_cnn_sha256='cnn',profile={},
                    view_epoch=-1,view_index=0,feature_evidence='checkpoint_partition_quality_unverified')

    def test_partition_audit_retains_every_own_group_without_aliasing(self):
        groups=[dict(owner=i,shell=j,clusters=i+j+1,variance_excess={'clusters':j})
                for i in range(3) for j in range(3)]
        audit={'scale1':{'roles':{'target_context':{'group_diagnostics':groups}},'profile_exceeded':True},
               'scale2':{'status':'REMOVED_BY_DESIGN'}}
        before=copy.deepcopy(audit)
        pairs=pair_audits(audit,3)
        for i,pair in enumerate(pairs):
            self.assertEqual(pair['batch_audit_sha256'],tree_hash(audit))
            self.assertEqual(pair['scale1']['roles']['target_context']['group_diagnostics'],
                             [g for g in groups if g['owner']==i])
        pairs[0]['scale1']['roles']['target_context']['group_diagnostics'][0]['variance_excess']['clusters']=99
        self.assertEqual(audit,before)
        self.assertEqual(pairs[1]['scale1']['roles']['target_context']['group_diagnostics'][0]['variance_excess']['clusters'],0)

    def test_audit_owner_outside_batch_rejected(self):
        for owner in (-1,3,False,1.5):
            audit={'scale1':{'roles':{'target_context':{'group_diagnostics':[{'owner':owner}]}}}}
            with self.assertRaisesRegex(ValueError,'owner'):pair_audits(audit,3)


if __name__=='__main__':unittest.main(verbosity=2)
