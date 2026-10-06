"""Explicit case benchmark provenance must not become a verified-patient claim."""
import copy
import unittest
from hiercp_v222.contracts import validate_identities,CASE_BENCHMARK_IDENTITY


class CaseBenchmarkDebug(unittest.TestCase):
    def setUp(self):
        self.split=dict(inner_train=['A','B'],inner_val=['C'],outer_train=['A','B','C'],outer_val=['D'])
        self.identity=dict(format=CASE_BENCHMARK_IDENTITY,independence_scope='published_case_only',
            patient_independence_verified=False,annotation_scope='provided_masks_may_omit_lesions',
            cases={c:dict(patient_group='case:'+c,identity_basis='published_case_id_only',annotation_complete=None) for c in 'ABCD'})

    def test_truthful_case_scope_is_retained(self):
        self.assertEqual(validate_identities(self.identity,self.split),self.identity)

    def test_cannot_promote_to_verified_or_complete(self):
        for key,value in [('format','hiercp_patient_identity_v1'),('patient_independence_verified',True),
                          ('annotation_scope','complete')]:
            changed=copy.deepcopy(self.identity);changed[key]=value
            with self.assertRaises(ValueError):validate_identities(changed,self.split)
        changed=copy.deepcopy(self.identity);changed['cases']['A']['annotation_complete']=True
        with self.assertRaises(ValueError):validate_identities(changed,self.split)

    def test_wrong_case_group_or_missing_case_is_rejected(self):
        changed=copy.deepcopy(self.identity);changed['cases']['D']['patient_group']='case:A'
        with self.assertRaises(ValueError):validate_identities(changed,self.split)
        changed=copy.deepcopy(self.identity);del changed['cases']['C']
        with self.assertRaises(ValueError):validate_identities(changed,self.split)

    def test_verified_patient_contract_still_requires_complete_provenance(self):
        old=dict(format='hiercp_patient_identity_v1',cases={c:dict(patient_group=c,identity_basis='DEBUG',annotation_complete=True) for c in 'ABCD'})
        validate_identities(old,self.split)
        old['cases']['D']['patient_group']='A'
        with self.assertRaises(ValueError):validate_identities(old,self.split)


if __name__=='__main__':unittest.main()
