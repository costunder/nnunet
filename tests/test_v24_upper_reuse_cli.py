"""CPU UNIT: complete native receipt serialization and rejection contracts."""
import copy
import json
import unittest

from tools.run_v24_upper_reuse import _require_verification


class UpperReceiptUnit(unittest.TestCase):
    def setUp(self):
        self.identity=dict(case_ids=tuple('case_'+str(i) for i in range(86)),
            input_binding_file_sha256='bound-native-input',CPU_affinity=[4,5,6,7],
            stages=[7,23,39,55,71,87,103,119,128],runtime={'source':'verified'})
        self.proof=json.loads(json.dumps(dict(
            status='FULL_NATIVE_EXISTING_GRAPH_EQUIVALENCE_PASS',identity=self.identity,
            stages=[dict(active_U=7,compared_cases=86)])))

    def test_actual_native_tuple_survives_json_receipt(self):
        _require_verification(self.proof,self.identity)

    def test_changed_values_or_case_order_are_rejected(self):
        for key,value in (('input_binding_file_sha256','different'),
                          ('CPU_affinity',[8,9,10,11]),
                          ('case_ids',tuple(reversed(self.identity['case_ids']))),
                          ('runtime',{'source':'different'})):
            changed=copy.deepcopy(self.identity);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                _require_verification(self.proof,changed)

    def test_missing_failed_or_incomplete_full_population_are_rejected(self):
        for patch in ({'status':'FAILED'},{'stages':[]},
                      {'stages':[dict(active_U=7,compared_cases=85)]},
                      {'stages':[dict(active_U=23,compared_cases=86)]}):
            changed=copy.deepcopy(self.proof);changed.update(patch)
            with self.subTest(patch=patch),self.assertRaises(ValueError):
                _require_verification(changed,self.identity)


if __name__=='__main__':
    unittest.main()
