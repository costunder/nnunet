"""Actual full-cohort equivalence for code-only deployment, not training evidence."""
import os
from pathlib import Path
import unittest
from tools.v1_server import observation_metadata
from hiercp_v222.v1_cache import configuration,assign_pairs
from hiercp_v222.contracts import read_json,ROOT


class RawRebuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rebuilt=Path(os.environ.get('HIERCP_TEST_REBUILT_INDEX',ROOT/'work/v222_server_raw_rebuild_test_20260924/index.json'))
        reference=Path(os.environ.get('HIERCP_TEST_OBSERVATION_INDEX',ROOT/'work/v222_raw_ct_r3_training_20260923/context/index_vram_affine.json'))
        if not rebuilt.is_file() or not reference.is_file():
            raise unittest.SkipTest('Both actual rebuilt and historical observation metadata required')
        cls.new=read_json(rebuilt);cls.old=read_json(reference);cls.cfg,_=configuration()

    def test_all_centers_targets_and_donor_assignments_identical(self):
        keys=('id','case_id','patient_group','component','center','target')
        project=lambda meta:[{k:r[k] for k in keys} for r in sorted(meta['records'],key=lambda r:r['id'])]
        self.assertEqual(project(self.new),project(self.old))
        self.assertEqual(self.new['split'],self.old['split'])
        self.assertEqual(self.new['donor_pool'],self.old['donor_pool'])
        self.assertEqual(assign_pairs(self.new,self.cfg),assign_pairs(self.old,self.cfg))

    def test_missing_case_fails_before_producing_metadata(self):
        inventory={r['case_id']:r for r in self.new['raw_records']}
        with self.assertRaisesRegex(ValueError,'Complete raw cohort'):
            observation_metadata(inventory,self.new['split'],self.new['identities'],self.cfg)


if __name__=='__main__':unittest.main()
