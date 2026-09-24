"""Training contracts using existing full metadata; no synthetic medical inputs."""
import unittest
import os
from pathlib import Path
from copy import deepcopy
from hiercp_v222.v1_cache import configuration,assign_pairs,read_json,ROOT
from hiercp_v222.v1_training import groups

class PairedContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,_=configuration()
        path=Path(os.environ.get('HIERCP_TEST_OBSERVATION_INDEX',ROOT/'work/v222_raw_ct_r3_training_20260923/context/index_vram_affine.json'))
        if not path.is_file():raise unittest.SkipTest('Actual observation index required: set HIERCP_TEST_OBSERVATION_INDEX')
        cls.meta=read_json(path)
        cls.rows=assign_pairs(cls.meta,cls.cfg)
    def test_all_observations_retained(self):
        self.assertEqual({r['id'] for r in self.rows},{r['id'] for r in self.meta['records']})
        self.assertEqual(len(self.rows),14102)
    def test_donor_independence(self):
        for r in self.rows:
            self.assertIn(r['donor_case_id'],self.meta['split']['inner_train'])
            self.assertNotEqual(r['donor_group'],r['patient_group'])
    def test_assignment_does_not_read_label(self):
        # Pair RNG input uses row ID only; a class-preserving permutation of
        # observation labels keeps per-case positive/negative totals unchanged.
        changed=deepcopy(self.meta)
        for c in changed['split']['outer_train']:
            rows=[r for r in changed['records'] if r['case_id']==c]
            values=[r['target'] for r in rows][::-1]
            for r,v in zip(rows,values):r['target']=v
        other=assign_pairs(changed,self.cfg)
        self.assertEqual([(r['donor_case_id'],r['donor_component']) for r in self.rows],
                         [(r['donor_case_id'],r['donor_component']) for r in other])
    def test_buckets_complete_without_mixed_query_groups(self):
        class Dataset:pass
        d=Dataset();d.rows=[r|dict(bounds=dict(edges=i)) for i,r in enumerate(self.rows)]
        batches=list(groups(d,32,42,3));flat=[i for b in batches for i in b]
        self.assertEqual(sorted(flat),list(range(len(d.rows))))
        self.assertEqual(len(flat),len(set(flat)))
        self.assertTrue(all(len({d.rows[i]['patient_group'] for i in b})==1 for b in batches))
    def test_outer_validation_donor_rejected(self):
        changed=deepcopy(self.meta)
        changed['donor_pool'][0]['case_id']=changed['split']['outer_val'][0]
        with self.assertRaises(ValueError):assign_pairs(changed,self.cfg)

if __name__=='__main__':unittest.main()
