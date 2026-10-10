"""DEBUG unit checks for immutable CP payloads and exact admission rewrites."""
import ast
import unittest
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage as ndi
from tools.v23_matched_geometry import distance_at_voxel
from tools.run_v23_matched_cp import compare_payload,clone_with_replacements

def admission(recipient,donor):
    if recipient==donor:raise ValueError('different patient required')
    if not recipient or not donor:raise ValueError('actual identity required')
    return recipient,donor

class MatchedCPDebug(unittest.TestCase):
    def test_point_distance_matches_full_edt_exactly(self):
        rng=np.random.default_rng(42)
        with ThreadPoolExecutor(max_workers=4)as pool:
            for spacing in ([1.,1.,1.],[.73,.91,2.5]):
                mask=rng.random((17,19,13))<.12
                reference=ndi.distance_transform_edt(~mask,sampling=np.asarray(spacing,np.float32)).astype(np.float32)
                for center in [(0,0,0),(16,18,12),*(tuple(map(int,row))for row in rng.integers([0,0,0],[17,19,13],size=(32,3)))]:
                    self.assertEqual(distance_at_voxel(mask,spacing,center,pool),reference[center])
            self.assertTrue(np.isposinf(distance_at_voxel(np.zeros((3,3,3),bool),[1,1,1],[1,1,1],pool)))
    def test_scores_only(self):
        original=dict(scores=np.arange(128,dtype=np.float32),candidate_centers=np.arange(384,dtype=np.int32).reshape(128,3),source_mask=np.ones((2,3,4),dtype=np.uint8))
        updated={k:v.copy()for k,v in original.items()};updated['scores']*=-1
        compare_payload(original,updated)
        updated['candidate_centers'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'payload changed'):compare_payload(original,updated)
    def test_incomplete_and_nonfinite_rejected(self):
        original=dict(scores=np.arange(128,dtype=np.float32))
        with self.assertRaises(ValueError):compare_payload(original,dict(scores=np.arange(127,dtype=np.float32)))
        bad=original['scores'].copy();bad[0]=np.nan
        with self.assertRaises(ValueError):compare_payload(original,dict(scores=bad))
    def test_admission_only_changes_explicit_condition(self):
        adapted=clone_with_replacements(admission,{'recipient == donor':'False'})
        self.assertEqual(adapted('patient','patient'),('patient','patient'))
        with self.assertRaises(ValueError):adapted('','')
        with self.assertRaises(ValueError):admission('patient','patient')
        with self.assertRaises(ValueError):clone_with_replacements(admission,{'recipient != donor':'False'})

if __name__=='__main__':unittest.main()
