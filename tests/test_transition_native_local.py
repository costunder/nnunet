"""CPU geometry/contract tests; no neural forward, training or fake metrics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from hiercp_v1x.transition_native_local import native_bounds,native_transition_spec,_json_metadata

ROOT=Path(__file__).resolve().parents[1]


class NativeGeometryContracts(unittest.TestCase):
    def test_native_formula_matches_real_voxel_bbox_margin(self):
        center=np.array([20,30,40]);spacing=np.array([.8,1.2,2.5])
        low=np.array([-2.5,-3.5,-1.5])*spacing;high=np.array([3.5,4.5,2.5])*spacing
        origin,end=native_bounds(center,low,high,spacing,[80,90,100])
        np.testing.assert_array_equal(origin,np.maximum(center+np.floor((low-10)/spacing+.5).astype(int),0))
        np.testing.assert_array_equal(end,np.minimum(center+np.ceil((high+10)/spacing+.5).astype(int),[80,90,100]))
        self.assertNotEqual(tuple(end-origin),(48,48,48))

    def test_native_boundary_clips_ct_domain_without_moving_anchor(self):
        origin,end=native_bounds([0,7,18],[-.5,-.5,-.5],[.5,.5,.5],[1,1,1],[19,19,19])
        np.testing.assert_array_equal(origin,[0,0,8]);np.testing.assert_array_equal(end,[11,18,19])

    def test_invalid_native_center_spacing_margin_rejected(self):
        for center,spacing,margin in (([1.5,2,3],[1,1,1],10),([1,2,3],[1,0,1],10),
                ([-1,2,3],[1,1,1],10),([1,2,20],[1,1,1],10),([1,2,3],[1,1,1],8)):
            with self.subTest(center=center,spacing=spacing,margin=margin),self.assertRaises(ValueError):
                native_bounds(center,[-.5]*3,[.5]*3,spacing,[20]*3,margin)

    def test_dependency_closed_changes_are_explicit(self):
        spec=native_transition_spec()
        self.assertFalse(spec['resampling']);self.assertFalse(spec['target_erasure'])
        self.assertFalse(spec['candidate_transforms_applied']);self.assertFalse(spec['relation_corruption_applied'])
        self.assertFalse(spec['original_six_view_consistency']);self.assertFalse(spec['auxiliary_role_embeddings'])
        self.assertEqual(spec['output'],'actual_native_fused128_only')
        self.assertEqual(spec['training_candidate_pool'],128);self.assertEqual(spec['GT_index'],0)
        self.assertEqual(spec['architecture']['channels'],[12,24,32])
        self.assertEqual(spec['architecture']['convolutions'],[2,3,3])

    def test_report_corruption_metadata_serializes_without_mutating_tensors(self):
        import torch
        offset=torch.tensor([1.,-2.,3.]);scalar=torch.tensor(4,dtype=torch.long)
        audit={'original_corruption':{'translation':offset,'nested':(scalar,np.float32(.5))}}
        report=_json_metadata(audit)
        self.assertEqual(json.loads(json.dumps(report,allow_nan=False)),
            {'original_corruption':{'translation':[1.,-2.,3.],'nested':[4,.5]}})
        report['original_corruption']['translation'][0]=99
        self.assertIs(audit['original_corruption']['translation'],offset)
        self.assertEqual(offset.tolist(),[1.,-2.,3.])
        self.assertIsInstance(audit['original_corruption']['nested'][0],torch.Tensor)

    @unittest.skipUnless((ROOT/'work/v1x_real_CT_DEBUG_20261003_prepare/fixture.pt').exists(),
        'Optional actual-CT original fixture not installed')
    def test_actual_raw_voxels_order_geometry_and_readonly_metadata(self):
        script=r'''
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from hiercp_v1x.transition_native_local import NativeCurriculumLoader
from hiercp_v1x.scope_probe_support import activate_original
activate_original(Path('work/v14_matched_learning_DEBUG_20261004_r3/source/v1.0'))
import torch,numpy as np,json
f=torch.load('work/v1x_real_CT_DEBUG_20261003_prepare/fixture.pt',map_location='cpu',weights_only=False,mmap=True)
# The native path has no dependency on old erased/resampled dense inputs.
samples=[{k:v for k,v in s.items() if k not in ('source_patch','target_patches')} for s in f['samples']]
l=NativeCurriculumLoader(samples,f['raw_records'],f['config'],workers=4,resident_bytes=4*2**30,rss_bytes=32*2**30,debug=True)
b=l.get([0,1])
assert b.indices.tolist()==list(range(16)) and l.counts([0,1])==(8,8)
assert l.case_ids([0,1])==('liver_5','liver_6')
assert b.images.shape[2:]!=(48,48,48)
assert b.donor[0]==b.recipient[0] and b.donor[8]==b.recipient[8]
for i,a in enumerate(b.audit):
    v=l._volume(a['case']);lo=a['origin'];shape=a['shape']
    sl=tuple(slice(start,start+n) for start,n in zip(lo,shape));mask=v['organ'][sl]
    low,high=f['config']['ct_clip'];expected=np.where(mask,(np.clip(np.where(mask,v['ct'][sl],0),low,high)-low)/(high-low),0).astype(np.float32)
    actual=b.images[(i,0,*[slice(0,n) for n in shape])].numpy()
    assert np.array_equal(expected,actual) and np.array_equal(mask,b.organ[(i,0,*[slice(0,n) for n in shape])].numpy())
    assert not a['resampled'] and not a['target_erasure'] and a['margin_mm']==10
assert all(r['source_own_case'] and not r['transform_applied'] and not r['relation_corruption_applied'] for r in l.records)
# Real canonical corruptions can contain tensor-valued CPU audit metadata.
# The report must be JSON, while originals and subsequent crop checks remain intact.
reported=json.loads(json.dumps(l.report(),allow_nan=False))
assert len(reported['records'])==len(l.records)
assert all(r['original_corruption']==json.loads(json.dumps(__import__('hiercp_v1x.transition_native_local',fromlist=['_json_metadata'])._json_metadata(o['original_corruption'])))
    for r,o in zip(reported['records'],l.records))
l.check()
l.samples[0]['candidate_centers'][0,0]+=1
try:l.get([0])
except ValueError:pass
else:raise AssertionError('Mutable original GT/center metadata admitted')
assert not torch.cuda.is_initialized()
print(json.dumps(dict(actual_raw_CT_geometry_verified=True,neural_forward=False,CUDA_context=False,rows=len(b))))
'''
        env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=''
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,env=env,text=True,
            capture_output=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertTrue(json.loads(result.stdout.strip().splitlines()[-1])['actual_raw_CT_geometry_verified'])


if __name__=='__main__':unittest.main()
