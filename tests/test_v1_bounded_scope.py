"""UNIT checks, explicitly synthetic geometry; no model-quality evidence."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hiercp_v1x import bounded_scope as scope


class MarginContractTests(unittest.TestCase):
    def test_invalid_explicit_margins(self):
        for value in (True, 0, -10, 2, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                scope._margin(value)

    def test_selected_margins(self):
        self.assertEqual([scope._margin(v) for v in (10, 20, 30)], [10., 20., 30.])

    def test_surface_removal_must_match_exactly_one_ast_site(self):
        tree = ast.parse("def f():\n if present:\n  shape = _expanded_surface_shape()\n return shape")
        self.assertEqual(scope._remove_surface_expansion(tree.body[0]), 1)
        self.assertFalse(any(isinstance(x, ast.Call) for x in ast.walk(tree.body[0])))
        self.assertEqual(scope._remove_surface_expansion(tree.body[0]), 0)

    def test_sdf_normalization_rejects_an_unexpected_source_expression(self):
        node = ast.parse('def f():\n context_radius = 10.0').body[0]
        with self.assertRaisesRegex(ValueError, 'SDF normalization'):
            scope._fix_sdf_normalization(node)


SCRIPT = r'''
import copy, hashlib, json, sys, uuid, zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from hiercp_v1x import bounded_scope as adapter
from tests.artifacts import unit_artifact_root
root=Path.cwd()
source=unit_artifact_root()/('v1_bounded_scope_UNIT_'+uuid.uuid4().hex)
source.mkdir(parents=True,exist_ok=False)
archive=root/'versions/v1/pipeline_v1_source.zip'
archive_before=hashlib.sha256(archive.read_bytes()).hexdigest()
with zipfile.ZipFile(archive) as z:
    z.extractall(source)
before={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob('*') if p.is_file()}
sys.path.insert(0,str(source))
receipt=adapter.install(10,source)
from hiercp import spatial, schema, model, sample
base=json.loads((source/'config/train.json').read_text())['graph']
configured=adapter.configure(base,10)
actual=configured.to_dict()
assert {k for k in base if base[k]!=actual[k]}=={'adaptive_roi_margin_mm','context_outer_radius_mm'}
assert actual['context_shells_mm']==[4.,12.,28.] and actual['context_radius_mm']==28.
assert actual['patch_size']==48
for m in (20,30):
    alternate=adapter.configure(base,m)
    assert alternate.adaptive_roi_margin_mm==m and alternate.context_outer_radius_mm==m
    assert alternate.context_shells_mm==(4.,12.,28.)
    try: spatial.adaptive_native_shape((5,7,9),(1,1,1),alternate,center_liver_depth_mm=40)
    except ValueError: pass
    else: raise AssertionError('Mixed configured scope accepted')
assert spatial.adaptive_native_shape((5,7,9),(1,1,1),configured,center_liver_depth_mm=45)==(25,27,29)
assert spatial.adaptive_native_shape((5,7,9),(.8,1.,1.2),configured,center_liver_depth_mm=60)==(31,27,27)
for depth in (float('nan'),float('inf'),-1.):
    try: spatial.adaptive_native_shape((5,7,9),(1,1,1),configured,center_liver_depth_mm=depth)
    except ValueError: pass
    else: raise AssertionError('Invalid depth hidden by bounded scope')
image=np.zeros((101,101,101),np.float32)
organ=np.ones_like(image,dtype=bool)
depth=np.full_like(image,45.)
footprint=np.ones((5,7,9),bool)
payload=spatial.build_patch_payload(image=image,center=(50,50,50),footprint=footprint,
    full_organ=organ,organ_depth=depth,spacing=(1,1,1),config=configured,
    erase_target=False,ct_clip=(-200,250))
assert payload['footprint'].shape==(25,27,29)
assert int(payload['footprint'].sum())==int(footprint.sum())==315
assert payload['model_input'].shape==(5,48,48,48)
expected_sdf=np.clip(spatial.signed_distance(payload['footprint'],np.array((1,1,1),np.float32))/28.,-2.,2.).astype(np.float32)
assert np.array_equal(payload['tumor_sdf_norm'],expected_sdf)
assert receipt['tumor_sdf_normalization_mm']==28.
fields=SimpleNamespace(**payload)
coords=spatial.canonical_coordinate_sets(fields,configured,(1,1,1))
assert len(coords['liver_surface'])==0
assert all(len(coords[k]) for k in ('surface','interior','context'))
missing=copy.deepcopy(coords);missing['context']=np.empty((0,3),np.int64)
try:spatial.validate_canonical_coordinates(fields,missing,configured)
except spatial.EmptyCanonicalNodeError:pass
else:raise AssertionError('Mandatory missing context silently accepted')
# Empty surface is legal only when the actual band is absent.
fields.liver_depth_mm=fields.organ_depth=np.ones_like(payload['liver_depth_mm'])
try:spatial.validate_canonical_coordinates(fields,coords,configured)
except ValueError:pass
else:raise AssertionError('Existing real surface was silently omitted')
try:sample.build_local_view({}, {}, configured, seed=42)
except ValueError as error:assert 'Unbound native' in str(error)
else:raise AssertionError('Unbound canonical payload accepted')
# Mixed missing-role minibatches retain original graph-major associations.
maps=torch.stack((torch.ones(2,3,3,3),torch.full((2,3,3,3),7.)))
grid=torch.zeros((2,3));owners=torch.tensor([1,1])
features=sample.sample_dense_features_variable(maps,grid,owners)
assert features.shape==(2,2) and torch.equal(features,torch.full((2,2),7.))
empty=sample.sample_dense_features_variable(maps,torch.empty((0,3)),torch.empty(0,dtype=torch.long))
assert empty.shape==(0,2)
net=model.HierarchicalPyGPlacementModel(**json.loads((source/'config/train.json').read_text())['model'])
assert sum(p.numel() for p in net.parameters())==10434532
assert '|bounded_scope_'+receipt['contract_sha256'] in net.architecture_version
pooled=net.local_encoder.pool['source_liver_surface'](torch.empty((0,128)),
    index=torch.empty(0,dtype=torch.long),dim_size=2)
assert pooled.shape==(2,128) and torch.equal(pooled,torch.zeros((2,128)))
try:adapter.install(20,source)
except RuntimeError:pass
else:raise AssertionError('Scope replaced in the same process')
after={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
assert after==before
assert hashlib.sha256(archive.read_bytes()).hexdigest()==archive_before
print(json.dumps({'UNIT':True,'synthetic':True,'production_ready':False,
    'checks':['only two config fields changed','fixed CNN48','fixed shell and coordinate scales',
    'mixed-scope rejection','bbox margin','anisotropic native spacing','finite depth',
    'full footprint retained','fixed SDF normalization','actual surface absence','mandatory context rejection',
    'present surface cannot be dropped','unbound native locals rejection','ragged dense owners',
    'empty dense role shape','parameter count unchanged','bound architecture',
    'empty pooling batch cardinality','scope replacement rejected','source and archive preserved']}))
'''


class IsolatedArchiveGeometryTests(unittest.TestCase):
    def test_original_snapshot_bounded_geometry_and_role_absence(self):
        import os
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,'-u','-c',SCRIPT],cwd=ROOT,env=env,
                              capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        report=json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(report['UNIT'])
        self.assertFalse(report['production_ready'])
        self.assertEqual(len(report['checks']),20)


if __name__=='__main__':unittest.main()
