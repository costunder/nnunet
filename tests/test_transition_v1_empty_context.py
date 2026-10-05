"""Explicit synthetic UNIT admission checks; actual CT/CUDA evidence is separate."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from hiercp_v1x import transition_v1_empty_context as adapter

ROOT = Path(__file__).resolve().parents[1]


def geometry():
    shape = (5, 5, 5)
    footprint = np.zeros(shape, dtype=bool)
    footprint[2, 2, 2] = True
    organ = np.zeros(shape, dtype=bool)
    organ[0, 0, 0] = True
    depth = np.zeros(shape, dtype=np.float32)
    depth[0, 0, 0] = 2.
    outside = np.full(shape, 3., dtype=np.float32)
    fields = SimpleNamespace(footprint=footprint, organ_mask=organ, organ_depth=depth, outside_tumor_mm=outside)
    config = SimpleNamespace(context_inner_radius_mm=2., context_outer_radius_mm=10.,
        boundary_depth_mm=3., context_liver_surface_separation_mm=1.)
    points = dict(context=np.empty((0, 3), dtype=np.int64), liver_surface=np.array([[0, 0, 0]], dtype=np.int64))
    spatial = SimpleNamespace(_field=lambda fields, *names: next(getattr(fields, name) for name in names if hasattr(fields, name)))
    return fields, points, config, spatial


class RecipientAbsenceUnit(unittest.TestCase):
    def test_complete_mask_absence_has_real_surface_proof(self):
        proof = adapter.geometry_proof(*geometry())
        self.assertEqual(proof['semantic_context_voxels'], 0)
        self.assertEqual(proof['liver_surface_voxels'], 1)
        self.assertEqual(proof['canonical_liver_surface_nodes'], 1)
        self.assertEqual(proof['context_outer_radius_mm'], 10.)

    def test_present_context_cannot_be_omitted(self):
        fields, points, config, spatial = geometry()
        fields.organ_mask[1, 1, 1] = True
        fields.organ_depth[1, 1, 1] = 5.
        with self.assertRaisesRegex(ValueError, 'Present real recipient'):
            adapter.geometry_proof(fields, points, config, spatial)

    def test_missing_actual_surface_is_rejected(self):
        fields, points, config, spatial = geometry()
        fields.organ_mask[:] = False
        with self.assertRaisesRegex(ValueError, 'actual liver-surface'):
            adapter.geometry_proof(fields, points, config, spatial)

    def test_missing_surface_nodes_is_rejected(self):
        fields, points, config, spatial = geometry()
        points['liver_surface'] = np.empty((0, 3), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, 'nonempty real liver-surface'):
            adapter.geometry_proof(fields, points, config, spatial)

    def test_nonfinite_semantics_are_rejected(self):
        fields, points, config, spatial = geometry()
        fields.organ_depth[1, 1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            adapter.geometry_proof(fields, points, config, spatial)

    def test_ast_sites_are_exact_and_target_only(self):
        node = ast.parse('def f():\n if values.shape[0] == 0: raise ValueError()').body[0]
        self.assertEqual(adapter._coordinate_transform(node), 1)
        self.assertEqual(adapter._coordinate_transform(node), 0)
        node = ast.parse('def f():\n if ids.size == 0: raise ValueError()').body[0]
        self.assertEqual(adapter._view_transform(node), 1)
        self.assertIn('target_context', ast.unparse(node))
        self.assertNotIn("'source_context'", ast.unparse(node))
        node = ast.parse('def f():\n if batch_index.numel() == 0: raise ValueError()').body[0]
        self.assertEqual(adapter._shell_transform(node), 1)
        self.assertIn("node_type != 'target_context'", ast.unparse(node))

    def test_canonical_proof_must_match_real_role_counts(self):
        proof = adapter.geometry_proof(*geometry())
        nodes = lambda n: {'x': np.zeros((n, 16), np.float16)}
        source = {'nodes': {'source_context': nodes(1)}}
        target = {'nodes': {'target_context': nodes(0), 'target_liver_surface': nodes(1)}, adapter.PROOF_KEY: proof}
        self.assertEqual(adapter.validate_local_proof(source, target), proof)
        broken = copy.deepcopy(target)
        broken[adapter.PROOF_KEY]['semantic_context_voxels'] = 1
        with self.assertRaises(ValueError):
            adapter.validate_local_proof(source, broken)
        broken = copy.deepcopy(target)
        broken['nodes']['target_liver_surface'] = nodes(0)
        with self.assertRaises(ValueError):
            adapter.validate_local_proof(source, broken)
        with self.assertRaisesRegex(ValueError, 'source context'):
            adapter.validate_local_proof({'nodes': {'source_context': nodes(0)}}, target)

    def test_source_scope_stays_byte_identical_and_contract_unchanged(self):
        import hashlib
        self.assertEqual(hashlib.sha256((ROOT/'hiercp_v1x/bounded_scope.py').read_bytes()).hexdigest(),
                         '4e52f4751e9d3e885f088bdf5639cbda649099c4db52de069902772a2223853b')

    def test_original_view_and_shell_paths_in_fresh_process(self):
        result = subprocess.run([sys.executable, '-c', SCRIPT], cwd=ROOT, text=True,
                                capture_output=True, timeout=150)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('recipient absence UNIT PASS', result.stdout)


SCRIPT = r'''
import copy,json,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
import numpy as np
import torch
from torch_geometric.data import Batch
from hiercp_v1x import bounded_scope as scope
from hiercp_v1x import transition_v1_empty_context as adapter
root=Path.cwd()
source=root/'work/v14_matched_learning_DEBUG_20261004_r3/source/v1.0'
sys.path.insert(0,str(source))
receipt=scope.install(10,source)
from hiercp import schema,spatial,local,sample,model
config=scope.configure(json.loads((source/'config/train.json').read_text())['graph'],10)
runtime=dict(schema=schema,spatial=spatial,local=local,sample=sample,model=model,scope=receipt,snapshot=source)
def node(n):
 x=torch.zeros((n,16),dtype=torch.float16);x[:,13]=0.
 return dict(x=x,grid=torch.zeros((n,3),dtype=torch.float16),pos=torch.zeros((n,3)),pos_mm=torch.zeros((n,3)))
nodes={role:node(1) for role in schema.LOCAL_NODE_TYPES}
edges={edge:torch.empty((2,0),dtype=torch.int32) for edge in schema.LOCAL_EDGE_TYPES}
base=dict(format='canonical-full-v22',geometry_contract=spatial.LEVEL0_GEOMETRY_CONTRACT,
 v1x_bounded_scope_contract=receipt['contract_sha256'],v1x_bounded_scope_margin_mm=10.)
src={**base,'nodes':{k:v for k,v in nodes.items() if k in schema.SOURCE_LOCAL_NODE_TYPES},
 'edges':{k:v for k,v in edges.items() if k[0] in schema.SOURCE_LOCAL_NODE_TYPES and k[2] in schema.SOURCE_LOCAL_NODE_TYPES}}
dst={**base,'nodes':{k:v for k,v in nodes.items() if k not in schema.SOURCE_LOCAL_NODE_TYPES},
 'edges':{k:v for k,v in edges.items() if k not in src['edges']},'transform':torch.eye(3)}
before=sample.build_local_view(src,dst,config,seed=42)
adapter.install(runtime)
# Recipient opt-in is task/thread scoped; other source workers retain rejection.
mask=np.zeros((5,5,5),bool);mask[2,2,2]=True
organ=np.zeros_like(mask);organ[0,0,0]=True
depth=np.zeros(mask.shape,np.float32);depth[0,0,0]=2.
fields=SimpleNamespace(footprint=mask,organ_mask=organ,organ_depth=depth,
 outside_tumor_mm=np.full(mask.shape,3.,np.float32))
points=dict(surface=np.array([[2,2,2]]),interior=np.array([[2,2,2]]),
 context=np.empty((0,3),np.int64),liver_surface=np.array([[0,0,0]]))
barrier=Barrier(2)
def coordinate_request(target):
 token=adapter._TARGET.set(target)
 try:
  barrier.wait()
  try:spatial.validate_canonical_coordinates(fields,points,config)
  except spatial.EmptyCanonicalNodeError:return 'rejected'
  return 'admitted'
 finally:adapter._TARGET.reset(token)
with ThreadPoolExecutor(max_workers=2) as pool:
 tasks=[pool.submit(coordinate_request,target) for target in (True,False)]
 assert [task.result() for task in tasks]==['admitted','rejected']
try:spatial.validate_canonical_coordinates(fields,points,config)
except spatial.EmptyCanonicalNodeError:pass
else:raise AssertionError('Recipient admission leaked to unbound/source validation')
after=sample.build_local_view(src,dst,config,seed=42)
for role in schema.LOCAL_NODE_TYPES:
 for key,value in before[role].items():assert torch.equal(value,after[role][key]),(role,key)
for edge in schema.LOCAL_EDGE_TYPES:
 for key,value in before[edge].items():assert torch.equal(value,after[edge][key]),(edge,key)
empty=copy.deepcopy(dst);empty['nodes']['target_context']=node(0)
empty[adapter.PROOF_KEY]=dict(format=adapter.FORMAT,target_context_observed_absent=True,
 semantic_context_voxels=0,liver_surface_voxels=1,canonical_liver_surface_nodes=1,
 roi_shape=[5,5,5],context_inner_radius_mm=2.,context_outer_radius_mm=10.,
 minimum_liver_depth_exclusive_mm=4.,source_context_required=True)
absent=sample.build_local_view(src,empty,config,seed=42)
assert absent['target_context'].x.shape==(0,16)
assert torch.equal(absent[adapter.GRAPH_PROOF_KEY],torch.tensor([[1,0,1,1]]))
assert Batch.from_data_list([after,absent])[adapter.GRAPH_PROOF_KEY].shape==(2,4)
assert Batch.from_data_list([absent,absent])[adapter.GRAPH_PROOF_KEY].shape==(2,4)
net=model.LocalTumorContextPyGEncoder(**{k:v for k,v in json.loads((source/'config/train.json').read_text())['model'].items()
 if k in ('hidden_dim','heads','dropout','dense_base_channels','dense_feature_dim','dense_batch_size',
 'channels_last_3d','checkpoint_local_blocks','checkpoint_dense_encoder')},layers=3)
fused,shells=net._pool_context_shells('target_context',torch.empty((0,128)),torch.empty((0,16)),torch.empty(0,dtype=torch.long),2)
assert fused.shape==(2,128) and len(shells)==3
for shell in shells:assert shell.shape==(2,128)
fused.sum().backward()
assert all(net.empty_context_shell['target_context_c'+str(i)].grad is not None for i in range(3))
try:net._pool_context_shells('source_context',torch.empty((0,128)),torch.empty((0,16)),torch.empty(0,dtype=torch.long),2)
except RuntimeError:pass
else:raise AssertionError('Absent source context accepted')
print('recipient absence UNIT PASS')
'''


if __name__ == '__main__':
    unittest.main()
