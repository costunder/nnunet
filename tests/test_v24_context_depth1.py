"""CPU UNIT checks for the explicitly selected depth1 inference profile.

Synthetic UNIT geometry is marked; no actual patient/CUDA/full-bank claim.
"""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from hiercp_v1x import v24_context_depth1 as profile

ROOT = Path(__file__).resolve().parents[1]


class ContextDepth1CPUUnit(unittest.TestCase):
    def test_explicit_selection_required(self):
        self.assertFalse(profile.identity()["enabled_by_default"])
        with self.assertRaises(PermissionError):
            profile.install({})

    def test_only_depth_changes_and_physical_boundaries_are_exact(self):
        shape = (4, 4, 4)
        footprint = np.zeros(shape, bool)
        organ = np.zeros(shape, bool)
        organ[1, 1, 1] = True
        outside = np.full(shape, 3., np.float32)
        depth = np.zeros(shape, np.float32)
        spatial = SimpleNamespace(_field=lambda obj, *names: next(getattr(obj, name) for name in names if hasattr(obj, name)))
        fields = SimpleNamespace(footprint=footprint, organ_mask=organ, outside_tumor_mm=outside, organ_depth=depth)
        for distance, value, admitted in ((2., 1.0001, True), (10., 2., True),
                                           (2., 1., False), (1.999, 3., False), (10.001, 4., False)):
            outside[1, 1, 1], depth[1, 1, 1] = distance, value
            context, surface, _ = profile.semantic_masks(fields, spatial)
            self.assertEqual(bool(context[1, 1, 1]), admitted)
            self.assertEqual(bool(surface[1, 1, 1]), 0. < value <= 3.)
        footprint[1, 1, 1] = True
        outside[1, 1, 1], depth[1, 1, 1] = 3., 2.
        context, surface, _ = profile.semantic_masks(fields, spatial)
        self.assertFalse(context.any())
        self.assertFalse(surface.any())

    def test_nonfinite_or_incomplete_geometry_is_rejected(self):
        spatial = SimpleNamespace(_field=lambda obj, *names: next(getattr(obj, name) for name in names if hasattr(obj, name)))
        for bad in (float("nan"), float("inf"), -1.):
            fields = SimpleNamespace(footprint=np.zeros((2, 2, 2), bool), organ_mask=np.ones((2, 2, 2), bool),
                outside_tumor_mm=np.ones((2, 2, 2), np.float32), organ_depth=np.full((2, 2, 2), bad, np.float32))
            with self.assertRaises(ValueError):
                profile.semantic_masks(fields, spatial)

    def test_ast_sites_are_exact_and_other_original_guards_remain(self):
        tree = ast.parse((ROOT / "work/v14_matched_learning_DEBUG_20261004_r3/source/v1.0/hiercp/spatial.py").read_text())
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(profile._canonical_transform(functions["canonical_coordinate_sets"]), 1)
        validator = functions["validate_canonical_coordinates"]
        self.assertEqual(profile._validator_transform(validator), 4)
        changed = ast.unparse(validator)
        for rule in ("out-of-bounds", "duplicate nodes", "pure liver parenchyma", "configured annulus", "actual liver-surface band"):
            self.assertIn(rule, changed)
        self.assertNotIn("context_rows", changed)
        self.assertIn("liver_depth[tuple(context.T)] <= 1.0", changed)

    def test_actual_original_builder_sampling_and_official_STU_cpu_UNIT(self):
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", ORIGINAL_SCRIPT], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf8", timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("depth1 original/STU CPU UNIT PASS", result.stdout)


ORIGINAL_SCRIPT = r'''
import copy,hashlib,json,sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch_geometric.data import Batch
from hiercp_v1x.historical_checkpoint import _activate
from hiercp_v1x.transition_v1_local import _runtime
from hiercp_v1x import bounded_scope as scope
from hiercp_v1x import transition_v1_empty_context as target_adapter
from hiercp_v1x import v24_context_depth1 as profile
from hiercp_v1x.v24_stunet import STUNetL0Encoder
root=Path.cwd();snapshot=root/'work/v14_matched_learning_DEBUG_20261004_r3/source/v1.0'
original_sha={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (snapshot/'hiercp').glob('*.py')}
torch.set_num_threads(2);_activate(snapshot,None);runtime=_runtime(expected_snapshot_root=snapshot)
from hiercp import schema,spatial,local,sample,model
from hiercp.common import organ_depth_mm
from hiercp_v22.data import sources,candidate_spec
from hiercp_v1x.v24_inputs import prepare_donor
config=scope.configure(json.loads((snapshot/'config/train.json').read_text())['graph'],10.)
base_config=config.to_dict();old_D_identity=target_adapter.identity();old_scope=copy.deepcopy(runtime['scope'])
kwargs={k:v for k,v in json.loads((snapshot/'config/train.json').read_text())['model'].items()
 if k in ('hidden_dim','heads','dropout','dense_base_channels','dense_feature_dim','dense_batch_size',
          'channels_last_3d','checkpoint_local_blocks','checkpoint_dense_encoder')}
torch.manual_seed(42)
original=model.LocalTumorContextPyGEncoder(**kwargs,layers=3)
original.eval()
cnn_state={name:value.detach().clone() for name,value in original.state_dict().items() if torch.is_tensor(value)}
cnn_param_ids={name:id(value) for name,value in original.named_parameters()}
original_cnn_forward=model.LocalTumorContextPyGEncoder.forward_graph
encoder=STUNetL0Encoder(original,root/'work/pretrained_l0_latency_DEBUG_20261008/assets/small_ep4k.model').eval()
state={name:value.detach().clone() for name,value in encoder.state_dict().items() if torch.is_tensor(value)}
param_ids={name:id(value) for name,value in encoder.named_parameters()}
def case(thickness,case_id):
 shape=(32,32,32);organ=np.zeros(shape,bool);organ[3:29,3:29,14:14+thickness]=True
 label=organ.astype(np.uint8);label[14:16,14:16,14:16]=2
 organ=np.isin(label,[1,2])
 return SimpleNamespace(image=np.random.default_rng(42).normal(60,15,shape).astype(np.float32),
  label=label,spacing=np.ones(3,np.float32),shape=shape,paths=SimpleNamespace(case_id=case_id),
  image_affine=np.eye(4),label_affine=np.eye(4)),organ
normal,normal_organ=case(12,'UNIT_normal_same_patient');normal_source=sources(normal,2,20)[0][0]
old_prepared=prepare_donor(normal,normal_source,config=config,seed=42,ct_clip=(-200.,250.))
thin,thin_organ=case(3,'UNIT_thin_same_patient');source=sources(thin,2,20)[0][0]
try:prepare_donor(thin,source,config=config,seed=42,ct_clip=(-200.,250.))
except spatial.EmptyCanonicalNodeError:pass
else:raise AssertionError('Olddepth4 must reject thin source')
selected=profile.install(runtime,approved_profile=True)
assert config.to_dict()==base_config and runtime['scope']==old_scope and target_adapter.identity()==old_D_identity
assert param_ids=={name:id(value) for name,value in encoder.named_parameters()}
for name,value in state.items():assert torch.equal(value,encoder.state_dict()[name]),name
try:local.build_local_graph(normal,normal_source,candidate_spec(normal_source,(20,18,18)),
 full_organ_mask=normal_organ,organ_depth=organ_depth_mm(normal_organ,normal.spacing),config=config,
 rng=np.random.default_rng(42),ct_clip=(-200.,250.),prepared_source=old_prepared)
except ValueError as error:assert 'Old or foreign prepared source' in str(error)
else:raise AssertionError('Old profile prepared source accepted')
prepared=prepare_donor(thin,source,config=config,seed=42,ct_clip=(-200.,250.))
assert prepared.canonical_counts['source_context']>0
transported=replace(prepared,source_footprint=prepared.source_footprint.copy())
transported.v1x_bounded_scope_contract=runtime['scope']['contract_sha256']
def construct(center,prepared_source=transported):
 return local.build_local_graph(thin,source,candidate_spec(source,center),full_organ_mask=thin_organ,
  organ_depth=organ_depth_mm(thin_organ,thin.spacing),config=config,rng=np.random.default_rng(42),
  ct_clip=(-200.,250.),prepared_source=prepared_source)
built=construct((20,18,15));proof=built.source_local[profile.PROOF_KEY]
assert proof['typed_overlap_voxels']>0 and proof['minimum_liver_depth_exclusive_mm']==1.
assert np.array_equal(built.source_patch,prepared.source_patch)
view=sample.build_local_view(built.source_local,built.target_local,config,seed=42)
assert view[profile.GRAPH_KEY]==selected['contract_sha256']
# Exercise the actual scoring runtime, including its exact wrapper traversal,
# private edge-attribute/radius clones and original two-view materializer.
from hiercp_v1x import v24_input_runtime as input_runtime
from hiercp_v1x import v24_inputs
chain=input_runtime._sampling_proof(sample.build_local_view,sample._edge_attributes)
assert len(chain)>=4 and any('_edge_attributes' in function.__code__.co_names for function,*_ in chain)
admitted=input_runtime._sampler()
assert admitted['build'] is sample.build_local_view and admitted['chain']==chain
record=dict(format=v24_inputs.LOCAL_FORMAT,case_id=thin.paths.case_id,donor_case_id=thin.paths.case_id,
 component_id=int(source.component_id),observation_id='UNIT_depth1_actual_materializer',seed=42,
 graph_config=config.to_dict(),source_patch=torch.from_numpy(built.source_patch.astype(np.float16)),
 target_patch=torch.from_numpy(built.target_patch.astype(np.float16)),source_local=built.source_local,
 target_local=built.target_local,recipient_GT_used_in_forward=False,P_U_labels_in_forward=False)
record['tensor_sha256']=v24_inputs.tensor_digest({key:record[key] for key in
 ('source_patch','target_patch','source_local','target_local')})
ordinary=v24_inputs.materialize_pair(record,epoch=0)
private=input_runtime._materialize_pair(record,epoch=0)
for first,second in zip(ordinary[0],private[0]):
 assert first[profile.GRAPH_KEY]==second[profile.GRAPH_KEY]==selected['contract_sha256']
 for role in schema.LOCAL_NODE_TYPES:
  for key,value in first[role].items():assert torch.equal(value,second[role][key]),(role,key)
 for edge in schema.LOCAL_EDGE_TYPES:
  for key,value in first[edge].items():assert torch.equal(value,second[edge][key]),(edge,key)
for first,second in zip(ordinary[1:],private[1:]):assert torch.equal(first,second)
assert input_runtime._sampler()['chain']==chain
try:input_runtime._sampling_proof(lambda *args,**kwargs:None,sample._edge_attributes)
except ValueError as error:assert 'Original sampling attribute branch changed' in str(error)
else:raise AssertionError('Foreign wrapper admission was weakened')
# Grids2.5/4mm may choose different representatives, even in an overlapping
# mask. Directly validate one real shared voxel as two distinct typed roles.
fields=local._patch_fields(thin,source.anchor_center,spatial.exact_source_footprint(source),thin_organ,
 organ_depth_mm(thin_organ,thin.spacing),config=config,erase_target=False,ct_clip=(-200.,250.))
points=spatial.canonical_coordinate_sets(fields,config,thin.spacing)
shared=next(row for row in points['context'] if spatial._field(fields,'organ_depth','organ_boundary_mm','liver_depth_mm')[tuple(row)]<=3.)
points['liver_surface']=np.unique(np.concatenate([points['liver_surface'],shared[None]]),axis=0)
spatial.validate_canonical_coordinates(fields,points,config)
duplicate=copy.deepcopy(points);duplicate['context']=np.concatenate([duplicate['context'],shared[None]])
try:spatial.validate_canonical_coordinates(fields,duplicate,config)
except ValueError as error:assert 'duplicate nodes' in str(error)
else:raise AssertionError('Duplicate within one typed role admitted')
changed=copy.deepcopy(built.source_local);changed['nodes']['source_context']['x'][0,0]+=1
try:sample.build_local_view(changed,built.target_local,config,seed=42)
except ValueError as error:assert 'changed actual node tensors' in str(error)
else:raise AssertionError('Node mutation accepted')
changed=copy.deepcopy(built.source_local);changed[profile.PROOF_KEY]['semantic_context_voxels']+=1
try:sample.build_local_view(changed,built.target_local,config,seed=42)
except ValueError as error:assert 'proof content changed' in str(error)
else:raise AssertionError('Proof mutation accepted')
# A target outside the organ has no actual surface and must never get a token.
try:construct((20,18,30))
except ValueError:pass
else:raise AssertionError('Target without real surface evidence admitted')
# An actual one-voxel-thin source still cannot form depth>1 context.
ultrathin,ultraorgan=case(1,'UNIT_very_thin_same_patient');ultrasource=sources(ultrathin,2,20)[0][0]
try:prepare_donor(ultrathin,ultrasource,config=config,seed=42,ct_clip=(-200.,250.))
except spatial.EmptyCanonicalNodeError:pass
else:raise AssertionError('Empty-source fallback introduced')
batch=Batch.from_data_list([view,view]);source_patch=torch.from_numpy(built.source_patch)[None]
target_patches=torch.from_numpy(np.stack([built.target_patch,built.target_patch]))
with torch.inference_mode():outputs=encoder(batch,source_patch,torch.zeros(2,dtype=torch.long),target_patches)
assert len(outputs)==12
for key,value in outputs.items():assert value.shape==(2,128) and torch.isfinite(value).all(),key
for name,value in state.items():assert torch.equal(value,encoder.state_dict()[name]),name
# The GPU5 original CNN/GAT L0 keeps its complete actual computation. Compare
# the wrapped path to the captured D/scope method on the same full dense maps.
with torch.inference_mode():
 source_map,target_map=original.encode_dense_maps(source_patch,torch.zeros(2,dtype=torch.long),target_patches)
 cnn_reference=original_cnn_forward(original,batch,source_map,target_map)
 cnn_actual=original.forward_graph(batch,source_map,target_map)
assert len(cnn_actual)==12 and cnn_actual.keys()==cnn_reference.keys()
for key,value in cnn_actual.items():
 assert value.shape==(2,128) and torch.isfinite(value).all(),key
 assert torch.equal(value,cnn_reference[key]),key
assert cnn_param_ids=={name:id(value) for name,value in original.named_parameters()}
for name,value in cnn_state.items():assert torch.equal(value,original.state_dict()[name]),name
# Every semantic role's ownership is checked before the original computation;
# optional empty liver-surface roles retain their existing policy.
for role in schema.LOCAL_NODE_TYPES:
 altered=batch.clone();altered[role].batch[0]=2
 with torch.inference_mode():
  try:original.forward_graph(altered,source_map,target_map)
  except ValueError as error:assert 'ownership/count differs' in str(error),(role,str(error))
  else:raise AssertionError('Foreign owner admitted for '+role)
altered=batch.clone();altered[profile.GRAPH_KEY]=['foreign','foreign']
with torch.inference_mode():
 try:original.forward_graph(altered,source_map,target_map)
 except ValueError as error:assert 'missing fresh depth1 profile' in str(error)
 else:raise AssertionError('Original CNN accepted a foreign profile')
original.train()
with torch.inference_mode():
 try:original.forward_graph(batch,source_map,target_map)
 except ValueError as error:assert 'eval/no-grad inference only' in str(error)
 else:raise AssertionError('Training input policy was silently changed')
original.eval()
for name,value in cnn_state.items():assert torch.equal(value,original.state_dict()[name]),name
assert original_sha=={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (snapshot/'hiercp').glob('*.py')}
print('depth1 original/STU CPU UNIT PASS')
'''


if __name__ == "__main__":
    unittest.main()
