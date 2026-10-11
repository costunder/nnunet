"""Explicit CPU UNIT proposal checks; no actual patient/CUDA/642 claim."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from hiercp_v1x import v24_matched_source_absence as proposal

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
    fields = SimpleNamespace(footprint=footprint, organ_mask=organ,
                             organ_depth=depth, outside_tumor_mm=outside)
    config = SimpleNamespace(context_inner_radius_mm=2., context_outer_radius_mm=10.,
                             boundary_depth_mm=3., context_liver_surface_separation_mm=1.)
    points = dict(context=np.empty((0, 3), dtype=np.int64),
                  liver_surface=np.array([[0, 0, 0]], dtype=np.int64))
    spatial = SimpleNamespace(_field=lambda value, *names:
        next(getattr(value, name) for name in names if hasattr(value, name)))
    return fields, points, config, spatial


class SourceAbsenceProposalCPUUnit(unittest.TestCase):
    def test_proposal_is_unenabled_and_requires_explicit_approval(self):
        self.assertFalse(proposal.identity()["enabled_by_default"])
        with self.assertRaisesRegex(PermissionError, "Explicit approval"):
            proposal.install({}, None)

    def test_complete_source_and_target_masks_have_separate_content_proofs(self):
        source = proposal.geometry_proof(*geometry(), role="source")
        target = proposal.geometry_proof(*geometry(), role="target")
        self.assertEqual(source["semantic_context_voxels"], 0)
        self.assertEqual(source["liver_surface_voxels"], 1)
        self.assertEqual(source["minimum_liver_depth_exclusive_mm"], 4.)
        self.assertNotEqual(source["geometry_sha256"], target["geometry_sha256"])

    def test_present_context_is_never_admitted_as_absent(self):
        for role in ("source", "target"):
            fields, points, config, spatial = geometry()
            fields.organ_mask[1, 1, 1] = True
            fields.organ_depth[1, 1, 1] = 5.
            with self.assertRaisesRegex(ValueError, "Present real " + role):
                proposal.geometry_proof(fields, points, config, spatial, role=role)

    def test_physical_boundaries_are_exact(self):
        # The full rule is 2<=distance<=10 and liver depth>4.
        for distance, depth, is_present in ((2., 4.01, True), (10., 4.01, True),
                                          (1.99, 6., False), (10.01, 6., False), (3., 4., False)):
            fields, points, config, spatial = geometry()
            fields.organ_mask[1, 1, 1] = True
            fields.organ_depth[1, 1, 1] = depth
            fields.outside_tumor_mm[1, 1, 1] = distance
            if is_present:
                with self.assertRaisesRegex(ValueError, "Present real source"):
                    proposal.geometry_proof(fields, points, config, spatial, role="source")
            else:
                self.assertEqual(proposal.geometry_proof(fields, points, config, spatial,
                    role="source")["semantic_context_voxels"], 0)

    def test_empty_context_requires_actual_unique_surface_voxels(self):
        for mode in ("empty", "duplicate", "non_surface", "outside", "fractional"):
            fields, points, config, spatial = geometry()
            points["liver_surface"] = {"empty": np.empty((0, 3), dtype=np.int64),
                "duplicate": np.array([[0, 0, 0], [0, 0, 0]]), "non_surface": np.array([[1, 1, 1]]),
                "outside": np.array([[5, 0, 0]]), "fractional": np.array([[0., 0., 0.]])}[mode]
            with self.assertRaises(ValueError):
                proposal.geometry_proof(fields, points, config, spatial, role="source")

    def test_nonfinite_shape_or_changed_semantic_scope_is_rejected(self):
        for mode in ("nan", "negative", "shape", "outer", "boundary"):
            fields, points, config, spatial = geometry()
            if mode == "nan": fields.organ_depth[1, 1, 1] = np.nan
            if mode == "negative": fields.organ_depth[1, 1, 1] = -1.
            if mode == "shape": fields.organ_mask = fields.organ_mask[:1]
            if mode == "outer": config.context_outer_radius_mm = 11.
            if mode == "boundary": config.boundary_depth_mm = 2.
            with self.assertRaises(ValueError):
                proposal.geometry_proof(fields, points, config, spatial, role="source")

    def test_canonical_tensor_or_source_binding_mutation_is_rejected(self):
        import torch
        nodes = {"source_context": {"x": torch.empty((0, 16))},
                 "source_liver_surface": {"x": torch.ones((1, 16))}}
        proof = proposal.bind_proof(proposal.geometry_proof(*geometry(), role="source"), nodes,
            source_binding=dict(case_id="UNIT_same_patient", component_id=1,
                source_mask_sha256="a" * 64, source_CT_patch_sha256="b" * 64,
                source_dense_patch_sha256="d" * 64))
        local = dict(nodes=nodes, v1x_bounded_scope_contract="c" * 64,
                     v1x_bounded_scope_margin_mm=10., **{proposal.PROOF_KEY: proof})
        self.assertEqual(proposal.validate_local_proof(local, role="source", contract="c" * 64), proof)
        changed = copy.deepcopy(local)
        changed["nodes"]["source_liver_surface"]["x"][0, 0] = 2.
        with self.assertRaisesRegex(ValueError, "canonical tensor binding"):
            proposal.validate_local_proof(changed, role="source", contract="c" * 64)
        changed = copy.deepcopy(local)
        changed[proposal.PROOF_KEY]["source_binding"]["component_id"] = 2
        with self.assertRaisesRegex(ValueError, "proof content changed"):
            proposal.validate_local_proof(changed, role="source", contract="c" * 64)

    def test_ast_changes_only_three_known_empty_guards(self):
        for transform, expression in ((proposal._coordinate_transform, "values.shape[0] == 0"),
                                      (proposal._view_transform, "ids.size == 0"),
                                      (proposal._shell_transform, "batch_index.numel() == 0")):
            tree = ast.parse("def f():\n if " + expression + ": raise RuntimeError()\n return x").body[0]
            self.assertEqual(transform(tree), 1)
            self.assertEqual(transform(tree), 0)
            self.assertIn("return x", ast.unparse(tree))

    def test_actual_original_sampling_and_official_STU_CPU_inference_in_fresh_process(self):
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", ORIGINAL_SCRIPT], cwd=ROOT,
                                text=True, encoding="utf-8", capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("source absence proposal original/STU CPU UNIT PASS", result.stdout)


ORIGINAL_SCRIPT = r'''
import copy, hashlib, json, sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch_geometric.data import Batch
from hiercp_v1x import bounded_scope as scope
from hiercp_v1x import transition_v1_empty_context as target_adapter
from hiercp_v1x import v24_matched_source_absence as proposal
from hiercp_v1x.historical_checkpoint import _activate
from hiercp_v1x.transition_v1_local import _runtime
from hiercp_v1x.v24_stunet import STUNetL0Encoder
root=Path.cwd(); snapshot=root/'work/v14_matched_learning_DEBUG_20261004_r3/source/v1.0'
torch.set_num_threads(2)
_activate(snapshot,None)
runtime=_runtime(expected_snapshot_root=snapshot)
from hiercp import schema,spatial,local,sample,model
from hiercp.common import organ_depth_mm
from hiercp_v22.data import sources,candidate_spec
from hiercp_v1x.v24_inputs import prepare_donor
config=scope.configure(json.loads((snapshot/'config/train.json').read_text())['graph'],10.)
contract=runtime['scope']['contract_sha256']
kwargs={k:v for k,v in json.loads((snapshot/'config/train.json').read_text())['model'].items()
 if k in ('hidden_dim','heads','dropout','dense_base_channels','dense_feature_dim','dense_batch_size',
          'channels_last_3d','checkpoint_local_blocks','checkpoint_dense_encoder')}
torch.manual_seed(42)
original=model.LocalTumorContextPyGEncoder(**kwargs,layers=3)
encoder=STUNetL0Encoder(original,root/'work/pretrained_l0_latency_DEBUG_20261008/assets/small_ep4k.model').eval()
state={name:value.detach().clone() for name,value in encoder.state_dict().items() if torch.is_tensor(value)}
param_ids={name:id(value) for name,value in encoder.named_parameters()}
def case(thin,case_id):
 shape=(32,32,32); organ=np.zeros(shape,bool)
 organ[3:29,3:29,14:17] = True if thin else False
 if not thin:organ[3:29,3:29,3:29]=True
 label=organ.astype(np.uint8);label[14:16,14:16,14:16]=2
 return SimpleNamespace(image=np.random.default_rng(42).normal(60,15,shape).astype(np.float32),
  label=label,spacing=np.ones(3,np.float32),shape=shape,paths=SimpleNamespace(case_id=case_id),
  image_affine=np.eye(4),label_affine=np.eye(4)),organ
normal,normal_organ=case(False,'UNIT_normal_same_patient')
normal_source=sources(normal,2,20)[0][0]
normal_prepared=prepare_donor(normal,normal_source,config=config,seed=42,ct_clip=(-200.,250.))
normal_built=local.build_local_graph(normal,normal_source,candidate_spec(normal_source,(20,18,18)),
 full_organ_mask=normal_organ,organ_depth=organ_depth_mm(normal_organ,normal.spacing),
 config=config,rng=np.random.default_rng(42),ct_clip=(-200.,250.),prepared_source=normal_prepared)
normal_before=sample.build_local_view(normal_built.source_local,normal_built.target_local,config,seed=42)
thin,thin_organ=case(True,'UNIT_thin_same_patient')
thin_source=sources(thin,2,20)[0][0]
try:prepare_donor(thin,thin_source,config=config,seed=42,ct_clip=(-200.,250.))
except spatial.EmptyCanonicalNodeError:pass
else:raise AssertionError('Old source admission must reject true absent donor context')
proposal.install(runtime,encoder,approved_inference_policy=True)
assert param_ids=={name:id(value) for name,value in encoder.named_parameters()}
assert set(state)=={name for name,value in encoder.state_dict().items() if torch.is_tensor(value)}
for name,value in state.items():assert torch.equal(value,encoder.state_dict()[name]),name
normal_after=sample.build_local_view(normal_built.source_local,normal_built.target_local,config,seed=42)
for role in schema.LOCAL_NODE_TYPES:
 for key,value in normal_before[role].items():assert torch.equal(value,normal_after[role][key]),(role,key)
for edge in schema.LOCAL_EDGE_TYPES:
 for key,value in normal_before[edge].items():assert torch.equal(value,normal_after[edge][key]),(edge,key)
prepared=prepare_donor(thin,thin_source,config=config,seed=42,ct_clip=(-200.,250.))
assert prepared.canonical_counts['source_context']==0
transported=replace(prepared,source_footprint=prepared.source_footprint.copy())
transported.v1x_bounded_scope_contract=contract
built=local.build_local_graph(thin,thin_source,candidate_spec(thin_source,(20,18,15)),
 full_organ_mask=thin_organ,organ_depth=organ_depth_mm(thin_organ,thin.spacing),
 config=config,rng=np.random.default_rng(42),ct_clip=(-200.,250.),prepared_source=transported)
assert built.source_local[proposal.PROOF_KEY]['role']=='source'
assert built.target_local[proposal.PROOF_KEY]['role']=='target'
assert built.source_local[proposal.PROOF_KEY]['source_binding']['component_id']==thin_source.component_id
absent=sample.build_local_view(built.source_local,built.target_local,config,seed=42)
assert absent['source_context'].x.shape==(0,16) and absent['target_context'].x.shape==(0,16)
# Both asymmetric cases use two real source components in their own UNIT case.
hybrid,hybrid_organ=case(True,'UNIT_hybrid_same_patient')
hybrid_organ[:]=False
hybrid_organ[3:13,3:29,14:17]=True
hybrid_organ[20:29,3:29,3:29]=True
hybrid.label=hybrid_organ.astype(np.uint8)
hybrid.label[5:7,14:16,14:16]=2
hybrid.label[24:26,14:16,14:16]=2
hybrid_sources=[source for source,diameter in sources(hybrid,2,20)]
thin_hybrid=next(source for source in hybrid_sources if source.anchor_center[0]<15)
thick_hybrid=next(source for source in hybrid_sources if source.anchor_center[0]>15)
def hybrid_graph(source,center):
 prepared=prepare_donor(hybrid,source,config=config,seed=42,ct_clip=(-200.,250.))
 constructed=local.build_local_graph(hybrid,source,candidate_spec(source,center),
  full_organ_mask=hybrid_organ,organ_depth=organ_depth_mm(hybrid_organ,hybrid.spacing),
  config=config,rng=np.random.default_rng(42),ct_clip=(-200.,250.),prepared_source=prepared)
 return constructed,sample.build_local_view(constructed.source_local,constructed.target_local,config,seed=42)
source_only,source_only_graph=hybrid_graph(thin_hybrid,(25,18,18))
target_only,target_only_graph=hybrid_graph(thick_hybrid,(6,18,15))
assert source_only.source_local['nodes']['source_context']['x'].shape[0]==0
assert source_only.target_local['nodes']['target_context']['x'].shape[0]>0
assert target_only.source_local['nodes']['source_context']['x'].shape[0]>0
assert target_only.target_local['nodes']['target_context']['x'].shape[0]==0
graph=Batch.from_data_list([normal_after,absent,source_only_graph,target_only_graph])
flags=proposal.validate_batch_provenance(graph,graph_count=4,node_types=encoder.node_types,contract=contract)
assert torch.equal(flags['source_context'],torch.tensor([False,True,True,False]))
assert torch.equal(flags['target_context'],torch.tensor([False,True,False,True]))
# Preserve every original recipient-proof check, even when the source is absent.
stale=copy.deepcopy(source_only.target_local)
stale[target_adapter.PROOF_KEY]=copy.deepcopy(built.target_local[target_adapter.PROOF_KEY])
try:target_adapter.validate_local_proof(source_only.source_local,stale)
except ValueError:pass
else:raise AssertionError('Present target may not carry a stale original absence proof')
for key,value in [('semantic_context_voxels',1),('source_context_required',False),
                  ('canonical_liver_surface_nodes',0),('roi_shape',[0,1,1]),('context_outer_radius_mm',11.)]:
 bad_target=copy.deepcopy(built.target_local)
 bad_target[target_adapter.PROOF_KEY][key]=value
 try:target_adapter.validate_local_proof(built.source_local,bad_target)
 except ValueError:pass
 else:raise AssertionError('Original recipient proof obligation was bypassed: '+key)
try:local.build_local_graph(normal,thin_source,candidate_spec(thin_source,(20,18,18)),
 full_organ_mask=normal_organ,organ_depth=organ_depth_mm(normal_organ,normal.spacing),
 config=config,rng=np.random.default_rng(42),ct_clip=(-200.,250.),prepared_source=transported)
except ValueError as error:assert 'same-patient component' in str(error),str(error)
else:raise AssertionError('Proposed absent donor cannot cross patients')
corrupted=replace(prepared,source_patch=prepared.source_patch.copy())
corrupted.v1x_bounded_scope_contract=contract
corrupted.source_patch[0,0,0,0]+=1.
try:local.build_local_graph(thin,thin_source,candidate_spec(thin_source,(20,18,15)),
 full_organ_mask=thin_organ,organ_depth=organ_depth_mm(thin_organ,thin.spacing),
 config=config,rng=np.random.default_rng(42),ct_clip=(-200.,250.),prepared_source=corrupted)
except ValueError as error:assert 'five-channel dense CT' in str(error),str(error)
else:raise AssertionError('Absent donor transport cannot change its dense CT input')
bad=graph.clone();bad[proposal.GRAPH_PROOF_KEY][1,0]=0
try:proposal.validate_batch_provenance(bad,graph_count=4,node_types=encoder.node_types,contract=contract)
except ValueError:pass
else:raise AssertionError('Unproven missing source must be rejected')
try:encoder._pool_context_shells('source_context',torch.empty((0,128)),torch.empty((0,16)),torch.empty(0,dtype=torch.long),2)
except ValueError:pass
else:raise AssertionError('Unbound direct source empty pool must remain rejected')
# Full official six-stage dense CT encoder and original twelve-field readout.
# Explicit UNIT synthetic CT patches, not actual patient or completed GNN weights.
source_patches=torch.from_numpy(np.stack([normal_built.source_patch,built.source_patch,
                                        source_only.source_patch,target_only.source_patch]))
target_patches=torch.from_numpy(np.stack([normal_built.target_patch,built.target_patch,
                                        source_only.target_patch,target_only.target_patch]))
with torch.no_grad():
 source_map,target_map=encoder.encode_dense_maps(source_patches,torch.arange(4),target_patches)
 fields=encoder.forward_graph(graph,source_map,target_map)
assert len(fields)==12 and all(value.shape==(4,128) and torch.isfinite(value).all() for value in fields.values())
for index in range(3):
 for row in (1,2):assert torch.equal(fields['source_c'+str(index)][row],encoder.empty_context_shell['source_context_c'+str(index)])
 for row in (1,3):assert torch.equal(fields['target_c'+str(index)][row],encoder.empty_context_shell['target_context_c'+str(index)])
audit=proposal.checkpoint_token_audit(encoder)
assert len(audit)==6 and all(item['equals_original_zero_initialization'] for item in audit.values())
assert all(item['nonzero_count']==0 and not item['evidence_of_having_learned_from_values'] for item in audit.values())
try:encoder.forward_graph(graph,source_map,target_map)
except ValueError:pass
else:raise AssertionError('Gradient-enabled use of inference proposal must be rejected')
encoder.train()
with torch.no_grad():
 try:encoder.forward_graph(graph,source_map,target_map)
 except ValueError:pass
 else:raise AssertionError('Training admission may not change through inference proposal')
print('source absence proposal original/STU CPU UNIT PASS')
'''


if __name__ == "__main__":
    unittest.main()
