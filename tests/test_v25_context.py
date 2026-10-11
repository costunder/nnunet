"""Explicit synthetic UNIT checks for the shared train/validation/CP profile.

No real CT, CUDA throughput, final training, or final evaluation is claimed.
The original-builder check runs in an isolated process because profile adapters
intentionally bind one original snapshot and policy for the process lifetime.
"""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

import numpy as np

from hiercp_v1x import v25_context as profile


ROOT = Path(__file__).resolve().parents[1]


class SharedContextProfileUNIT(unittest.TestCase):
    def test_explicit_selection_and_fresh_training_identity(self):
        identity = profile.identity()
        self.assertTrue(identity["enabled_by_default"])
        self.assertTrue(identity["same_train_validation_CP_definition"])
        self.assertTrue(identity["requires_fresh_training"])
        self.assertEqual(identity["context_inner_radius_mm"],0.)
        self.assertEqual(identity["context_outer_radius_mm"],10.)
        with self.assertRaises(PermissionError):
            profile.install({})

    def test_exact_open_tumor_boundary_closed_outer_and_open_liver_depth(self):
        shape = (3,3,3)
        footprint = np.zeros(shape,bool)
        organ = np.zeros(shape,bool); organ[1,1,1] = True
        outside = np.ones(shape,np.float32)
        depth = np.zeros(shape,np.float32)
        fields = SimpleNamespace(footprint=footprint,organ_mask=organ,outside_tumor_mm=outside,organ_depth=depth)
        spatial = SimpleNamespace(_field=lambda obj,*names:next(getattr(obj,name) for name in names if hasattr(obj,name)))
        for distance,value,admitted in ((0.,2.,False),(.0001,1.0001,True),(1.5,3.,True),
                                         (10.,2.,True),(10.001,4.,False),(.5,1.,False)):
            outside[1,1,1],depth[1,1,1] = distance,value
            context,surface,_ = profile.semantic_masks(fields,spatial)
            self.assertEqual(bool(context[1,1,1]),admitted)
            self.assertEqual(bool(surface[1,1,1]),0.<value<=3.)
        footprint[1,1,1] = True
        outside[1,1,1],depth[1,1,1] = 1.,2.
        context,surface,_ = profile.semantic_masks(fields,spatial)
        self.assertFalse(context.any()); self.assertFalse(surface.any())

    def test_nonfinite_geometry_and_all_exact_AST_transform_sites(self):
        spatial = SimpleNamespace(_field=lambda obj,*names:next(getattr(obj,name) for name in names if hasattr(obj,name)))
        for value in (float("nan"),float("inf"),-1.):
            fields = SimpleNamespace(footprint=np.zeros((2,2,2),bool),organ_mask=np.ones((2,2,2),bool),
                outside_tumor_mm=np.ones((2,2,2),np.float32),organ_depth=np.full((2,2,2),value,np.float32))
            with self.assertRaises(ValueError):
                profile.semantic_masks(fields,spatial)
        with ZipFile(ROOT / "versions/v1/pipeline_v1_source.zip") as archive:
            original = archive.read("hiercp/spatial.py").decode("utf8")
        functions = {node.name:node for node in ast.parse(original).body if isinstance(node,ast.FunctionDef)}
        self.assertEqual(profile._canonical_transform(functions["canonical_coordinate_sets"]),2)
        validator = functions["validate_canonical_coordinates"]
        self.assertEqual(profile._validator_transform(validator),5)
        changed = ast.unparse(validator)
        self.assertIn("distance <= 0.0",changed)
        self.assertIn("liver_depth[tuple(context.T)] <= 1.0",changed)
        self.assertNotIn("context_rows",changed)
        for rule in ("out-of-bounds","duplicate nodes","pure liver parenchyma","configured annulus","actual liver-surface band"):
            self.assertIn(rule,changed)
        target = {node.name:node for node in ast.parse((ROOT/"hiercp_v1x/transition_v1_empty_context.py").read_text()).body
                  if isinstance(node,ast.FunctionDef)}
        self.assertEqual(profile._target_proof_transform(target["geometry_proof"]),3)
        self.assertIn("outside > 0.0",ast.unparse(target["geometry_proof"]))
        self.assertEqual(profile._target_validator_transform(target["validate_local_proof"]),2)

    def test_original_preparation_materializer_full_STU_training_and_independent_donor(self):
        result = subprocess.run([sys.executable,"-X","utf8","-B","-c",_original_script_UNIT()],
            cwd=ROOT,capture_output=True,text=True,encoding="utf8",timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("v25 shared train/validation/CP original/STU UNIT PASS",result.stdout)


def _original_script_UNIT():
    # Reuse the extensive frozen original-builder fixture rather than duplicating
    # its source and making the original/profile comparison drift. Every policy
    # substitution below requires exactly one known site in that fixture.
    from tests.test_v24_context_depth1 import ORIGINAL_SCRIPT
    replacements = (
        ("from hiercp_v1x import v24_context_depth1 as profile", "from hiercp_v1x import v25_context as profile"),
        ("root=Path.cwd();snapshot=root/'work/v14_matched_learning_DEBUG_20261004_r3/source/v1.0'",
         "root=Path.cwd()\nfrom zipfile import ZipFile\nfrom pathlib import PurePosixPath\nimport uuid\n"
         "from hiercp_v1x.contracts import V1_ARCHIVE_SHA256\n"
         "archive=root/'versions/v1/pipeline_v1_source.zip'\n"
         "assert hashlib.sha256(archive.read_bytes()).hexdigest()==V1_ARCHIVE_SHA256\n"
         "snapshot=root/'outputs'/('v25_context_original_CPU_UNIT_'+uuid.uuid4().hex)\n"
         "snapshot.mkdir(parents=True,exist_ok=False)\n"
         "with ZipFile(archive) as zipped:\n"
         " for name in zipped.namelist():\n"
         "  if not ((name.startswith('hiercp/') and name.endswith('.py')) or name=='config/train.json'):continue\n"
         "  relative=PurePosixPath(name)\n"
         "  if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\\\' in name:raise ValueError('Unsafe archived UNIT source')\n"
         "  target=snapshot.joinpath(*relative.parts);target.parent.mkdir(parents=True,exist_ok=True)\n"
         "  with target.open('xb') as stream:stream.write(zipped.read(name))"),
        ("selected=profile.install(runtime,approved_profile=True)",
         "selected=profile.install(runtime,approved_profile=True)\nconfig=replace(config,context_inner_radius_mm=0.)\nbase_config['context_inner_radius_mm']=0.\n"
         "assert profile.install(runtime,approved_profile=True)==selected\n"
         "foreign=dict(runtime);foreign['scope']=copy.deepcopy(runtime['scope']);foreign['scope']['contract_sha256']='0'*64\n"
         "try:profile.install(foreign,approved_profile=True)\n"
         "except RuntimeError as error:assert 'cannot switch' in str(error)\n"
         "else:raise AssertionError('Foreign scope silently installed')"),
        ("assert 'missing fresh depth1 profile' in str(error)", "assert 'missing fresh v25 profile' in str(error)"),
        ("original.train()\nwith torch.inference_mode():\n try:original.forward_graph(batch,source_map,target_map)\n except ValueError as error:assert 'eval/no-grad inference only' in str(error)\n else:raise AssertionError('Training input policy was silently changed')\noriginal.eval()",
         "original.train()\nwith torch.inference_mode():\n training_outputs=original.forward_graph(batch,source_map,target_map)\nassert len(training_outputs)==12\noriginal.eval()\n"
         "encoder.train();encoder.zero_grad(set_to_none=True)\n"
         "training_outputs=encoder(batch,source_patch,torch.zeros(2,dtype=torch.long),target_patches)\n"
         "loss=sum((index+1)*value.float().square().mean() for index,value in enumerate(training_outputs.values()))\n"
         "loss.backward()\n"
         "missing=[name for name,value in encoder.named_parameters() if value.requires_grad and value.grad is None]\n"
         "nonfinite=[name for name,value in encoder.named_parameters() if value.grad is not None and not torch.isfinite(value.grad).all()]\n"
         "assert not missing,missing\nassert not nonfinite,nonfinite\n"
         "encoder.eval()"),
        ("print('depth1 original/STU CPU UNIT PASS')", "print('v25 shared train/validation/CP original/STU UNIT PASS')"),
    )
    result = ORIGINAL_SCRIPT
    for before,after in replacements:
        if result.count(before) != 1:
            raise ValueError("Frozen original profile UNIT fixture changed: " + before)
        result = result.replace(before,after)
    marker = "assert prepared.canonical_counts['source_context']>0"
    independent = r'''
from hiercp_v1x.v24_inputs import recipient_context,build_local_record
recipient=recipient_context('UNIT_independent_recipient',thin.image+3.,thin_organ,thin.spacing,thin.image_affine)
row=dict(id='UNIT_independent_query',case_id=recipient.case_id,center=[20,18,15],
 donor_case_id=thin.paths.case_id,donor_component=int(source.component_id),target=1,component=999)
independent_record=build_local_record(recipient,thin,source,prepared,row,config=config,seed=42,
 ct_clip=(-200.,250.),scope_contract=runtime['scope']['contract_sha256'])
assert independent_record['case_id']!=independent_record['donor_case_id']
assert independent_record['source_local'][profile.PROOF_KEY]['source_identity']['case_id']==thin.paths.case_id
assert independent_record['source_local'][profile.PROOF_KEY]['source_identity']['case_id']!=recipient.case_id
assert independent_record['source_local'][profile.PROOF_KEY]['minimum_liver_depth_exclusive_mm']==1.
assert independent_record['target_local'][profile.PROOF_KEY]['context_inner_radius_mm']==0.
'''
    if result.count(marker) != 1:
        raise ValueError("Frozen donor UNIT fixture changed")
    return result.replace(marker,marker+independent)


if __name__ == "__main__":
    unittest.main()
