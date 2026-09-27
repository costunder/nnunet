"""Explicit CPU counterexamples; not full CT/GPU training evidence."""
import ast
import copy
import itertools
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from tools.v22_runtime_receipts import with_semantics
from tools.v22_rank_objective import ranking_metrics
from tools.v22_rank_recommendation import rank_then_filter
from tools.v22_artifacts import best_snapshot,validate_best,validate_memory,tree_hash
from hiercp_v22.data import donor_in_target_spacing
from hiercp_v222.placement import PlacementSpec,validate_grid

ROOT=Path(__file__).resolve().parents[1]

@dataclass
class Donor:
    patch_mask: object
    patch_image: object
    anchor_center: tuple
    patch_slices: tuple
    voxel_count: int


class ReceiptTests(unittest.TestCase):
    def test_atomic_failure_keeps_previous_checkpoint(self):
        from hiercp_v222.v1_execution import atomic_torch
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'latest.pt';atomic_torch(path,{'step':1})
            def fail(value,stream):
                stream.write(b'partial');raise OSError('injected disk failure')
            with patch('torch.save',fail),self.assertRaises(OSError):atomic_torch(path,{'step':2})
            self.assertEqual(torch.load(path,weights_only=False),{'step':1})
            self.assertEqual(list(Path(root).glob('*.tmp')),[])
    def test_actual_process_main_installs_wrapper_for_new_and_reuse_lists(self):
        from tools import run_v222_process_runtime as runtime
        from hiercp_v222 import v1_execution as ex
        captured=[]
        payload=[dict(physical_batch=32,accepted=True,graphs_per_second=2,peak_vram_bytes=1,
                      executed=True,budget_bytes=2,nodes=3,edges=4,training=True)]
        before=copy.deepcopy(payload)
        def body():
            from tools import v222_runtime_execution as execution
            with execution.installed(dict(runtime_sha256=runtime.runtime_identity())):
                ex.write_new('memory_batch_calibration.json',payload)
                ex.write_new('training_batch_calibration.json',copy.deepcopy(payload))
                ex.write_new('initialization.json',dict(training_objective='observed_rank_v1'))
                with self.assertRaisesRegex(ValueError,'Conflicting'):
                    ex.write_new('bad.json',dict(training_objective='observation_ce'))
        with patch('tools.v222_resume_guard.assert_source_runs_idle'),patch.object(runtime.base,'main',body), \
             patch.object(ex,'write_new',lambda path,value:captured.append((str(path),value))), \
             patch.object(runtime.sys,'argv',['runtime']):
            runtime.main()
        self.assertEqual(payload,before);self.assertEqual(captured[0][1],payload)
        self.assertEqual(captured[1][1],payload);self.assertEqual(captured[2][1]['feature_coordinates'],'stride4')

    def test_no_input_mutation_or_conflict_overwrite(self):
        original={'a':1}
        self.assertEqual(with_semantics(original,{'b':2}),{'a':1,'b':2})
        self.assertEqual(original,{'a':1})
        with self.assertRaises(ValueError):with_semantics(original,{'a':2})


class RankingTieTests(unittest.TestCase):
    def test_two_positive_tie_has_one_actual_order(self):
        metrics,rows=ranking_metrics([1,1,0],[1,1,0],['A']*3,candidate_keys=['c','a','b'])
        self.assertEqual(rows[0]['observed_ranks'],[1,2])
        self.assertEqual(metrics['ranking_recall_at_1'],.5);self.assertEqual(metrics['ranking_mrr'],1)

    def test_all_ties_and_permutations_are_label_independent(self):
        reference=None
        for ids in itertools.permutations(range(3)):
            labels=[1,0,1];keys=['b','a','c']
            result=ranking_metrics([1]*3,[labels[i] for i in ids],['A']*3,candidate_keys=[keys[i] for i in ids])
            if reference is None:reference=result
            self.assertEqual(result,reference)

    def test_no_pairs_not_fake_zero_best(self):
        for truth in ([1,1],[0,0]):
            with self.assertRaises(ValueError):ranking_metrics([1,2],truth,['A']*2,candidate_keys=['a','b'])


class GeometryTests(unittest.TestCase):
    def test_rotated_scaled_spec_uses_same_integer_voxels_for_filter_paste(self):
        from hiercp_v222.placement import placement_spec
        from hiercp_v222.v1_cache import configuration
        from hiercp.schema import graph_config_from_dict
        _,base=configuration();cfg=graph_config_from_dict(base['graph'])
        mask=np.zeros((7,7,7),bool);mask[2:5,2:4,2:6]=True
        source=SimpleNamespace(component_id=1,patch_mask=mask,patch_image=np.arange(343,dtype=np.float32).reshape(mask.shape),anchor_center=(3,3,3),patch_slices=(slice(0,7),)*3)
        spacing=np.array([.7,1.3,2.1]);affine=np.diag([*spacing,1.])
        case=SimpleNamespace(paths=SimpleNamespace(case_id='r'),image=np.zeros((40,40,40)),label=np.ones((40,40,40),dtype=np.uint8),spacing=spacing,image_affine=affine,label_affine=affine)
        angle=.41;rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        for transform in [rotation,rotation@np.diag([1.2,.8,1.1]),np.diag([-1.,1.,1.])]:
            spec=placement_spec(case,source,(20,20,20),'d',forward_mm=transform,graph_config=cfg)
            decision=rank_then_filter([1],[spec.center],spec.mask,spec.anchor,case.label,min_liver_coverage=1)
            self.assertFalse(decision['keep_original'])
            image,label=spec.paste(case.image,case.label)
            self.assertEqual(set(map(tuple,np.argwhere(label==2))),set(map(tuple,spec.coordinates())))
            self.assertTrue(np.array_equal(image[label==2],spec.image[spec.mask]))

    def test_regrid_padding_preserves_every_relative_voxel(self):
        from scipy import ndimage as ndi
        for shape,ds,ts in itertools.product([(6,6,6),(5,7,4)],[(2.,2.,2.),(1.1,2.3,.7)],[(1.,1.,1.),(1.4,.8,2.1)]):
            with self.subTest(shape=shape,ds=ds,ts=ts):
                anchor=np.array(shape)//2;mask=np.zeros(shape,bool);mask[tuple(anchor)]=True;mask[1:3,1:4,1:3]=True
                source=Donor(mask,np.arange(np.prod(shape),dtype=np.float32).reshape(shape),tuple(anchor),(slice(0,shape[0]),slice(0,shape[1]),slice(0,shape[2])),int(mask.sum()))
                ds=np.array(ds);ts=np.array(ts)
                before=np.ceil(anchor*ds/ts).astype(int);after=np.ceil((np.array(shape)-1-anchor)*ds/ts).astype(int)
                coordinates=(np.indices(tuple(before+after+1))-before[:,None,None,None])*ts[:,None,None,None]/ds[:,None,None,None]+anchor[:,None,None,None]
                oracle=ndi.map_coordinates(mask.astype(np.uint8),coordinates,order=0,mode='constant',cval=0)>0
                result,_=donor_in_target_spacing(source,ds,ts)
                self.assertTrue(np.array_equal(np.array(result.anchor_center),np.array(result.patch_mask.shape)//2))
                self.assertEqual(set(map(tuple,np.argwhere(oracle)-before)),set(map(tuple,np.argwhere(result.patch_mask)-result.anchor_center)))

    def test_grid_rejects_affine_nan_and_spacing_mismatch(self):
        case=SimpleNamespace(image=np.zeros((3,3,3)),label=np.ones((3,3,3)),spacing=np.ones(3),image_affine=np.eye(4),label_affine=np.eye(4))
        validate_grid(case)
        for attribute,value in [('label_affine',np.diag([1.,1.,-1.,1.])),('spacing',np.array([1.,np.nan,1.])),('spacing',np.array([2.,1.,1.]))]:
            bad=copy.deepcopy(case);setattr(bad,attribute,value)
            with self.assertRaises(ValueError):validate_grid(bad)

    def test_same_spec_filter_and_exact_paste_with_bbox_outside(self):
        mask=np.zeros((5,5,5),bool);mask[2,2,2]=True;mask[3,2,2]=True
        spec=PlacementSpec('r','d',1,(0,1,1),(1.,1.,1.),tuple(map(tuple,np.eye(4))),np.full(mask.shape,77),mask,(2,2,2),tuple(map(tuple,np.eye(3))))
        image=np.zeros((4,4,4));label=np.ones(image.shape,dtype=np.uint8)
        result=rank_then_filter([1],[spec.center],spec.mask,spec.anchor,label,min_liver_coverage=1)
        self.assertFalse(result['keep_original'])
        pasted,seg=spec.paste(image,label)
        self.assertEqual(set(map(tuple,np.argwhere(seg==2))),set(map(tuple,spec.coordinates())))
        self.assertTrue(np.all(pasted[seg==2]==77));self.assertFalse(np.any(image));self.assertFalse(np.any(label==2))

    def test_empty_noop_and_nan_error(self):
        self.assertTrue(rank_then_filter([],[],None,None,None,min_liver_coverage=1)['keep_original'])
        with self.assertRaises(ValueError):rank_then_filter([np.nan],[[0,0,0]],None,None,None,min_liver_coverage=1)


class CheckpointTests(unittest.TestCase):
    def test_typed_final_artifact_rejects_each_corruption(self):
        import json
        from tools.v22_artifacts import ARTIFACT_CONTRACT,validate_artifact
        from tools.v22_rank_objective import OBJECTIVE,configuration as objective
        from tools.run_v222_process_runtime import runtime_identity
        from hiercp_v222.v1_cache import provenance,configuration
        from hiercp_v222.v1_training import CHECKPOINT_FORMAT
        from hiercp_v222.placement import GEOMETRY_CONTRACT
        split=json.loads((ROOT/'config/split_cp80_fold0.json').read_text())
        recipient,donor=split['inner_train'][:2]
        identities=dict(format='hiercp_public_case_benchmark_v1',independence_scope='published_case_only',
            patient_independence_verified=False,annotation_scope='provided_masks_may_omit_lesions',
            cases={c:dict(patient_group='case:'+c,identity_basis='published_case_id_only',annotation_complete=None)
                   for c in split['outer_train']+split['outer_val']})
        row=dict(id='synthetic:0',case_id=recipient,patient_group='case:'+recipient,target=1,
                 donor_case_id=donor,donor_component=1,donor_group='case:'+donor)
        meta=dict(split=split,identities=identities,donor_pool=[dict(case_id=donor,component_id=1)],raw_records=[])
        memory=dict(embeddings=torch.ones((1,128)),owners=torch.tensor([0]),classes=torch.tensor([row['target']]),
            case_ids=[row['case_id']],patient_groups=[row['patient_group']],donor_groups=[row['donor_group']],record_ids=[row['id']])
        cfg,base=configuration();weights={'weight':torch.ones(2)}
        value=dict(format=CHECKPOINT_FORMAT,artifact_contract=ARTIFACT_CONTRACT,artifact_kind='final',debug=True,
            training_objective=OBJECTIVE,ranking_contract=objective(),feature_coordinates='stride4',support_task_contract='patient_group_v1',
            geometry_contract=GEOMETRY_CONTRACT,run_id='synthetic-test',config=dict(cfg,gnn_epochs=1),base=base,
            source_identity=provenance(),execution_policy_runtime_sha256=runtime_identity(),completed_epochs=1,selected_epoch=1,
            state_dict=weights,memory=memory,support_records=[row],identities=meta['identities'],split=meta['split'],
            donor_pool=meta['donor_pool'],raw_records=meta['raw_records'],selected_model_sha256=tree_hash(weights),
            memory_model_sha256=tree_hash(weights),memory_sha256=tree_hash(memory))
        validate_artifact(value,'final',allow_debug=True)
        with self.assertRaisesRegex(ValueError,'DEBUG'):validate_artifact(value,'final')
        for key in list(value):
            if key in ('memory_sha256','identities','support_records','donor_pool','run_id','execution_policy_runtime_sha256'):
                bad=copy.deepcopy(value);del bad[key]
                with self.subTest(missing=key),self.assertRaises(ValueError):validate_artifact(bad,'final',allow_debug=True)
        for key,replacement in [('format','wrong'),('completed_epochs',0),('selected_epoch',2),('execution_policy_runtime_sha256',{}),('memory_model_sha256','wrong')]:
            bad=copy.deepcopy(value);bad[key]=replacement
            with self.subTest(changed=key),self.assertRaises(ValueError):validate_artifact(bad,'final',allow_debug=True)

    def test_best_is_cpu_immutable_and_run_bound(self):
        network=torch.nn.Linear(3,2);identity={'config':{'gnn_epochs':40},'run_id':'A'}
        snapshot=best_snapshot(network,2,.4,identity);before=tree_hash(snapshot['state_dict'])
        with torch.no_grad():network.weight.add_(1)
        self.assertEqual(tree_hash(snapshot['state_dict']),before);validate_best(snapshot,identity)
        with self.assertRaises(ValueError):validate_best(snapshot,dict(identity,run_id='B'))
        snapshot['state_dict']['weight'][0,0]+=1
        with self.assertRaises(ValueError):validate_best(snapshot,identity)

    def fixture(self):
        identities={'cases':{'a':{'patient_group':'A'},'b':{'patient_group':'B'},'held':{'patient_group':'H'}}}
        rows=[dict(id='a:0',case_id='a',target=1,patient_group='A',donor_case_id='b',donor_component=1,donor_group='B')]
        memory=dict(embeddings=torch.ones((1,128)),owners=torch.tensor([0]),classes=torch.tensor([1]),case_ids=['a'],patient_groups=['A'],donor_groups=['B'],record_ids=['a:0'])
        return memory,rows,identities,{'inner_train':['a','b']},[dict(case_id='b',component_id=1)]

    def test_memory_accepts_exact_coverage(self):validate_memory(*self.fixture())

    def test_memory_rejects_metadata_corruption(self):
        for change in ('donor','group','owner','nan','coverage','dtype'):
            with self.subTest(change=change):
                m,r,i,s,p=self.fixture()
                if change=='donor':r[0]['donor_case_id']='held'
                if change=='group':m['patient_groups']=['B']
                if change=='owner':m['owners'][0]=1
                if change=='nan':m['embeddings'][0,0]=float('nan')
                if change=='coverage':m['record_ids']=['other']
                if change=='dtype':m['embeddings']=m['embeddings'].double()
                with self.assertRaises(ValueError):validate_memory(m,r,i,s,p)


class PairAuditTests(unittest.TestCase):
    def test_unequal_pair_weights_are_reported_not_normalized_away(self):
        from tools.v22_pair_audit import pair_audit
        rows=[dict(case_id='A',target=t,bounds={'edges':e}) for t,e in [(1,10),(0,20),(0,30)]]
        report=pair_audit(rows,[[0,1],[2]])
        case=report['cases'][0]
        self.assertEqual((case['coefficient_min'],case['coefficient_max']),(.5,1.5))
        self.assertEqual((case['occurrence_min'],case['occurrence_max']),(1,2))
        self.assertTrue(report['estimator_unchanged'])
        with self.assertRaises(ValueError):pair_audit(rows,[[0,1]])


if __name__=='__main__':unittest.main()
