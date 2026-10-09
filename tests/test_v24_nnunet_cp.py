"""Mechanical UNIT contracts only; no claimed patient/model training."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io
import subprocess
import sys
import ast
import hashlib
from collections import OrderedDict
import numpy as np
from hiercp_v1x import v24_nnunet_cp as cp


def split_fixture():
    train=[f'UNIT_train_{i}' for i in range(84)]
    val=[f'UNIT_inner_{i}' for i in range(21)]
    return dict(inner_train=train,inner_val=val,outer_train=train+val,
                outer_val=[f'UNIT_outer_{i}' for i in range(26)])


def pin_fixture():
    rows=[]
    for gpu,score in ((1,.3),(5,.4),(6,.2)):
        rows.append(dict(gpu=gpu,best_path=f'/UNIT_gpu{gpu}/training/checkpoint_best.pt',
            best_file_sha256='a'*64,best_content_sha256='b'*64,best_model_sha256='c'*64,identity_sha256='d'*64,
            best_record=dict(epoch=15,updates=178,selection_key=[score,.2,-.4],
                candidate_universe='all native P + fixed128U',selected_by='full_validation_only')))
    return dict(format=cp.PIN_FORMAT,all_three_paused=True,recipient_annotation_exposed=True,
        selection_metric='fixed full128 per-P patient-macro MRR, top1, negativepairloss',
        candidates=rows,selected=copy.deepcopy(rows[1]))


class NativeContracts(unittest.TestCase):
    def test_complete_nested_split(self):
        split=split_fixture()
        result=cp.validate_split(split)
        self.assertEqual(len(result['outer_train']),105)
        result['inner_train'][0]='different'
        self.assertNotEqual(result,split)

    def test_held_out_recipient_leakage_rejected(self):
        split=split_fixture();split['outer_val'][0]=split['outer_train'][0]
        with self.assertRaisesRegex(ValueError,'leakage'):cp.validate_split(split)

    def test_incomplete_dataset_rejected(self):
        split=split_fixture();split['inner_train'].pop()
        with self.assertRaises(ValueError):cp.validate_split(split)

    def test_best_pin_recomputes_comparison_winner(self):
        pin=pin_fixture();self.assertEqual(cp.validate_pin(pin)['selected']['gpu'],5)
        pin['selected']=pin['candidates'][0]
        with self.assertRaisesRegex(ValueError,'actual best'):cp.validate_pin(pin)

    def test_stage_validation_and_unpaused_pin_rejected(self):
        for key,value in (('all_three_paused',False),('recipient_annotation_exposed',False)):
            pin=pin_fixture();pin[key]=value
            with self.assertRaises(ValueError):cp.validate_pin(pin)
        pin=pin_fixture();pin['candidates'][0]['best_record']['selected_by']='stage_validation'
        with self.assertRaises(ValueError):cp.validate_pin(pin)

    def test_nonfinite_selection_key_rejected(self):
        pin=pin_fixture();pin['candidates'][0]['best_record']['selection_key'][0]=float('nan')
        with self.assertRaises(ValueError):cp.validate_pin(pin)

    def test_joint_P_retained_U_bank_mapping(self):
        bank=[f'UNIT_U_{i}' for i in range(128)]
        rows=[dict(id='UNIT_P',center=[1,1,1])]+[dict(id=name,center=[i+2,3,4]) for i,name in enumerate(bank)]
        plan=SimpleNamespace(record_ids=('UNIT_P',*bank),rows=rows,bank_record_ids=tuple(reversed(bank)),
            unobserved_indices=tuple(range(1,129)),unobserved_bank_positions=tuple(reversed(range(128))))
        values,centers=cp.bank_u_values(np.arange(129),plan)
        self.assertEqual(values.tolist(),list(range(128,0,-1)))
        self.assertEqual(centers[0].tolist(),[129,3,4])
        self.assertNotIn(0,values)

    def test_nonfinite_or_incomplete_CP_scores_rejected(self):
        plan=SimpleNamespace(record_ids=tuple(range(129)))
        with self.assertRaises(ValueError):cp.bank_u_values(np.zeros(8),plan)
        values=np.zeros(129);values[0]=float('nan')
        with self.assertRaises(ValueError):cp.bank_u_values(values,plan)

    def test_historical_baseline_split_and_patch_exact(self):
        with tempfile.TemporaryDirectory(prefix='v24_UNIT_',dir=Path.cwd()) as temp:
            pre=Path(temp)/'Dataset730_LiverOnlineCP_OF0';pre.mkdir();(pre/'UNIT_arrays').mkdir()
            split=split_fixture()
            documents={'splits_final.json':[dict(train=split['outer_train'],val=split['outer_val'])],
                'dataset.json':{},'dataset_fingerprint.json':{},cp.PLANS+'.json':
                {'configurations':{'3d_fullres':{'patch_size':[128]*3,'batch_size':2,
                    'architecture':{'network_class_name':'UNIT.ResidualEncoderUNet'},'data_identifier':'UNIT_arrays'}}}}
            for name,value in documents.items():(pre/name).write_text(json.dumps(value),encoding='utf8')
            actual=cp.validate_baseline(pre,split)
            self.assertEqual(actual['epochs'],250);self.assertEqual(actual['cp_probability'],.5)
            self.assertIn('not retrospectively certified',actual['normalization_fitting_scope'])
            plans=documents[cp.PLANS+'.json'];plans['configurations']['3d_fullres']['patch_size']=[64]*3
            (pre/(cp.PLANS+'.json')).write_text(json.dumps(plans),encoding='utf8')
            with self.assertRaises(ValueError):cp.validate_baseline(pre,split)


class ActualPastePredicates(unittest.TestCase):
    def fixture(self):
        case=SimpleNamespace(image=np.zeros((20,20,20),np.float32),label=np.ones((20,20,20),np.int16),spacing=np.ones(3))
        source=SimpleNamespace(patch_mask=np.ones((3,3,3),bool),voxel_count=27)
        centers=np.array([[x,y,z] for x in range(3,11) for y in range(3,7) for z in range(3,7)],np.int64)
        settings=dict(occupied_clearance_vox=0,min_center_separation_mm=0.,min_center_separation_vox=0.,min_liver_coverage=1.)
        return case,source,centers,settings

    def test_all128_full_footprints_are_checked(self):
        case,source,centers,settings=self.fixture()
        result=cp.verify_cp_centers(case,source,centers,settings)
        self.assertEqual(result['source_occupied_voxels'],27)
        self.assertTrue(result['all128_full_footprints_valid'])

    def test_late_candidate_tumor_overlap_not_silently_ignored(self):
        case,source,centers,settings=self.fixture()
        case.label[tuple(centers[-1])]=2
        result=cp.verify_cp_centers(case,source,centers,settings)
        self.assertFalse(result['eligible_mask'][-1])
        self.assertIn('existing_tumor',[row['reason'] for row in result['invalid_candidate_reasons']][-1])
        self.assertGreater(result['eligible_count'],0)

    def test_complete_bounds_and_liver_coverage(self):
        case,source,centers,settings=self.fixture();centers[-1]=[0,0,0]
        result=cp.verify_cp_centers(case,source,centers,settings)
        self.assertEqual(result['invalid_candidate_reasons'][-1]['reason'],'complete_footprint_outside_CT')
        case,source,centers,settings=self.fixture();case.label[tuple(centers[-1])]=0
        result=cp.verify_cp_centers(case,source,centers,settings)
        self.assertIn('insufficient_liver_coverage',[row['reason'] for row in result['invalid_candidate_reasons']])

    def test_center_clearance_is_enforced(self):
        case,source,centers,settings=self.fixture();case.label[14,6,6]=2
        settings['min_center_separation_mm']=5.
        result=cp.verify_cp_centers(case,source,centers,settings)
        self.assertIn('center_separation_mm',[row['reason'] for row in result['invalid_candidate_reasons']])

    def test_all_invalid_reports_audit_and_never_falls_back(self):
        case,source,centers,settings=self.fixture();case.label[:]=0
        with self.assertRaisesRegex(ValueError,'No physically eligible.*no recipient skipped') as error:
            cp.verify_cp_centers(case,source,centers,settings)
        self.assertIn('"eligible_count": 0',str(error.exception))
        self.assertIn('"invalid_candidates": 128',str(error.exception))

    def test_eligible_argmax_preserves_full_scores_and_stable_tie_order(self):
        values=np.arange(128,dtype=np.float32);values[6]=values[7]=200;values[127]=300
        mask=np.ones(128,bool);mask[127]=False
        original=values.copy();self.assertEqual(cp.eligible_argmax(values,mask),6)
        np.testing.assert_array_equal(values,original)
        with self.assertRaisesRegex(ValueError,'No eligible'):cp.eligible_argmax(values,np.zeros(128,bool))


class FrozenLoaderIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Real production parent definitions; only external framework setup is
        # omitted. The actual parent's RNG sampler and raw-paste plan run here.
        cls.namespace=dict(np=np,torch=__import__('torch'),hashlib=hashlib,json=json,
            OrderedDict=OrderedDict,Path=Path,nnUNetDataLoader=object,
            TRAINER_FORMAT='hiercp_online_trainer_v2',BANK_FORMAT='hiercp_online_bank_v2')
        original=ast.parse((cp.ROOT/'custom_trainers/nnUNetTrainer_OnlinePairedCP.py').read_text(encoding='utf8'))
        names={'OnlineCPError','OnlineCPBank','_stable_u64','_stable_seed','_select_candidate_index','_anchored_slices','nnUNetDataLoaderOnlineCP'}
        nodes=[node for node in original.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
        if set(node.name for node in nodes)!=names:raise AssertionError('Actual parent definitions missing')
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),'actual_parent_loader','exec'),cls.namespace)
        cls.namespace.update(eligible_argmax=cp.eligible_argmax,CP_SELECTION=cp.CP_SELECTION,sha=cp.sha,read=cp.read)
        tree=ast.parse((cp.ROOT/'custom_trainers/nnUNetTrainer_FrozenV23CP.py').read_text(encoding='utf8'))
        nodes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in ('FrozenV23Bank','FrozenV23Loader')]
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual_FrozenV23Loader','exec'),cls.namespace)

    def test_parent_sampler_keeps_five_draws_and_selects_admitted_raw_payload(self):
        loader=self.namespace['FrozenV23Loader'].__new__(self.namespace['FrozenV23Loader'])
        values=np.arange(128,dtype=np.float32);mask=np.ones(128,np.uint8);mask[127]=0
        centers=np.stack([np.arange(128)+2,np.full(128,3),np.full(128,4)],axis=1)
        entry=dict(case_id=np.asarray(['UNIT_CASE']),scores=values,candidate_centers=centers,
            candidate_eligibility=mask,selected_candidate=np.asarray([126]),selection_policy=np.asarray([cp.CP_SELECTION]))
        class Bank:
            cp_probability=.5;intensity_scale=(.95,1.05);intensity_shift_hu=(-5.,5.)
            source_slots_by_case=None;paste_contract='onlinecp_raw_target_paste_v1';hier_top_k=1
            entries_by_case={'UNIT_CASE':['UNIT_entry']}
            def entry_names(self,case):return self.entries_by_case[case]
            def load_for_case(self,case,index):return entry
            def load_raw_candidate(self,actual,index):
                self.actual_index=index
                return {'metadata':{'preprocessed_shape':[200,200,200]}},{'output_bbox':np.asarray([[0,2],[0,2],[0,2]])}
        class RNG:
            count=0
            def random(self):self.count+=1;return [0.1,.4,.9,.2,.8][self.count-1]
        loader.online_bank=Bank();loader.online_epoch=3;loader.online_policy='hier_argmax'
        rng=RNG();loader._rng=lambda:rng
        plan,token=loader._sample_paste_plan('UNIT_CASE')
        self.assertEqual(rng.count,5);self.assertEqual(plan['candidate_index'],126)
        self.assertEqual(loader.online_bank.actual_index,126)
        self.assertEqual(plan['center'],tuple(centers[126]));self.assertAlmostEqual(plan['scale'],.97)
        self.assertAlmostEqual(plan['shift_hu'],3.)
        self.assertEqual(len(plan['entry']['scores']),128);self.assertIsInstance(token,int)

    def test_actual_frozen_bank_rejects_changed_scores_mask_and_wrong_payload_index(self):
        with tempfile.TemporaryDirectory(prefix='v24_payload_UNIT_',dir=Path.cwd()) as temp:
            root=Path(temp);(root/'scores').mkdir();case='UNIT_CASE';scores=np.arange(128,dtype=np.float32)
            mask=np.ones(128,np.uint8);mask[127]=0
            centers=np.stack([np.arange(128)+2,np.full(128,3),np.full(128,4)],axis=1)
            path=root/'scores'/(case+'.json');path.write_text(json.dumps(dict(scores=scores.tolist(),centers=centers.tolist())),encoding='utf8')
            checksum=cp.sha(path)
            bank=self.namespace['FrozenV23Bank'].__new__(self.namespace['FrozenV23Bank'])
            bank.root=root;bank.paste_contract='onlinecp_raw_target_paste_v1'
            bank.metadata=dict(split=dict(outer_train=[case]),CP_audits={case:dict(donor_case_id='UNIT_DONOR',donor_component=2,
                selected_candidate=126,eligibility=dict(eligible_mask=mask.astype(bool).tolist()),scores_file_sha256=checksum)})
            bank.v24_score_manifest=dict(score_files={case:dict(path='scores/'+case+'.json',sha256=checksum)})
            entry=dict(case_id=np.asarray([case]),donor_case_id=np.asarray(['UNIT_DONOR']),donor_component_id=np.asarray([2]),
                candidate_centers=centers.copy(),candidate_raw_centers=centers.copy(),scores=scores.copy(),source_component=np.asarray([2]),
                source_diameter_mm=np.asarray([5.]),candidate_eligibility=mask.copy(),selection_policy=np.asarray([cp.CP_SELECTION]),
                selected_candidate=np.asarray([126]),paste_contract=np.asarray([bank.paste_contract]),selected_payload=np.asarray(['UNIT_payload']),
                selected_payload_sha256=np.asarray(['a'*64]),raw_case_reference=np.asarray(['UNIT_reference']),raw_case_reference_sha256=np.asarray(['b'*64]))
            bank._validate_raw_entry(entry,root/'UNIT.npz')
            changed=copy.deepcopy(entry);changed['scores'][126]-=1
            with self.assertRaisesRegex(ValueError,'score/center'):bank._validate_raw_entry(changed,root/'UNIT.npz')
            changed=copy.deepcopy(entry);changed['candidate_eligibility'][127]=1
            with self.assertRaisesRegex(ValueError,'argmax admission'):bank._validate_raw_entry(changed,root/'UNIT.npz')
            with self.assertRaisesRegex(ValueError,'eligible original128'):bank.load_raw_candidate(entry,127)


class ScorePublication(unittest.TestCase):
    def fixture(self):
        split=split_fixture();cases=split['outer_train'];pin=pin_fixture()
        # Actual admission needs only the same explicit original plan fields.
        plans={}
        for case in cases:
            bank=tuple(case+'_U_'+str(i) for i in range(128))
            plans[case]=SimpleNamespace(record_ids=bank,rows=[dict(id=name,center=[i+1,2,3]) for i,name in enumerate(bank)],
                bank_record_ids=bank,unobserved_indices=tuple(range(128)),unobserved_bank_positions=tuple(range(128)),
                donor_case_id=cases[1] if case!=cases[1] else cases[0],donor_component=1,observed_P=0)
        population=SimpleNamespace(partition_cases=lambda partition:split[partition],case=lambda case,count:plans[case],
            manifest=lambda:dict(UNIT_explicit_population=True))
        rows=[dict(case_id=case,donor_case_id=plans[case].donor_case_id,donor_component=1,scores=list(range(128)),
            centers=[row['center'] for row in plans[case].rows],joint_observed_P=0,joint_candidates=128,
            all128U_scored=True,all_P_in_joint_context=True,model_sha256=pin['selected']['best_model_sha256']) for case in cases]
        return population,pin,rows

    def test_full_score_admission_rejects_missing_mutated_donor_center_and_nonfinite(self):
        population,pin,rows=self.fixture();self.assertEqual(len(cp.validate_score_rows(rows,population,pin)),105)
        with self.assertRaises(ValueError):cp.validate_score_rows(rows[:-1],population,pin)
        for key,value in [('donor_component',2),('centers',[[0,0,0]]*128),('scores',[float('nan')]*128)]:
            changed=copy.deepcopy(rows);changed[0][key]=value
            with self.assertRaises(ValueError):cp.validate_score_rows(changed,population,pin)

    def test_complete_scores_are_sealed_before_materialization_and_bound_to_bytes(self):
        population,pin,rows=self.fixture()
        with tempfile.TemporaryDirectory(prefix='v24_score_UNIT_',dir=Path.cwd()) as temp:
            root=Path(temp);(root/'scores').mkdir()
            request=dict(inventory_sha256='a'*64,baseline={'UNIT':True},source_files_sha256={'UNIT':'b'*64})
            (root/'request.json').write_text(json.dumps(request),encoding='utf8')
            for row in rows:(root/'scores'/(row['case_id']+'.json')).write_text(json.dumps(row),encoding='utf8')
            manifest=cp.seal_scores(root,rows,population,pin)
            self.assertTrue(manifest['complete']);self.assertEqual(len(manifest['score_files']),105)
            self.assertFalse(manifest['raw_materialization_started'])
            row=rows[0];path=root/'scores'/(row['case_id']+'.json');path.write_text('{}',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'Persisted complete score'):cp.seal_scores(root,rows,population,pin)

    def test_scoring_ast_excludes_only_new_publication_and_rejects_changed_inference(self):
        text=(cp.ROOT/cp.FILES[0]).read_text(encoding='utf8')
        without=text.replace('    seal_scores(output,rows,population,pin)\n','')
        self.assertEqual(cp._scoring_core_ast(text),cp._scoring_core_ast(without))
        changed=without.replace('physical_candidate_batch=32','physical_candidate_batch=8',1)
        self.assertNotEqual(cp._scoring_core_ast(text),cp._scoring_core_ast(changed))

    def test_lossless_publisher_wiring_preserves_original_numerical_code_objects(self):
        from hiercp_v1x import v24_lossless_raw_storage as lossless
        original=cp._root_raw_preparation()
        adapted=cp.lossless_raw_case_preparer()
        self.assertIs(adapted.__code__,original.prepare_raw_case.__code__)
        inner=adapted.__globals__['_prepare_raw_case']
        self.assertIs(inner.__code__,original._prepare_raw_case.__code__)
        self.assertIs(inner.__globals__['save_case'],lossless.save_case)
        self.assertIsNot(original._prepare_raw_case.__globals__['save_case'],lossless.save_case)
        self.assertEqual(adapted.__kwdefaults__,original.prepare_raw_case.__kwdefaults__)
        self.assertEqual(inner.__kwdefaults__,original._prepare_raw_case.__kwdefaults__)

    def test_fresh_raw_helper_resolution_ignores_archived_custom_package_shadow(self):
        script='''import sys,types,json
from pathlib import Path
from hiercp_v1x import v24_nnunet_cp as cp
shadow=types.ModuleType('custom_trainers');shadow.__path__=['/DEBUG_archived_source/custom_trainers']
sys.modules['custom_trainers']=shadow
for name in ('onlinecp_raw_bank','onlinecp_raw_resampling'):
 module=types.ModuleType('custom_trainers.'+name)
 module.__file__='/DEBUG_archived_source/custom_trainers/'+name+'.py'
 sys.modules[module.__name__]=module
helper=cp._root_raw_preparation();adapted=cp.lossless_raw_case_preparer()
from hiercp_v1x import v24_lossless_raw_storage as storage
assert Path(storage.original.__file__).resolve()==(cp.ROOT/'custom_trainers/onlinecp_raw_bank.py').resolve()
assert Path(helper.prepare_case.__code__.co_filename).resolve()==(cp.ROOT/'custom_trainers/onlinecp_raw_resampling.py').resolve()
assert Path(helper.run_case_jobs.__code__.co_filename).resolve()==(cp.ROOT/'hiercp/preparation_runtime.py').resolve()
assert sys.modules['custom_trainers'] is shadow
assert adapted.__globals__['_prepare_raw_case'].__globals__['save_case'] is storage.save_case
print(json.dumps(dict(ROOT_math_and_publication_pinned=True,archived_namespace_untouched=True)))
'''
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=cp.ROOT,capture_output=True,text=True,check=True)
        self.assertTrue(json.loads(result.stdout)['archived_namespace_untouched'])

    def test_new_storage_binding_does_not_require_new_file_in_old_scoring_release(self):
        self.assertEqual(len(cp.LEGACY_FILES),3);self.assertEqual(len(cp.LOSSLESS_SOURCE_FILES),4)
        self.assertEqual(cp.FILES[:3],cp.LEGACY_FILES)
        self.assertEqual(cp.FILES[3],'hiercp_v1x/v24_lossless_raw_storage.py')
        self.assertTrue(set(cp.RAW_HELPER_FILES)<=set(cp.FILES))

    def test_raw_scheduler_admits_all_cases_with_conservative_measured_memory(self):
        # Mechanical DEBUG resource observations, not measured patient memory.
        tree=ast.parse((cp.ROOT/cp.FILES[0]).read_text(encoding='utf8'))
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_materialize_bank')
        invocation=next(node for node in ast.walk(function) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Name) and node.func.id=='bounded_run')
        mode=next(item.value for item in invocation.keywords if item.arg=='workers')
        self.assertEqual(ast.literal_eval(mode),'auto')
        runtime=cp._pinned_root_module('hiercp/preparation_runtime.py')
        class DebugMeasurement:
            def __enter__(self):
                self.before={'rss_bytes':0};self.peak_rss=32*2**30
                self.report={'elapsed_seconds':1.,'status':'complete'}
                return self
            def __exit__(self,*exception):return False
        completed=[]
        with tempfile.TemporaryDirectory(prefix='v24_scheduler_DEBUG_',dir=Path.cwd()) as temp:
            with patch.dict(runtime.run_case_jobs.__globals__,
                    snapshot=lambda:dict(cpu_capacity=4,available_memory_bytes=48*2**30),
                    Measurement=DebugMeasurement),contextlib.redirect_stdout(io.StringIO()):
                report=runtime.run_case_jobs(tasks=range(5),function=lambda case:case,
                    commit=completed.append,workers=ast.literal_eval(mode),report_path=Path(temp)/'resources.json')
        self.assertEqual(sorted(completed),list(range(5)))
        self.assertEqual(report['status'],'complete')
        self.assertEqual(report['applied_worker_counts'],[1]*5)
        self.assertEqual(report['configured_tasks'],report['completed_tasks'])

    def test_lossless_case_admission_requires_exact_original_dtypes_and_full_voxel_proof(self):
        volumes={name:dict(dtype=dtype,shape=[1,2,3,4],verified_voxels=24,
            uncompressed_bytes=24*size,stored_bytes=80,
            source_tile_bytes_sha256='a'*64,decoded_tile_bytes_sha256='a'*64)
            for name,dtype,size in [('baseline_unclipped','<f8',8),('baseline_seg','<i2',2)]}
        proof=dict(format=cp.RAW_STORAGE,volume_count=2,volumes=volumes,
            full_precision_volume_roundtrip_verified=True,whole_volume_runtime_decode=False,candidate_storage_unchanged=True,
            compression=dict(lossy_filters=False,original_dtype_preserved=True),verified_voxels=48,
            original_volume_bytes=240,compressed_volume_file_bytes=160)
        self.assertEqual(cp.validate_lossless_case_receipt({'lossless_storage':proof})['verified_voxels'],48)
        for mutation in [lambda value:value['volumes']['baseline_unclipped'].update(dtype='<f4'),
                lambda value:value['volumes']['baseline_seg'].update(dtype='|u1'),
                lambda value:value['volumes']['baseline_unclipped'].update(decoded_tile_bytes_sha256='b'*64),
                lambda value:value.update(verified_voxels=24),lambda value:value.update(whole_volume_runtime_decode=True)]:
            altered=copy.deepcopy(proof);mutation(altered)
            with self.assertRaises(ValueError):cp.validate_lossless_case_receipt({'lossless_storage':altered})


class CLITests(unittest.TestCase):
    def test_fresh_prepare_activates_original_before_population_or_outputs(self):
        # Run actual prepare_bank imports in a fresh interpreter: earlier
        # geometry tests intentionally import current hiercp.common.
        script='''import json,sys
from unittest.mock import patch
from hiercp_v1x import v24_nnunet_cp as cp
from hiercp_v1x import comparison_native_upper_cache as upper
from tools import local_cnn_device
class OriginalAdmissionReached(RuntimeError):pass
def loader(experiment,arm,inventory):
 shadow=[name for name in sys.modules if name=='hiercp' or name.startswith('hiercp.')]
 assert not shadow,shadow
 assert (experiment,arm,inventory)==('UNIT_native','native','UNIT_inventory')
 raise OriginalAdmissionReached('UNIT archive boundary reached before population/output')
with patch.object(local_cnn_device,'select',return_value=None),patch.object(upper,'_load_geometry_inputs',side_effect=loader),patch.object(cp,'admit_pin',return_value=({},dict(native_experiment='UNIT_native'))):
 try:cp.prepare_bank(pin_path='UNIT_pin',inventory_path='UNIT_inventory',baseline_preprocessed='UNIT_pre',output='UNIT_never_created',gpu=1)
 except OriginalAdmissionReached:print(json.dumps(dict(admission_before_population=True,preactivation_hiercp_modules=[])))
 else:raise AssertionError('Original source admission boundary bypassed')
'''
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).resolve().parents[1],
            capture_output=True,text=True,check=True)
        self.assertTrue(json.loads(result.stdout)['admission_before_population'])

    def test_action_specific_required_inputs_and_single_gpu(self):
        from tools.run_v24_nnunet_cp import parse
        self.assertEqual(parse(['train','--native','UNIT.json']).gpu,1)
        recover=parse(['materialize-bank','--pin','UNIT_pin','--inventory','UNIT_inventory','--baseline-preprocessed','UNIT_pre',
            '--scores','UNIT_scores','--score-source-code','UNIT_old_code','--output','UNIT_new'])
        self.assertEqual(recover.action,'materialize-bank');self.assertEqual(str(recover.score_source_code),'UNIT_old_code')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):parse(['train','--native','UNIT.json','--gpu','5'])
            with self.assertRaises(SystemExit):parse(['prepare-bank','--pin','UNIT.json'])

    def test_CP_geometry_includes_zero_P_train_cases(self):
        from hiercp_v1x.v23_geometry import V23UpperGeometryCache
        split=split_fixture()
        population=SimpleNamespace(case=lambda case,count:SimpleNamespace(case_id=case,active_u_count=count))
        with patch.object(V23UpperGeometryCache,'__init__',return_value=None):
            geometry=cp.make_geometry(None,population,'UNIT',dict(workers=12,rss_gib_per_rank=192,
                geometry_resident_gib_per_rank=128,memoize_cpu_geometry=True,cache_stage_admission=True),None)
        geometry.train_cases=tuple(split['inner_train']);geometry.validation_cases=tuple(split['inner_val'])
        geometry.population=population;geometry._plan_stages={}
        cases,plans,count,selection=geometry._plans_for_stage(128,None)
        self.assertEqual(len(cases),105);self.assertEqual(len(plans),105)
        self.assertEqual(count,128);self.assertIsNone(selection)
        with self.assertRaises(ValueError):geometry._plans_for_stage(7,None)


class NativeCloneAdmission(unittest.TestCase):
    def trial(self,batch=2):
        measurement=dict(input_shape=[batch,1,128,128,128],loss=.6,gradient_finite=True,
            present_gradient_tensors=100,changed_parameter_tensors=100,encoder_and_decoder_updated=True,
            project_resource_snapshot=dict(actual_process_and_children_RSS_bytes=8*2**30))
        return dict(format='v24_actual_native_CP_full128_clone_calibration_trial_v1',debug=True,
            accepted=True,bank_sha256='a'*64,patch_size=[128]*3,cp_probability=.5,
            original_model_architecture_unchanged=True,production_updates=0,baseline_physical_batch=2,
            physical_batch=batch,project_resource_contract=cp.resource_contract(),
            measurements=[copy.deepcopy(measurement) for _ in range(3)],warmup=measurement,
            clone_optimizer_updates=4,native_CP_transport=dict(raw_events=2,native_support_voxels=20),
            full105_training_loader=True,ordinary26_validation_loader=True)

    def test_fixed_baseline_actual_native_updates_required(self):
        self.assertTrue(cp.validate_native_trial(self.trial(),baseline_batch=2,bank_sha256='a'*64)['accepted'])
        value=self.trial();value['measurements'].pop()
        with self.assertRaises(ValueError):cp.validate_native_trial(value,baseline_batch=2,bank_sha256='a'*64)

    def test_larger_CUDA_OOM_is_explicit_rejection_and_baseline_is_preserved(self):
        value=self.trial(4);value.update(accepted=False,rejection='actual_torch_CUDA_OutOfMemoryError')
        self.assertFalse(cp.validate_native_trial(value,baseline_batch=2,bank_sha256='a'*64)['accepted'])
        value['physical_batch']=2
        with self.assertRaises(ValueError):cp.validate_native_trial(value,baseline_batch=2,bank_sha256='a'*64)

    def test_other_errors_cannot_be_disguised_as_OOM(self):
        value=self.trial(4);value.update(accepted=False,rejection='ValueError')
        with self.assertRaises(ValueError):cp.validate_native_trial(value,baseline_batch=2,bank_sha256='a'*64)

    def test_nonfinite_disconnected_smaller_input_and_overbudget_rejected(self):
        mutations=(lambda x:x['measurements'][0].update(loss=float('nan')),
            lambda x:x['measurements'][0].update(encoder_and_decoder_updated=False),
            lambda x:x['measurements'][0].update(input_shape=[2,1,64,64,64]),
            lambda x:x['measurements'][0]['project_resource_snapshot'].update(actual_process_and_children_RSS_bytes=49*2**30))
        for mutate in mutations:
            value=self.trial();mutate(value)
            with self.assertRaises(ValueError):cp.validate_native_trial(value,baseline_batch=2,bank_sha256='a'*64)

    def test_missing_actual_CP_transport_rejected(self):
        value=self.trial();value['native_CP_transport']['native_support_voxels']=0
        with self.assertRaises(ValueError):cp.validate_native_trial(value,baseline_batch=2,bank_sha256='a'*64)

    def test_project_budget_is_explicit_not_host_available(self):
        budget=cp.resource_contract()
        self.assertEqual(budget['logical_CPU_cores'],4)
        self.assertEqual(budget['process_and_children_RSS_bytes'],48*2**30)
        self.assertEqual(budget['data_augmentation_workers'],4)
        self.assertTrue(budget['host_available_resources_are_not_project_allocation'])


if __name__=='__main__':unittest.main()
