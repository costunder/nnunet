"""CPU DEBUG fixtures for active-only dispatch; no model, CUDA or PBS is run."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from hiercp_v1x import basic_cp_single_parent_resources as resource
from hiercp_v1x.basic_cp_single_parent_resources import ARMS
from tools import run_basic_cp_single_parent as runner
from tools import start_basic_cp_direct_MIG as direct
from tools import start_basic_cp_single_parent_pbs as bootstrap


class ActiveContinuationDEBUGTests(unittest.TestCase):
    def setUp(self):
        self.workspace=Path(__file__).resolve().parents[1]
        self.temporary=tempfile.TemporaryDirectory(prefix='DEBUG_active_continuation_',dir=str(self.workspace))
        self.addCleanup(self.temporary.cleanup);self.root=Path(self.temporary.name).resolve()
        self.assertIn(self.workspace,self.root.parents)
        self.fixture_number=0;self.document_revision=0;self.fixture_root=self.root/'fixture_0'
        self.slots={'multi5':0,'multi7':6,'multi10':2,'candidates512':3,'candidates1024':4,'candidates2048':5}
        if os.name=='nt':
            # Managed Windows denies GetFinalPathName on freshly written JSON.
            # Keep actual production pin/digest and real byte hashes; only the
            # fixture path admission uses its already admitted own temp root.
            def fixture_exact(path,**kwargs):
                value=Path(path);self.assertTrue(value.is_absolute()and value.is_relative_to(self.root)and value.is_file())
                return value
            admitted=patch.object(resource,'exact',side_effect=fixture_exact);admitted.start();self.addCleanup(admitted.stop)

    def write(self,name,value):
        path=self.fixture_root/name;path.parent.mkdir(parents=True,exist_ok=True)
        data=value if isinstance(value,bytes)else json.dumps(value,sort_keys=True).encode()
        with path.open('xb')as stream:stream.write(data)
        return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest()}

    def fixture(self):
        # Each fixture is immutable and has fresh paths even on filesystems
        # with coarse stat timestamps. Production's SHA cache stays untouched.
        self.fixture_number+=1;self.fixture_root=self.root/('fixture_'+str(self.fixture_number))
        active=['candidates512','candidates1024','candidates2048'];evidence={}
        for arm in ('multi5','multi7','multi10'):
            output=self.fixture_root/arm;cases=['CPU_DEBUG_case_'+str(i)for i in range(26)]
            fold=output/'nnUNet_results'/'CPU_DEBUG_trainer'/'fold_0'
            best=self.write(str(Path(arm)/'nnUNet_results'/'CPU_DEBUG_trainer'/'fold_0'/'checkpoint_best.pth'),
                ('CPU_DEBUG_BEST_bytes_'+arm).encode())
            final=self.write(str(Path(arm)/'nnUNet_results'/'CPU_DEBUG_trainer'/'fold_0'/'checkpoint_final.pth'),
                ('CPU_DEBUG_FINAL_witness_bytes_'+arm).encode())
            reports={name:self.write(arm+'/'+name+'.json',{'CPU_DEBUG':True,'arm':arm,'report':name})
                for name in ('evidence','text','json')}
            archive=self.write(arm+'/archive.json',{'CPU_DEBUG':True,'complete':True})
            proof=self.write(arm+'/BEST26.json',dict(epochs=250,checkpoint='checkpoint_best.pth',
                evaluation_checkpoint_is_final=False,validation_cases=cases,best_sha256=best['sha256'],
                final_training_witness_sha256=final['sha256'],fold=str(fold)))
            status=self.write(arm+'/status.json',dict(arm=arm,stage='complete',status='COMPLETE',
                completed_epochs=250,evaluated_cases=26,full_native_training=True,
                full_native_evaluation=True,historical_metrics_complete=True,reports=reports,archive=archive))
            request=self.write(arm+'/request.json',dict(arm=arm,logical_slot=self.slots[arm],root=str(output),
                trainer='CPU_DEBUG_trainer',split={'outer_val':cases}))
            evidence[arm]=dict(status=status,training_and_BEST26=proof,BEST=best,FINAL=final,reports=reports,archive=archive,
                request=request,output_root=str(output),logical_slot=self.slots[arm])
        return dict(active_arms=active,retained_complete_arms=['blend50','multi5','multi7','multi10'],
            retained_completion_evidence=evidence,debug=False)

    def change_document(self,value,arm,key,update):
        old=value['retained_completion_evidence'][arm][key];document=json.loads(Path(old['path']).read_bytes())
        update(document)
        self.document_revision+=1
        value['retained_completion_evidence'][arm][key]=self.write(
            arm+'/changed_'+key+'_'+str(self.document_revision)+'.json',document)

    def test_legacy_six_continue_with_only_completed_blend50_retained(self):
        self.assertEqual(runner.continuation_arms({'retained_complete_arms':['blend50'],'debug':False}),ARMS)

    def test_exact_three_candidates_retain_original_completion_report_archive_bytes(self):
        value=self.fixture();before={str(p):p.read_bytes()for p in self.root.rglob('*')if p.is_file()}
        self.assertEqual(runner.continuation_arms(value),tuple(value['active_arms']))
        self.assertEqual(before,{str(p):p.read_bytes()for p in self.root.rglob('*')if p.is_file()})

    def test_active_list_must_be_nonempty_unique_canonical_and_catalog_only(self):
        for arms in ([],['candidates1024','candidates512'],['candidates512']*2,['blend50'],['unknown']):
            value=self.fixture();value['active_arms']=arms
            with self.subTest(arms=arms),self.assertRaises(ValueError):runner.continuation_arms(value)

    def test_omitted_arms_cannot_disappear_or_be_marked_complete_without_evidence(self):
        for failure in ('missing_retained','missing_proof','extra_proof','DEBUG'):
            value=self.fixture()
            if failure=='missing_retained':value['retained_complete_arms'].remove('multi7')
            if failure=='missing_proof':value['retained_completion_evidence'].pop('multi7')
            if failure=='extra_proof':value['retained_completion_evidence']['candidates512']=copy.deepcopy(value['retained_completion_evidence']['multi5'])
            if failure=='DEBUG':value['debug']=True
            with self.subTest(failure=failure),self.assertRaises(ValueError):runner.continuation_arms(value)

    def test_incomplete_training_or_statistics_cannot_exclude_an_experiment(self):
        for field,replacement in (('status','RUNNING'),('stage','historical_metrics'),('completed_epochs',249),
                ('evaluated_cases',25),('full_native_training',False),('full_native_evaluation',False),
                ('historical_metrics_complete',False),('arm','multi7')):
            value=self.fixture();self.change_document(value,'multi5','status',lambda status:status.update({field:replacement}))
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'has not completed'):
                runner.continuation_arms(value)

    def test_FINAL_or_incomplete_BEST26_never_becomes_retained_completion(self):
        for field,replacement in (('epochs',249),('checkpoint','checkpoint_final.pth'),
                ('evaluation_checkpoint_is_final',True),('validation_cases',['CPU_DEBUG_duplicate']*26)):
            value=self.fixture();self.change_document(value,'multi5','training_and_BEST26',lambda proof:proof.update({field:replacement}))
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'BEST26 completion'):
                runner.continuation_arms(value)

    def test_reports_and_archive_are_SHA_bound_to_actual_completed_status(self):
        for role in ('evidence','text','json','archive','request','status','training_and_BEST26','BEST','FINAL'):
            value=self.fixture();record=value['retained_completion_evidence']['multi5']
            reference=record['reports'][role]if role in record['reports']else record[role]
            Path(reference['path']).write_bytes(b'CPU_DEBUG changed exact artifact')
            with self.subTest(role=role),self.assertRaisesRegex(ValueError,'SHA changed'):
                runner.continuation_arms(value)
        value=self.fixture();record=value['retained_completion_evidence']['multi5']
        record['reports']['text']=self.write('unrelated_text.json',{'CPU_DEBUG':'different bytes'})
        with self.assertRaisesRegex(ValueError,'report bytes changed'):runner.continuation_arms(value)
        value=self.fixture();value['retained_completion_evidence']['multi5']['archive']=self.write('unrelated_archive.json',{'CPU_DEBUG':0})
        with self.assertRaisesRegex(ValueError,'archive marker changed'):runner.continuation_arms(value)

    def test_partial_inputs_cannot_bind_or_dispatch_a_completed_experiment(self):
        value=self.fixture();value.update(format=bootstrap.INPUT_FORMAT,canonical_code=str(runner.ROOT),
            output_root='/dev/shm/aicompetition06_basic_cp/experiments',
            source_snapshot=self.write('source.json',{'CPU_DEBUG':True}),
            VRAM_measurement_evidence=self.write('VRAM.json',{'CPU_DEBUG':True}),
            bindings=self.write('bindings.json',{arm:{'CPU_DEBUG':True}for arm in value['active_arms']}))
        self.assertIs(bootstrap.admit_inputs(value),value)
        value['bindings']=self.write('wrong_bindings.json',{arm:{}for arm in ARMS})
        with self.assertRaisesRegex(ValueError,'unfinished input bindings'):bootstrap.admit_inputs(value)

    def test_completed_arm_explicit_child_entry_is_rejected_without_rewriting_completion(self):
        value=self.fixture();value.update(root=str(self.root),experiments={arm:{}for arm in value['active_arms']},
            allocation={'path':'/CPU_DEBUG_allocation','sha256':'a'*64})
        allocation=dict(active_arms=value['active_arms'],execution_UUID_by_arm={arm:'MIG-'+arm for arm in value['active_arms']})
        actual_stat=Path.stat
        def stat(path,*args,**kwargs):
            result=actual_stat(path,*args,**kwargs)
            if path==self.root:
                class OwnedRoot:st_uid=41833;st_mode=result.st_mode
                return OwnedRoot()
            return result
        with patch.object(sys,'argv',['CPU_DEBUG_runner','--cohort',str(self.root/'cohort.json'),
                '--cohort-sha256','a'*64,'--arm','multi5','--execute']), \
                patch.object(runner,'pin',side_effect=lambda reference,**kwargs:Path(reference['path'])), \
                patch.object(runner,'read',side_effect=[value,allocation]), \
                patch.object(runner,'validate_cohort',return_value=value), \
                patch.object(runner,'continuation_arms',return_value=tuple(value['active_arms'])), \
                patch.object(Path,'stat',stat), \
                patch.object(runner,'record_child_failure')as failure, \
                patch.object(runner.psutil,'Process')as actor, \
                patch.object(runner,'train_arm')as train:
            with self.assertRaisesRegex(ValueError,'Completed or unbound'):runner.main()
            failure.assert_not_called();actor.assert_not_called();train.assert_not_called()

    def test_partial_prepare_receives_exact_original_inputs_reference(self):
        value=self.fixture();value.update(source_snapshot={'path':'/DEBUG/source','sha256':'a'*64},
            bindings={'path':'/DEBUG/bindings','sha256':'b'*64},_inputs_reference={'path':'/DEBUG/inputs','sha256':'c'*64},
            output_root='/dev/shm/aicompetition06_basic_cp/experiments',
            shared_preprocessed_dataset_directory='/DEBUG/shared131',evaluator_root=str(runner.ROOT),project='/DEBUG/project')
        with patch.object(direct,'digest',return_value='d'*64),patch.object(direct.subprocess,'run')as execute:
            direct.prepare_and_run(value,Path('/DEBUG/real_allocation'))
        command=execute.call_args_list[0].args[0]
        self.assertEqual(command[command.index('--continuation-inputs')+1],'/DEBUG/inputs')
        self.assertEqual(command[command.index('--continuation-inputs-sha256')+1],'c'*64)
        self.assertEqual(command[command.index('--bindings')+1],'/DEBUG/bindings')


if __name__=='__main__':unittest.main()
