"""Stdlib metadata UNIT tests; no CT, tensors, neural scores, or GPU work.

Evaluation values are artificial UNIT validation sentinels, never model
predictions or research results. Sources/files belong to unique tiny UNIT
workspaces and are only used to test integrity and publication refusal.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
from zipfile import ZipFile

from tools import collect_v17_crossed_smoke as collector


ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_metadata():
    """Artificial UNIT metadata only; no model or metric computation."""
    identities=[f'UNIT-recipient:{index}' for index in range(136)]
    scores=[dict(record_id=name,candidate_key=[index,0,0],score=index/136.,observed=int(index<8))
        for index,name in enumerate(identities)]
    contract=dict(format='hiercp_transition_whole_case_scoring_v1',scored_record_ids=identities,
        l0_only_chunking=True,upper_chunking=False,upper_execution='single_joint_case',query_GT_in_forward=False,
        upper_invocations=1,l0_batch_size=16,
        cases=[dict(case_id='UNIT-recipient',records=136,l0_chunks=9,upper_invocations=1)])
    return dict(format='hiercp_transition_whole128_evaluation_v1',debug=True,
        task='native_observed_P_vs_unobserved_U',full_128_U_per_case=True,production_full_inner_val=False,
        quality_verified=False,full_training_complete=False,full_evaluation=False,
        execution_contract_bound=True,scoring_callbacks_executed=True,raw_CT_execution_verified=True,
        callback_internal_execution_verified=True,support_records=401,C_curriculum_support_reused=False,
        GT_is_donor_compatibility=False,original_eight_candidate_metrics=False,
        cohort=dict(case_ids=['UNIT-recipient'],records=136,observed_P=8,unobserved_U=128,debug=True),
        denominators=dict(cases=1,rank_evaluable_cases=1,zero_P_cases=0,observed_P=8,unobserved_U=128,
            P_U_pairs=1024,case_hit_at_1=1,observed_micro_recall=8),
        metrics=dict(case_first_P_mrr=.5,case_hit_at_1=0.,observed_micro_recall_at_1=0.,
            observed_micro_recall_at_5=0.,observed_micro_recall_at_10=0.,P_U_pair_win_rate=0.,
            P_U_pair_tie_rate=0.,P_U_softplus_loss=1.,score_std=.1,score_min=0.,score_max=1.),
        cases=[dict(case_id='UNIT-recipient',records=136,observed_P=8,unobserved_U=128,
            scored_record_ids=identities,ordered_record_ids=list(reversed(identities)),case_scores=scores)],
        scoring_contract=contract,model_sha256='a'*64,initial_upper_sha256='b'*64,scores_sha256='c'*64,
        support_GT='native observed P/unobserved U',
        upper_support_policy='full inner-train bank; exclude query patient on both support recipient and donor sides')


class EvaluationRefusals(unittest.TestCase):
    def test_complete_DEBUG_metadata_is_accepted(self):
        self.assertIsInstance(collector.validate_evaluation(evaluation_metadata()),dict)

    def test_execution_population_and_truth_contract_failures_are_refused(self):
        changes=[{'debug':False},{'full_128_U_per_case':False},{'support_records':400},
            {'raw_CT_execution_verified':False},{'callback_internal_execution_verified':False},
            {'scoring_callbacks_executed':False},{'execution_contract_bound':False},
            {'quality_verified':True},{'C_curriculum_support_reused':True}]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                collector.validate_evaluation({**evaluation_metadata(),**change})
        for key,value in (('upper_chunking',True),('upper_invocations',2),('query_GT_in_forward',True),
                ('upper_execution','separate_query_calls')):
            metadata=evaluation_metadata();metadata['scoring_contract'][key]=value
            with self.subTest(contract=key),self.assertRaises(ValueError):collector.validate_evaluation(metadata)

    def test_missing_duplicate_nonfinite_and_wrong_class_scores_are_refused(self):
        edits=[lambda value:value['cases'][0]['case_scores'].pop(),
            lambda value:value['cases'][0]['case_scores'][1].update(record_id='UNIT-recipient:0'),
            lambda value:value['cases'][0]['case_scores'][0].update(score=float('nan')),
            lambda value:value['cases'][0]['case_scores'][0].update(score=float('inf')),
            lambda value:value['cases'][0]['case_scores'][0].update(score=True),
            lambda value:value['cases'][0]['case_scores'][0].update(observed=0),
            lambda value:value['denominators'].update(observed_P=7)]
        for ordinal,edit in enumerate(edits):
            value=evaluation_metadata();edit(value)
            with self.subTest(edit=ordinal),self.assertRaises(ValueError):collector.validate_evaluation(value)


class SourceIntegrityAndPublication(unittest.TestCase):
    def setUp(self):
        self.base=ROOT/'work'/('v17_collector_metadata_UNIT_'+uuid.uuid4().hex)
        self.workspace=self.base/'workspace';self.workspace.mkdir(parents=True)
        module=self.workspace/'hiercp_v1x/UNIT_source.py';module.parent.mkdir();module.write_text('UNIT_METADATA_ONLY=True\n',encoding='utf8')
        config=self.workspace/'config/v17_crossed_training.json';config.parent.mkdir()
        config.write_text(json.dumps(dict(UNIT_metadata_only=True,shared=dict(hidden_dim=128,heads=4,
            upper_task_layers=2,upper_alignment_layers=2,temperature=.2)))+'\n',encoding='utf8')
        source=self.workspace/'work/UNIT_verified_snapshot';source.mkdir(parents=True)
        archive=self.workspace/'versions/v1/pipeline_v1_source.zip';archive.parent.mkdir(parents=True)
        verified={}
        with ZipFile(archive,'w') as bundle:
            for ordinal in range(12):
                name=f'hiercp/UNIT_{ordinal:02d}.py';value=f'UNIT_SOURCE_ID={ordinal}\n'
                path=source/name;path.parent.mkdir(exist_ok=True);path.write_text(value,encoding='utf8')
                bundle.writestr(name,value);verified[name]=sha(path)
            path=source/'config/train.json';path.parent.mkdir();path.write_text('{"UNIT_only":true}',encoding='utf8')
            bundle.writestr('config/train.json',path.read_bytes());verified['config/train.json']=sha(path)
        self.manifest=dict(sources={'hiercp_v1x/UNIT_source.py':sha(module),
            'config/v17_crossed_training.json':sha(config)},
            source=dict(source=str(source),archive_sha256=sha(archive),verified_files=verified))
        self.module=module;self.config=config;self.source=source;self.archive=archive

    def test_all_listed_source_and_config_bytes_are_required(self):
        self.assertIsInstance(collector.validate_sources(self.manifest,self.workspace),dict)
        for path in (self.module,self.config):
            original=path.read_bytes();path.write_bytes(original+b'UNIT_CHANGED_BYTES')
            with self.subTest(path=path.name),self.assertRaises(ValueError):
                collector.validate_sources(self.manifest,self.workspace)
            path.write_bytes(original)

    def test_original_archive_and_verified_snapshot_bytes_are_required(self):
        original=self.archive.read_bytes();self.archive.write_bytes(original+b'UNIT_CHANGED_ARCHIVE')
        with self.assertRaises(ValueError):collector.validate_sources(self.manifest,self.workspace)
        self.archive.write_bytes(original)
        (self.source/'hiercp/UNIT_00.py').write_text('UNIT_CHANGED_ORIGINAL=True',encoding='utf8')
        with self.assertRaises(ValueError):collector.validate_sources(self.manifest,self.workspace)

    def test_source_and_original_proof_path_traversal_are_refused(self):
        escape=self.base/'UNIT_escape.py';escape.write_text('UNIT_OUTSIDE_WORKSPACE=True',encoding='utf8')
        value=copy.deepcopy(self.manifest);value['sources']['../UNIT_escape.py']=sha(escape)
        with self.assertRaises(ValueError):collector.validate_sources(value,self.workspace)
        outside=self.source.parent/'UNIT_outside_original.py';outside.write_text('UNIT_OUTSIDE_SNAPSHOT=True',encoding='utf8')
        value=copy.deepcopy(self.manifest);value['source']['verified_files']['../UNIT_outside_original.py']=sha(outside)
        with self.assertRaises(ValueError):collector.validate_sources(value,self.workspace)

    def test_collection_failure_creates_no_output_or_partial_results(self):
        output=self.base/'must_not_be_created'
        with patch.object(collector,'read_arm',side_effect=ValueError('UNIT invalid actual arm report')):
            with self.assertRaisesRegex(ValueError,'UNIT invalid actual arm'):
                collector.collect(self.base/'C',self.base/'D',self.base/'preparation',output,workspace=self.workspace)
        self.assertFalse(output.exists())

    def test_late_preparation_failure_and_cross_arm_identity_mismatch_create_no_output(self):
        arm=dict(source=dict(execution_sources={'UNIT_source.py':'a'*64}),
            evaluation=dict(initial_upper_sha256='b'*64),architecture={},
            native_inventory_sha256='c'*64,scope_contract='d'*64,reused_prepared_cache=None)
        output=self.base/'late_failure_must_not_create_output'
        with patch.object(collector,'read_arm',return_value=(arm,[])),\
                patch.object(collector,'read_preparation',side_effect=ValueError('UNIT invalid preparation coverage')):
            with self.assertRaisesRegex(ValueError,'UNIT invalid preparation'):
                collector.collect(self.base/'C',self.base/'D',self.base/'preparation',output,workspace=self.workspace)
        self.assertFalse(output.exists())
        other=copy.deepcopy(arm);other['evaluation']['initial_upper_sha256']='c'*64
        with patch.object(collector,'read_arm',side_effect=[(arm,[]),(other,[])]),\
                patch.object(collector,'read_preparation',return_value=({'input_L0_contract':{}},[])):
            with self.assertRaisesRegex(ValueError,'initial common upper bytes differ'):
                collector.collect(self.base/'C',self.base/'D',self.base/'preparation',output,workspace=self.workspace)
        self.assertFalse(output.exists())

    def test_existing_output_is_refused_and_original_bytes_preserved(self):
        output=self.base/'existing_result';output.mkdir();sentinel=output/'UNIT_original.json';sentinel.write_bytes(b'UNIT_original_result_bytes')
        arm=dict(source=dict(execution_sources={'UNIT_source.py':'a'*64}),
            evaluation=dict(initial_upper_sha256='b'*64),architecture={},
            native_inventory_sha256='c'*64,scope_contract='d'*64,reused_prepared_cache=None)
        with patch.object(collector,'read_arm',return_value=(arm,[])),\
                patch.object(collector,'read_preparation',return_value=({'input_L0_contract':{},
                    'input_inventory_sha256':'c'*64,'scope_contract':'d'*64},[])),\
                self.assertRaisesRegex(ValueError,'destination already exists'):
            collector.collect(self.base/'C',self.base/'D',self.base/'preparation',output,workspace=self.workspace)
        self.assertEqual(sentinel.read_bytes(),b'UNIT_original_result_bytes')
        self.assertEqual([path.name for path in output.iterdir()],['UNIT_original.json'])


class ResumedArmMetadata(unittest.TestCase):
    """Artificial execution receipts only; fake checkpoint bytes are never loaded."""
    def setUp(self):
        SourceIntegrityAndPublication.setUp(self)

    def write_json(self,path,value,mtime=None):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,allow_nan=False),encoding='utf8')
        if mtime is not None:os.utime(path,(mtime,mtime))

    def write_jsonl(self,path,rows,mtime=None):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows),encoding='utf8')
        if mtime is not None:os.utime(path,(mtime,mtime))

    def arm_fixture(self,arm):
        root=self.base/('UNIT_'+arm+'_arm');training=root/'training';training.mkdir(parents=True)
        manifest={**copy.deepcopy(self.manifest), 'arm':arm,'debug':True,
            'settings':dict(debug_updates=2,workers=4),'hardware':dict(UNIT_metadata_only=True),
            'native_inventory_sha256':'c'*64,'scope':'d'*64}
        self.write_json(root/'manifest.json',manifest)
        constructed=dict(arm=arm,debug=True,stage='constructed_crossed_model',
            upper=dict(hidden_dim=128,heads=4,L1_layers=2,L2_layers=2),
            local_architecture=dict(channels=[12,24,32],hidden_dim=128),
            actual_common_train_observations=401,actual_common_validation_observations=136,
            worker_count=4,total_parameters=1000,trainable_parameters=1000,
            UNIT_metadata_not_a_neural_execution=True)
        self.write_jsonl(root/'model_execution.jsonl',[constructed,copy.deepcopy(constructed)])
        self.write_jsonl(root/'resources.jsonl',[dict(UNIT_metadata_only=True)])
        checkpoint=training/'checkpoint_latest.pt'
        checkpoint.write_bytes(b'UNIT_STORAGE_SENTINEL_NOT_A_TENSOR_OR_MODEL')
        first=evaluation_metadata();first['model_sha256']='d'*64
        final=evaluation_metadata();final['model_sha256']='e'*64
        first_evaluation=root/'UNIT_first_evaluation.json';final_evaluation=root/'UNIT_final_evaluation.json'
        self.write_json(first_evaluation,first);self.write_json(final_evaluation,final)
        if arm=='C':
            report=dict(format='crossed_native_C_training_state_v1',arm='C',status='DEBUG_COMPLETE',debug=True,
                updates=2,actual_CUDA=True,actual_raw_CT=True,full_training=False,full_evaluation=False,
                quality_verified=False,source_preserved=True,missing_gradients=[],run_identity_sha256='f'*64,
                final_DEBUG_evaluation=dict(common=dict(evaluation_artifact=str(first_evaluation))),
                checkpoint=str(checkpoint),input_spec=dict(UNIT_metadata_only=True),
                execution=dict(epochs=40,physical_sample_batch=2,workers=4))
            old_path=training/'c_training_report_first.json';final_path=training/'c_training_report_resumed.json'
            self.write_json(old_path,report,100.)
            report['final_DEBUG_evaluation']['common']['evaluation_artifact']=str(final_evaluation)
            self.write_json(final_path,report,200.)
            updates=[]
            for step in (1,2):
                updates.append(dict(status='OPTIMIZER_UPDATED',update=step,loss=1.,
                    gradient=dict(finite=True,nonfinite_groups=[],norms={name:.1 for name in collector.C_MODULES}),
                    parameter_updates={name:dict(L2_change=.01,changed_parameter_tensors=1) for name in collector.C_MODULES}))
            self.write_jsonl(training/'c_updates_first.jsonl',updates,100.)
            self.write_jsonl(training/'c_updates_resumed.jsonl',[],200.)
        else:
            report=dict(debug=True,actual_CT=True,actual_CUDA=True,updates=2,full_training=False,
                full_evaluation=False,quality_verified=False,
                optimizer_weight_changes={name:.01 for name in collector.D_MODULES},
                initial_neural_sha256='a'*64,final_neural_sha256='b'*64,evaluation_artifact=str(final_evaluation))
            self.write_jsonl(training/'DEBUG_neural_execution.jsonl',[report])
            updates=[]
            for step in (1,2):
                gradients={name:.1 for name in collector.D_MODULES}
                gradients.update(trainable_parameter_tensors=10,parameter_tensors_with_gradient=10,
                    missing_parameter_gradients=[])
                updates.append(dict(step=step,loss=1.,gradients=gradients,
                    physical_observations=2,local_graphs=4))
            self.write_jsonl(training/'updates.jsonl',updates)
        return root

    def test_identical_resumed_constructors_reports_and_zero_update_journal_are_accepted(self):
        root=self.arm_fixture('C');summary,files=collector.read_arm(root,'C',self.workspace)
        self.assertEqual(summary['successful_optimizer_updates'],2)
        self.assertEqual(summary['evaluation']['model_sha256'],'e'*64)
        self.assertFalse(summary['checkpoint']['tensor_deserialized'])
        self.assertFalse(summary['checkpoint']['copied'])
        self.assertIn(root/'training/c_updates_resumed.jsonl',files)
        self.assertIn(root/'training/c_training_report_first.json',files)
        self.assertTrue(all(path.suffix in ('.json','.jsonl') for path in files))

    def test_changed_resumed_model_configuration_is_refused(self):
        root=self.arm_fixture('C');path=root/'model_execution.jsonl'
        rows=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
        rows[1]['local_architecture']['channels']=[12,24,64]
        self.write_jsonl(path,rows)
        with self.assertRaises(ValueError):collector.read_arm(root,'C',self.workspace)

    def test_third_successful_update_in_any_C_invocation_is_refused(self):
        root=self.arm_fixture('C')
        first=json.loads((root/'training/c_updates_first.jsonl').read_text(encoding='utf8').splitlines()[0])
        first['update']=3
        self.write_jsonl(root/'training/c_updates_extra.jsonl',[first],300.)
        with self.assertRaisesRegex(ValueError,'two successful C optimizer updates'):
            collector.read_arm(root,'C',self.workspace)

    def test_every_earlier_C_report_must_preserve_sources_and_completed_contract(self):
        root=self.arm_fixture('C');path=root/'training/c_training_report_first.json'
        value=json.loads(path.read_text(encoding='utf8'));value['source_preserved']=False
        self.write_json(path,value,100.)
        with self.assertRaises(ValueError):collector.read_arm(root,'C',self.workspace)

    def test_D_core_gradient_receipt_accepts_count_metadata_and_complete_parameter_connectivity(self):
        root=self.arm_fixture('D');summary,files=collector.read_arm(root,'D',self.workspace)
        self.assertEqual(summary['successful_optimizer_updates'],2)
        self.assertFalse(summary['checkpoint']['tensor_deserialized'])
        self.assertIn(root/'training/updates.jsonl',files)

    def test_D_missing_or_inconsistent_parameter_gradient_coverage_is_refused(self):
        root=self.arm_fixture('D');path=root/'training/updates.jsonl'
        original=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
        changes=[{'trainable_parameter_tensors':0},{'trainable_parameter_tensors':True},
            {'parameter_tensors_with_gradient':9},{'missing_parameter_gradients':['UNIT_disconnected_parameter']},
            {collector.D_MODULES[0]:0.}]
        for change in changes:
            rows=copy.deepcopy(original);rows[0]['gradients'].update(change);self.write_jsonl(path,rows)
            with self.subTest(change=change),self.assertRaises(ValueError):
                collector.read_arm(root,'D',self.workspace)


if __name__=='__main__':unittest.main()
