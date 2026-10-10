"""CPU UNIT metadata snapshots; no CT, model forward or production resume.

Fixtures are retained in an owned outputs directory. Tiny checkpoint tensors
exercise exact serialization/state guards, never substitute for actual trials.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

os.environ['CUDA_VISIBLE_DEVICES']=''
import torch

from hiercp_v1x import v24_training_continuation as runtime
from hiercp_v1x.u_bridge_training import digest
from tools import run_v24_all_p as cli


ROOT=Path(__file__).resolve().parents[1]
RUNTIME_UNIT={'format':'explicit_CPU_UNIT_metadata_runtime','GPU_work_performed':False}


def _json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8')


def _save(path,value):
    value=copy.deepcopy(value)
    value['content_sha256']=digest({k:v for k,v in value.items() if k!='content_sha256'})
    torch.save(value,path)


def _unit_main(argv=None):
    from hiercp_v1x.v24_factory import V24NativeInputs,build_runtime
    result=build_runtime('explicit_CPU_UNIT_no_forward')
    result[-1]()
    return 'CPU_UNIT_RETURN'


class ExactContinuationDebug(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/'outputs'/('v24_continuation_CPU_UNIT_'+uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.old=self.root/'old';self.old.mkdir()
        self.new=self.root/'fresh'
        self.inventory=self.root/'UNIT_inventory.json'
        _json(self.inventory,{'debug':'CPU_UNIT_metadata_only','no_real_samples':True})
        self.config=ROOT/'config/v24_gpu5_GT_blind.json'
        self.args=SimpleNamespace(mode='train',config=self.config,native_experiment=self.root/'UNIT_native',
            inventory=self.inventory,input_cache=self.root/'UNIT_cache',output=self.new,gpu=5,
            stunet_checkpoint=None,cpu_affinity=[0,1,2,3],resume=self.new/'training/checkpoint_latest.pt')
        self.request=cli.request(self.args,json.loads(self.config.read_text()))
        self.train=[f'UNIT_train_{i:02d}' for i in range(65)]
        self.val=[f'UNIT_val_{i:02d}' for i in range(21)]
        config=copy.deepcopy(self.request['config'])
        self.calibration=dict(format='explicit_CPU_UNIT_NOT_native_calibration',
            request_sha256=self.request['request_sha256'],physical_GPU=5,
            initial_state_sha256='UNIT_initial_model',model_contract={'debug':'metadata_only'})
        config['v24_runtime']['batch_calibration']=self.calibration
        binding=dict(identity=self.request,config=config,train_cases=self.train,val_cases=self.val,
            physical_patient_batch=4,physical_candidate_batch=32,epochs=40,
            initial_state_sha256=self.calibration['initial_state_sha256'],
            model_contract=self.calibration['model_contract'])
        self.owner=dict(binding=binding,identity_sha256=digest(binding))
        full=dict(phase='full_validation',active_u=128,
            metrics=dict(per_P_patient_mrr=.4,per_P_patient_top1=.3,patient_balanced_pair_loss=.5))
        winner=dict(epoch=2,updates=34,selection_key=[.4,.3,-.5],selected_by='fixed_full128_validation_only')
        policy=config['v24_runtime']['curriculum']
        state=dict(epoch=4,history=[{'epoch':i,'debug':'CPU_UNIT'} for i in (1,2,3)],
            phase='full_validation',status='RUNNING',updates=68,attempts=68,overflows=0,
            curriculum=dict(policy=policy,active_u=7,stage_epochs=3),
            train_order=self.train,train_rows=[{'case_id':case} for case in self.train],train_position=65,
            evaluation_position=0,evaluation_rows=[],best=winner)
        self.latest=dict(format='v24_GT_blind_all_P_patient_balanced_training_v1',
            identity_sha256=self.owner['identity_sha256'],model={'UNIT.weight':torch.arange(7,dtype=torch.float32)},
            optimizer={'state':{0:{'step':torch.tensor(68.),'exp_avg':torch.ones(7),'exp_avg_sq':torch.ones(7)}}},
            scheduler={'last_epoch':3},scaler={'scale':8.},rank_rng=[{'UNIT_CPU_rng':torch.arange(4)}],
            shuffle_generator=torch.arange(4,dtype=torch.uint8),state=state)
        self.best=copy.deepcopy(self.latest)
        self.best['state'].update(epoch=2,history=[{'epoch':1,'debug':'CPU_UNIT'}],phase='epoch_completion',updates=34,attempts=34,
            pending_epoch_completion={'full_validation':full})
        self.best['scheduler']['last_epoch']=1
        self.best['optimizer']['state'][0]['step']=torch.tensor(34.)
        _json(self.old/'request.json',self.request)
        _json(self.old/'calibration.json',self.calibration)
        _json(self.old/'training/training_identity.json',self.owner)
        _save(self.old/'training/checkpoint_latest.pt',self.latest)
        _save(self.old/'training/checkpoint_best.pt',self.best)
        (self.old/'training/STOP_AFTER_BATCH').write_text('old pause must be preserved only here')
        (self.old/'training/curve.jsonl').write_text('{"debug":"CPU_UNIT_only"}\n')
        _json(self.old/'training/execution_contract.json',{'debug':'UNIT_original_contract'})
        _json(self.old/'resources_train_CPU_UNIT.json',{'debug':'no_GPU'})

    def prepare(self):
        with patch.object(runtime,'_runtime_proof',return_value=RUNTIME_UNIT):
            return runtime.prepare_continuation(self.args,source_output=self.old,source_code=ROOT)

    def admit(self):
        with patch.object(runtime,'_runtime_proof',return_value=RUNTIME_UNIT):
            return runtime.admit_continuation(self.args)

    def test_exact_copy_excludes_pause_preserves_six_states_best_and_reports(self):
        before=runtime.inspect_continuation(self.old,ROOT)
        receipt=self.prepare()
        self.assertTrue(receipt.is_file())
        self.assertTrue((self.old/'training/STOP_AFTER_BATCH').exists())
        self.assertFalse((self.new/'training/STOP_AFTER_BATCH').exists())
        for name in ('checkpoint_latest.pt','checkpoint_best.pt','training_identity.json','curve.jsonl','execution_contract.json'):
            self.assertEqual((self.old/'training'/name).read_bytes(),(self.new/'training'/name).read_bytes())
        _,value=self.admit()
        self.assertEqual(value['source']['latest']['numerical_state_sha256'],before['latest']['numerical_state_sha256'])
        self.assertEqual(value['source']['latest']['state_sha256'],before['latest']['state_sha256'])
        self.assertEqual(value['source']['BEST']['raw_sha256'],before['BEST']['raw_sha256'])
        self.assertEqual(runtime.inspect_continuation(self.old,ROOT),before)

    def test_gpu5_partial_validation_and_gpu6_start_phase_preserved(self):
        for position,epoch,updates in ((8,1,17),(0,4,68)):
            with self.subTest(cursor=position):
                saved=copy.deepcopy(self.latest)
                saved['state'].update(epoch=epoch,history=[{'epoch':i} for i in range(1,epoch)],
                    updates=updates,attempts=updates,evaluation_position=position,
                    evaluation_rows=[{'case_id':case} for case in self.val[:position]])
                saved['scheduler']['last_epoch']=epoch-1
                saved['optimizer']['state'][0]['step']=torch.tensor(float(updates))
                _save(self.old/'training/checkpoint_latest.pt',saved)
                _,proof=runtime._checkpoint(self.old/'training/checkpoint_latest.pt',self.owner)
                self.assertEqual((proof['epoch'],proof['updates'],proof['evaluation_position']),(epoch,updates,position))

    def test_incomplete_physical_cursor_and_duplicate_patient_rejected(self):
        changed=copy.deepcopy(self.latest)
        changed['state'].update(evaluation_position=1,evaluation_rows=[{'case_id':self.val[0]}])
        _save(self.old/'training/checkpoint_latest.pt',changed)
        with self.assertRaisesRegex(ValueError,'complete physical batch'):
            runtime._checkpoint(self.old/'training/checkpoint_latest.pt',self.owner)
        changed=copy.deepcopy(self.latest);changed['state']['train_order'][0]=changed['state']['train_order'][1]
        _save(self.old/'training/checkpoint_latest.pt',changed)
        with self.assertRaises(ValueError):runtime._checkpoint(self.old/'training/checkpoint_latest.pt',self.owner)

    def test_content_corruption_optimizer_history_and_curriculum_changes_fail(self):
        corrupt=copy.deepcopy(self.latest);corrupt['content_sha256']='0'*64
        torch.save(corrupt,self.old/'training/checkpoint_latest.pt')
        with self.assertRaisesRegex(ValueError,'content/identity'):runtime.inspect_continuation(self.old,ROOT)
        for kind in ('optimizer','curriculum'):
            saved=copy.deepcopy(self.latest)
            if kind=='optimizer':saved['optimizer']['state'][0]['step']=torch.tensor(67.)
            else:saved['state']['curriculum']['policy']['increment_u']+=1
            _save(self.old/'training/checkpoint_latest.pt',saved)
            with self.assertRaises(ValueError):runtime.inspect_continuation(self.old,ROOT)

    def test_actual_best_metric_not_only_winner_metadata_required(self):
        changed=copy.deepcopy(self.best)
        changed['state']['pending_epoch_completion']['full_validation']['metrics']['per_P_patient_mrr']=.41
        _save(self.old/'training/checkpoint_best.pt',changed)
        with self.assertRaisesRegex(ValueError,'BEST'):runtime.inspect_continuation(self.old,ROOT)

    def test_original_request_output_resume_independence_and_changed_cache_rejected(self):
        other=copy.copy(self.args);other.output=self.root/'other_output';other.resume=self.root/'other_checkpoint.pt'
        self.assertEqual(cli.request(other,self.request['config']),self.request)
        other.input_cache=self.root/'different_cache'
        with patch.object(runtime,'_runtime_proof',return_value=RUNTIME_UNIT),self.assertRaisesRegex(ValueError,'request identity'):
            runtime.prepare_continuation(other,source_output=self.old,source_code=ROOT)

    def test_existing_output_and_source_namespace_refused_without_overwrite(self):
        self.new.mkdir();marker=self.new/'preserve';marker.write_bytes(b'owned original')
        with self.assertRaises(FileExistsError):self.prepare()
        self.assertEqual(marker.read_bytes(),b'owned original')
        self.args.output=self.old/'nested'
        with self.assertRaises(FileExistsError):self.prepare()

    def test_admission_fails_on_old_source_or_prepared_bytes_mutation(self):
        self.prepare()
        (self.old/'training/curve.jsonl').write_text('old source changed')
        with self.assertRaisesRegex(ValueError,'preserved source'):self.admit()
        (self.new/'training/curve.jsonl').write_text('fresh snapshot changed')
        with self.assertRaisesRegex(ValueError,'continuation bytes'):self.admit()

    def test_original_scientific_source_manifest_change_is_rejected(self):
        with patch.object(cli,'FILES',tuple(cli.FILES[:-1])):
            with self.assertRaisesRegex(ValueError,'source manifest'):runtime.inspect_continuation(self.old,ROOT)

    def test_exact_copy_refuses_existing_destination(self):
        source=self.root/'copy_source';destination=self.root/'copy_destination'
        source.write_bytes(bytes(range(255)));destination.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):runtime._copy_exact(source,destination)
        self.assertEqual(destination.read_bytes(),b'preserve')

    def test_private_import_hook_preserves_code_globals_and_actual_factory(self):
        from hiercp_v1x import v24_factory
        old=v24_factory.build_runtime
        original_builtins=_unit_main.__globals__.get('__builtins__')
        builder=Mock(return_value=(None,None,None,None,None,Mock()))
        clone=runtime._clone_main(_unit_main,lambda actual:builder)
        self.assertIs(clone.__code__,_unit_main.__code__)
        self.assertEqual(clone([]),'CPU_UNIT_RETURN')
        builder.assert_called_once_with('explicit_CPU_UNIT_no_forward')
        self.assertIs(v24_factory.build_runtime,old)
        self.assertIs(_unit_main.__globals__.get('__builtins__'),original_builtins)

    def test_argv_roundtrip_keeps_original_resume_config_gpu_affinity(self):
        restored=cli.parse(runtime._argv(self.args,self.args.resume))
        for field in ('mode','config','native_experiment','inventory','input_cache','output','gpu','cpu_affinity','resume'):
            self.assertEqual(getattr(restored,field),getattr(self.args,field))

    def test_binding_failure_closes_own_runtime_before_original_finally(self):
        self.prepare()
        from hiercp_v1x import v24_factory,v24_memory_runtime,v24_preparation_runtime
        close=Mock();net=torch.nn.Linear(2,2)
        result=(net,SimpleNamespace(),None,{},None,close)
        with patch.object(runtime,'_runtime_proof',return_value=RUNTIME_UNIT),\
                patch.object(cli,'main',_unit_main),patch.object(v24_factory,'build_runtime',return_value=result),\
                patch.object(v24_preparation_runtime,'install_runtime',return_value={'CPU_UNIT':True}),\
                patch.object(v24_memory_runtime,'install_memory_runtime',return_value={'CPU_UNIT':True}),\
                patch.object(v24_memory_runtime,'bind_memory_runtime',side_effect=ValueError('UNIT rejected binding')):
            with self.assertRaisesRegex(ValueError,'rejected binding'):runtime.run_continuation(self.args)
        close.assert_called_once()


if __name__=='__main__':unittest.main()
